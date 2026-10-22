"""The work behind each MCP tool. Plain functions so they can be tested without a client."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from . import edit, media
from .paths import clean_path, resolve_media_path
from .detect import TargetDetector, detect_in_frames, postprocess
from .workspace import (
    Project,
    ProjectError,
    create_project,
    list_images,
    load_image_bgra,
    load_project,
    parse_indices,
    parse_time,
)
from .workspace import IMAGE_EXTS

OPERATIONS = ("replace", "remove", "blur", "pixelate", "fill", "highlight")


def _select(project: Project, frame_indices=None, start_time=None, end_time=None, *, editing: bool) -> list[int]:
    """Frames picked by indices AND time range; editing tools are further limited to the edit window."""
    idx = parse_indices(frame_indices, project.frame_count)
    start, end = parse_time(start_time), parse_time(end_time)
    if (start is not None or end is not None) and is_photos(project):
        raise ProjectError("start_time/end_time apply to videos only; select photos with frame_indices")
    if start is not None or end is not None:
        span = set(project.frames_between(start, end))
        idx = [i for i in idx if i in span]
    if editing:
        window = project.edit_window_frames()
        if window is not None:
            allowed = set(window)
            idx = [i for i in idx if i in allowed]
    return idx


def _window_info(project: Project) -> dict | None:
    w = project.edit_window
    if w is None:
        return None
    frames = project.edit_window_frames()
    return {
        "start_time": w.get("start"),
        "end_time": w.get("end"),
        "frames": _ranges(frames),
        "frame_count": len(frames),
        "rule": "Only these frames can be edited; every editing tool skips frames outside this time window.",
    }


def _no_frames_error(project: Project, what: str) -> ProjectError:
    msg = f"no selected frames {what}"
    if project.edit_window:
        w = project.edit_window
        msg += f" inside the edit window ({w.get('start')}s - {w.get('end')}s)"
    return ProjectError(msg)


def _ranges(indices: list[int]) -> str:
    """[1,2,3,7,9,10] -> "1-3,7,9-10" (compact for LLM context)."""
    if not indices:
        return ""
    parts, start, prev = [], indices[0], indices[0]
    for i in indices[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if i is not None:
            start = prev = i
    return ",".join(parts)


def _encode(img: np.ndarray, max_width: int = 1024, quality: int = 85) -> bytes:
    if img.shape[1] > max_width:
        k = max_width / img.shape[1]
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ProjectError("failed to encode preview")
    return buf.tobytes()


# ---- split -------------------------------------------------------------


def split_video(
    video_path: str,
    target_image_path: str | None = None,
    fps: float | None = None,
    trim_start_time=None,
    trim_end_time=None,
    max_frames: int | None = None,
    max_width: int | None = None,
    image_format: str = "png",
    match_threshold: float = 0.7,
    detection_method: str = "auto",
    edit_start_time=None,
    edit_end_time=None,
) -> tuple[dict, bytes | None]:
    if image_format not in ("png", "jpg"):
        raise ProjectError("image_format must be png or jpg")
    video = resolve_media_path(video_path, "any", allow_dir=True)
    if video.is_dir() or video.suffix.lower() in IMAGE_EXTS:
        if any(v is not None for v in (trim_start_time, trim_end_time, edit_start_time, edit_end_time)):
            raise ProjectError("time options apply to videos only; this is a photo (or folder of photos)")
        return open_photos(str(video), target_image_path, match_threshold, detection_method)
    info = media.probe(video)
    rate = float(fps or info["fps"] or 25.0)
    start = parse_time(trim_start_time) or 0.0
    trim_end = parse_time(trim_end_time)
    duration = None
    if trim_end is not None:
        if trim_end <= start:
            raise ProjectError("trim_end_time must be greater than trim_start_time")
        duration = trim_end - start

    project = create_project(info, video.stem)
    try:
        count = media.extract_frames(
            info["path"], project.frames_dir, fps=rate, image_format=image_format,
            start=start or None, duration=duration, max_frames=max_frames, max_width=max_width,
        )
        if count == 0:
            raise ProjectError("no frames extracted; check trim_start_time/trim_end_time")
        first = cv2.imread(str(project.frames_dir / f"frame_000000.{image_format}"))
        project.manifest["extract"] = {
            "fps": rate, "start": start, "duration": round(count / rate, 3), "format": image_format,
            "width": int(first.shape[1]), "height": int(first.shape[0]),
        }
        project.manifest["frame_count"] = count
        project.log("split", frames=count, fps=rate)
        if edit_start_time is not None or edit_end_time is not None:
            _set_window(project, edit_start_time, edit_end_time)
        project.save()
    except Exception:
        shutil.rmtree(project.root, ignore_errors=True)
        raise

    result = {
        "project_id": project.id,
        "frame_count": count,
        "fps": rate,
        "time_span": [round(t, 3) for t in project.time_span()],
        "frame_size": [project.manifest["extract"]["width"], project.manifest["extract"]["height"]],
        "frames_dir": str(project.frames_dir),
        "source": info,
    }
    if project.edit_window:
        result["edit_window"] = _window_info(project)
    preview = None
    if target_image_path:
        located, preview = locate_target(
            project.id, target_image_path, match_threshold=match_threshold, method=detection_method
        )
        result["target"] = located
    else:
        result["next_steps"] = (
            "No target image given. Call locate_target with the image to track, or use "
            "view_frame / contact_sheet to inspect frames and update_frame to edit them directly."
        )
    return result, preview


def is_photos(project: Project) -> bool:
    return project.manifest.get("media_type") == "photos"


def open_photos(
    photo_path: str,
    target_image_path: str | None = None,
    match_threshold: float = 0.7,
    detection_method: str = "auto",
) -> tuple[dict, bytes | None]:
    """A photo, or a folder of photos, as a project: each photo is one frame. Same edit tools as videos."""
    src = resolve_media_path(photo_path, "image", allow_dir=True)
    paths = list_images(src) if src.is_dir() else [src]
    if not paths:
        raise ProjectError(f"no photos found in {src}")

    project = create_project({"path": str(src), "kind": "photos", "photo_count": len(paths)}, src.stem)
    try:
        project.frames_dir.mkdir(parents=True, exist_ok=True)
        photos = []
        for n, path in enumerate(paths):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # applies EXIF orientation
            if img is None:
                raise ProjectError(f"could not read photo {path}")
            # Lossless working copy; the photo is re-encoded in its own format only when saved.
            cv2.imwrite(str(project.frames_dir / f"frame_{n:06d}.png"), img)
            photos.append({"path": str(path.resolve()), "width": int(img.shape[1]), "height": int(img.shape[0])})
        project.manifest.update(
            media_type="photos",
            photos=photos,
            frame_count=len(photos),
            extract={"fps": 1.0, "start": 0.0, "duration": float(len(photos)), "format": "png",
                     "width": photos[0]["width"], "height": photos[0]["height"]},
        )
        project.log("open_photos", photos=len(photos))
        project.save()
    except Exception:
        shutil.rmtree(project.root, ignore_errors=True)
        raise

    result = {
        "project_id": project.id,
        "media_type": "photos",
        "photo_count": len(photos),
        "photos": [{"frame_index": n, **ph} for n, ph in enumerate(photos[:50])],
        "note": "Each photo is one frame (frame_index). Edit them with the same tools as video frames, then call "
                "save_photos.",
    }
    preview = None
    if target_image_path:
        located, preview = locate_target(project.id, target_image_path, match_threshold=match_threshold,
                                         method=detection_method)
        result["target"] = located
    return result, preview


# ---- detection -----------------------------------------------------------


def locate_target(
    project_id: str,
    target_image_path: str,
    match_threshold: float = 0.7,
    method: str = "auto",
    max_gap: int = 3,
    smooth_window: int = 3,
) -> tuple[dict, bytes | None]:
    project = load_project(project_id)
    target_file = resolve_media_path(target_image_path, "image")
    target = load_image_bgra(target_file)
    target_copy = project.root / "target.png"
    cv2.imwrite(str(target_copy), target)

    # Only search where edits are allowed: faster, and nothing outside the window is ever marked editable.
    searched = project.edit_window_frames()
    if searched is None:
        searched = list(range(project.frame_count))
    detector = TargetDetector(target, method=method, threshold=match_threshold)
    photos = is_photos(project)
    raw = detect_in_frames(detector, lambda i: project.read_frame(i, edited=False), searched, independent=photos)
    # Separate photos are unrelated: never interpolate or smooth positions between them.
    detections = postprocess([raw.get(i) for i in range(project.frame_count)], max_gap=0 if photos else max_gap,
                             smooth_window=1 if photos else smooth_window)

    project.manifest["target"] = {
        "source_path": str(target_file),
        "path": str(target_copy),
        "size": [int(target.shape[1]), int(target.shape[0])],
        "threshold": match_threshold,
        "method": method,
        "searched_frames": _ranges(searched),
        "detections": detections,
    }
    project.log("locate_target", threshold=match_threshold, method=method)
    project.save()
    return target_summary(project), _first_match_preview(project)


def _set_window(project: Project, start_time, end_time) -> None:
    start, end = parse_time(start_time), parse_time(end_time)
    if start is None and end is None:
        project.manifest.pop("edit_window", None)
        return
    lo, hi = project.time_span()
    frames = project.frames_between(start, end)
    if not frames:
        raise ProjectError(f"edit window {start}s - {end}s contains no frames; the extracted video spans "
                           f"{lo:.3f}s - {hi:.3f}s")
    project.manifest["edit_window"] = {"start": start, "end": end}


def set_edit_window(project_id: str, start_time=None, end_time=None) -> dict:
    project = load_project(project_id)
    if is_photos(project):
        raise ProjectError("edit windows apply to videos only; select photos with frame_indices")
    _set_window(project, start_time, end_time)
    project.log("set_edit_window", start=parse_time(start_time), end=parse_time(end_time))
    project.save()

    result: dict = {"edit_window": _window_info(project) or "cleared: all frames can be edited"}
    target = project.manifest.get("target")
    if target:
        window = project.edit_window_frames() or list(range(project.frame_count))
        searched = set(parse_indices(target.get("searched_frames") or "all", project.frame_count))
        missing = [i for i in window if i not in searched]
        if missing:
            result["warning"] = (
                f"The target was not searched in frames {_ranges(missing)} of this window. Call locate_target "
                "again so those frames can be edited too."
            )
    outside = [i for i in project.edited_indices() if not project.in_edit_window(i)]
    if outside:
        result["edited_frames_outside_window"] = _ranges(outside)
        result["note"] = ("These frames were edited earlier and are outside the new window; they stay edited. "
                          "Use reset_frames to revert them.")
    return result


def target_summary(project: Project) -> dict:
    dets = project.detections()
    found = project.target_frames()
    scores = [d["score"] for d in dets if d]
    missing_best = max((d["score"] for d in dets if d and not d.get("found")), default=None)
    summary = {
        "target_image": project.manifest["target"]["path"],
        "frames_with_target": len(found),
        "frames_without_target": project.frame_count - len(found),
        "frame_ranges_with_target": _ranges(found),
        "mean_score": round(float(np.mean([dets[i]["score"] for i in found])), 3) if found else None,
        "best_score_in_unmatched_frames": missing_best,
        "threshold": project.manifest["target"]["threshold"],
        "searched_frames": project.manifest["target"].get("searched_frames"),
    }
    if project.edit_window:
        summary["edit_window"] = _window_info(project)
    if found:
        summary["example_detection"] = {"frame_index": found[0], **dets[found[0]]}
        summary["next_steps"] = (
            "The target image was located in the frames listed above (quad = 4 corners TL,TR,BR,BL in pixels). "
            "To change it in ALL those frames at once call apply_to_target: operation='replace' with an image "
            "you generated (it is warped onto the target's position/perspective in every frame), 'remove' to "
            "erase it, or 'blur'/'pixelate'/'fill'. Pass a folder as replacement_path to show different images "
            "over time. For frame-specific edits use export_target_regions -> edit the crops -> "
            "import_target_regions. Then call assemble_video (videos) or save_photos (photos)."
        )
    else:
        summary["next_steps"] = (
            "Target not found in any frame. If best_score_in_unmatched_frames is close to the threshold, retry "
            "locate_target with a lower match_threshold (e.g. 0.55) or method='features' for rotated or "
            "perspective-distorted targets. Make sure the target image is a tight crop of what appears in the video."
        )
    if scores and not found:
        summary["max_score_seen"] = round(max(scores), 3)
    return summary


def _first_match_preview(project: Project) -> bytes | None:
    found = project.target_frames()
    if not found:
        return None
    i = found[0]
    return _encode(edit.annotate(project.read_frame(i, edited=False), project.detection(i)["quad"], f"target #{i}"))


# ---- editing -------------------------------------------------------------


def _replacement_source(replacement_path: str) -> list[np.ndarray]:
    p = resolve_media_path(replacement_path, "image", allow_dir=True)
    if p.is_dir():
        files = list_images(p)
        if not files:
            raise ProjectError(f"no images found in {p}")
        return [load_image_bgra(f) for f in files]
    return [load_image_bgra(p)]


def apply_to_target(
    project_id: str,
    operation: str,
    replacement_path: str | None = None,
    frame_indices=None,
    fit: str = "cover",
    feather: int = 2,
    padding: int = 4,
    opacity: float = 1.0,
    strength: int = 25,
    fill_color: str = "#000000",
    frames_per_image: int | None = None,
    base: str = "original",
    start_time=None,
    end_time=None,
) -> tuple[dict, bytes | None]:
    project = load_project(project_id)
    if operation not in OPERATIONS:
        raise ProjectError(f"operation must be one of {', '.join(OPERATIONS)}")
    if not project.manifest.get("target"):
        raise ProjectError("no target located yet; call locate_target first")
    if base not in ("original", "current"):
        raise ProjectError("base must be 'original' or 'current'")

    frames = [i for i in _select(project, frame_indices, start_time, end_time, editing=True) if project.detection(i)]
    if not frames:
        raise _no_frames_error(project, "contain the target")

    images: list[np.ndarray] = []
    if operation == "replace":
        if not replacement_path:
            raise ProjectError("replace needs replacement_path (an image file, or a folder of images)")
        images = _replacement_source(replacement_path)
    color = _parse_color(fill_color)

    per_image = frames_per_image or max(1, -(-len(frames) // len(images))) if images else 1
    for n, i in enumerate(frames):
        frame = project.read_frame(i, edited=(base == "current"))
        quad = project.detection(i)["quad"]
        if operation == "replace":
            img = images[(n // per_image) % len(images)]
            out = edit.replace(frame, quad, img, fit=fit, feather=feather, padding=padding, opacity=opacity)
        elif operation == "remove":
            out = edit.remove(frame, quad, padding=padding)
        elif operation == "blur":
            out = edit.blur(frame, quad, strength=strength, padding=padding)
        elif operation == "pixelate":
            out = edit.pixelate(frame, quad, block=max(2, strength // 2), padding=padding)
        elif operation == "fill":
            out = edit.fill(frame, quad, color, padding=padding)
        else:  # highlight: draw the detected outline, useful to check tracking in the final video
            out = edit.annotate(frame, quad)
        project.write_edited(i, out)

    project.log("apply_to_target", operation=operation, frames=len(frames), replacement=replacement_path)
    project.save()
    result = {
        "operation": operation,
        "frames_edited": len(frames),
        "frame_ranges": _ranges(frames),
        "time_range": [round(project.frame_time(frames[0]), 3), round(project.frame_time(frames[-1]) + 1 / project.fps, 3)],
        "replacement_images": len(images) or None,
        "next_steps": "Check the preview / contact_sheet(only_edited=True). If it looks right call assemble_video.",
    }
    preview_i = frames[len(frames) // 2]
    return result, _encode(project.read_frame(preview_i))


def _parse_color(value: str) -> tuple[int, int, int]:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) != 6:
        raise ProjectError("fill_color must be hex like #RRGGBB")
    r, g, b = int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    return (b, g, r)


def export_target_regions(project_id: str, frame_indices=None, base: str = "original", start_time=None,
                          end_time=None) -> dict:
    project = load_project(project_id)
    frames = [i for i in _select(project, frame_indices, start_time, end_time, editing=True) if project.detection(i)]
    if not frames:
        raise _no_frames_error(project, "contain the target")
    out_dir = project.regions_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Use one size for all crops so an editor can treat them as a uniform batch.
    tw, th = project.manifest["target"]["size"]
    sizes = [edit.quad_size(project.detection(i)["quad"]) for i in frames]
    w = int(np.median([s[0] for s in sizes]))
    w = max(64, min(w, 1024))
    h = max(1, int(round(w * th / tw)))

    exported = {}
    for i in frames:
        crop = edit.extract_region(project.read_frame(i, edited=(base == "current")), project.detection(i)["quad"], (w, h))
        path = out_dir / f"region_{i:06d}.png"
        cv2.imwrite(str(path), crop)
        exported[str(i)] = hashlib.md5(path.read_bytes()).hexdigest()
    project.manifest["regions"] = {"size": [w, h], "base": base, "hashes": exported}
    project.log("export_target_regions", frames=len(frames))
    project.save()
    return {
        "regions_dir": str(out_dir),
        "count": len(frames),
        "crop_size": [w, h],
        "filename_pattern": "region_<6-digit frame index>.png",
        "frame_ranges": _ranges(frames),
        "next_steps": (
            "Each file is the target area of one frame, rectified to an upright rectangle. Edit any of them "
            "IN PLACE (same filename; any size, PNG transparency allowed), then call import_target_regions. "
            "Unchanged files are skipped."
        ),
    }


def import_target_regions(project_id: str, feather: int = 2) -> tuple[dict, bytes | None]:
    project = load_project(project_id)
    meta = project.manifest.get("regions")
    if not meta:
        raise ProjectError("no regions exported; call export_target_regions first")
    applied, skipped, outside = [], 0, []
    window = project.edit_window_frames()
    allowed = None if window is None else set(window)
    for key, digest in meta["hashes"].items():
        i = int(key)
        if allowed is not None and i not in allowed:
            outside.append(i)  # the window changed after export
            continue
        path = project.regions_dir / f"region_{i:06d}.png"
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() == digest:
            skipped += 1
            continue
        frame = project.read_frame(i, edited=(meta.get("base") == "current"))
        out = edit.warp_onto(frame, project.detection(i)["quad"], load_image_bgra(path), feather=feather)
        project.write_edited(i, out)
        applied.append(i)
    project.log("import_target_regions", frames=len(applied))
    project.save()
    preview = _encode(project.read_frame(applied[len(applied) // 2])) if applied else None
    result = {"frames_updated": len(applied), "frame_ranges": _ranges(applied), "unchanged_skipped": skipped}
    if outside:
        result["outside_edit_window_skipped"] = _ranges(sorted(outside))
    return result, preview


def update_frame(project_id: str, frame_index: int, image_path: str) -> dict:
    project = load_project(project_id)
    i = project.check_index(frame_index)
    if not project.in_edit_window(i):
        w = project.edit_window
        raise ProjectError(
            f"frame {i} ({project.frame_time(i):.3f}s) is outside the edit window "
            f"({w.get('start')}s - {w.get('end')}s); only frames {_ranges(project.edit_window_frames())} can be edited"
        )
    img = load_image_bgra(resolve_media_path(image_path, "image"))
    h, w = project.read_frame(i, edited=False).shape[:2]
    resized = img.shape[:2] != (h, w)
    if resized:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    # Transparent pixels keep the current frame, so an overlay-only PNG works too.
    out = edit._composite(project.read_frame(i), img[:, :, :3], img[:, :, 3] / 255.0)
    path = project.write_edited(i, out)
    project.log("update_frame", frame=i)
    project.save()
    return {"frame_index": i, "saved_to": str(path), "resized_to_frame_size": resized}


def reset_frames(project_id: str, frame_indices=None, start_time=None, end_time=None) -> dict:
    project = load_project(project_id)
    # Reverting is always safe, so it is not limited to the edit window.
    selective = frame_indices is not None or start_time is not None or end_time is not None
    indices = _select(project, frame_indices, start_time, end_time, editing=False) if selective else None
    removed = project.reset(indices)
    project.log("reset_frames", frames=removed)
    project.save()
    return {"frames_reverted": removed}


# ---- inspection ------------------------------------------------------------


def frame_info(project: Project, i: int) -> dict:
    det = project.detection(i)
    info = {
        "frame_index": i,
        **({"photo": project.manifest["photos"][i]["path"]} if is_photos(project)
           else {"time": round(project.frame_time(i), 3)}),
        "path": str(project.current_path(i)),
        "edited": project.is_edited(i),
        "has_target": det is not None,
    }
    if project.edit_window:
        info["in_edit_window"] = project.in_edit_window(i)
    if det:
        info.update(quad=det["quad"], bbox=det["bbox"], score=det["score"], method=det.get("method"))
    return info


def project_status(project_id: str) -> dict:
    project = load_project(project_id)
    m = project.manifest
    status = {
        "project_id": project.id,
        "source": m["source"],
        "extract": m.get("extract"),
        "frame_count": project.frame_count,
        "time_span": [round(t, 3) for t in project.time_span()],
        "edit_window": _window_info(project),
        "edited_frames": _ranges(project.edited_indices()),
        "edited_count": len(project.edited_indices()),
        "outputs": [str(p) for p in sorted(project.output_dir.glob("*"))] if project.output_dir.exists() else [],
        "history": [h["op"] for h in m.get("history", [])][-20:],
    }
    if m.get("target"):
        status["target"] = target_summary(project)
    return status


def list_frames(project_id: str, frame_indices=None, only_with_target: bool = False, limit: int = 100,
                start_time=None, end_time=None) -> dict:
    project = load_project(project_id)
    idx = _select(project, frame_indices, start_time, end_time, editing=False)
    if only_with_target:
        idx = [i for i in idx if project.detection(i)]
    total = len(idx)
    return {"total": total, "frames": [frame_info(project, i) for i in idx[:limit]], "truncated": total > limit}


def view_frame(project_id: str, frame_index: int, edited: bool = True, annotate: bool = False,
               max_width: int = 1024) -> tuple[dict, bytes]:
    project = load_project(project_id)
    i = project.check_index(frame_index)
    img = project.read_frame(i, edited=edited)
    det = project.detection(i)
    if annotate and det:
        img = edit.annotate(img, det["quad"], f"score {det['score']:.2f}")
    return frame_info(project, i), _encode(img, max_width=max_width)


def contact_sheet(project_id: str, count: int = 12, frame_indices=None, only_with_target: bool = False,
                  only_edited: bool = False, edited: bool = True, annotate: bool = False,
                  columns: int = 4, start_time=None, end_time=None) -> tuple[dict, bytes]:
    project = load_project(project_id)
    idx = _select(project, frame_indices, start_time, end_time, editing=False)
    if only_with_target:
        idx = [i for i in idx if project.detection(i)]
    if only_edited:
        edited_set = set(project.edited_indices())
        idx = [i for i in idx if i in edited_set]
    if not idx:
        raise ProjectError("no frames match the selection")
    count = max(1, min(count, 36, len(idx)))
    picks = [idx[int(round(k))] for k in np.linspace(0, len(idx) - 1, count)]
    tiles = []
    for i in dict.fromkeys(picks):
        img = project.read_frame(i, edited=edited)
        det = project.detection(i)
        if annotate and det:
            img = edit.annotate(img, det["quad"])
        tiles.append((img, f"#{i}" + (" *" if project.is_edited(i) else "")))
    sheet = edit.contact_sheet(tiles, columns=columns)
    return {"frames_shown": list(dict.fromkeys(picks)), "legend": "#index, * = edited"}, _encode(sheet, max_width=1600)


# ---- assemble --------------------------------------------------------------


KEEP_CONTAINERS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}


def _user_video_path(project: Project) -> Path:
    """Where the user's video lives (its original path, even after replace_original moved it to a backup)."""
    src = project.manifest["source"]
    return Path(src.get("user_path") or src["path"])


def _edited_beside(project: Project, src: Path, ext: str, folder: Path | None = None) -> Path:
    """<folder>/<name>_edited<ext>; never overwrites a file this project didn't write itself."""
    if folder is None:
        folder = src.parent if os.access(src.parent, os.W_OK) else project.output_dir
    ours = set(project.manifest.get("outputs", []))
    candidate, n = folder / f"{src.stem}_edited{ext}", 2
    while candidate.exists() and str(candidate) not in ours:
        candidate, n = folder / f"{src.stem}_edited_{n}{ext}", n + 1
    return candidate


def _default_output(project: Project) -> Path:
    src = _user_video_path(project)
    return _edited_beside(project, src, src.suffix.lower() if src.suffix.lower() in KEEP_CONTAINERS else ".mp4")


PHOTO_FORMATS = {".jpg": [cv2.IMWRITE_JPEG_QUALITY, 95], ".jpeg": [cv2.IMWRITE_JPEG_QUALITY, 95],
                 ".webp": [cv2.IMWRITE_WEBP_QUALITY, 95], ".png": [], ".bmp": [], ".tif": [], ".tiff": []}


def _write_image(img: np.ndarray, dest: Path) -> None:
    """Write via a temp file in the same folder so a failure never leaves a half-written photo."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.stem}.flowedits-tmp{dest.suffix}")
    if not cv2.imwrite(str(tmp), img, PHOTO_FORMATS.get(dest.suffix.lower(), [])):
        tmp.unlink(missing_ok=True)
        raise ProjectError(f"failed to write {dest}")
    os.replace(tmp, dest)


def save_photos(project_id: str, output_path: str | None = None, replace_original: bool = False,
                frame_indices=None) -> dict:
    project = load_project(project_id)
    if not is_photos(project):
        raise ProjectError("this project is a video; use assemble_video")
    if replace_original and output_path:
        raise ProjectError("pass either output_path or replace_original, not both")
    photos = project.manifest["photos"]
    edited = set(project.edited_indices())
    targets = [i for i in parse_indices(frame_indices, project.frame_count) if i in edited]
    if not targets:
        raise ProjectError("no edited photos to save; edit them first (apply_to_target, update_frame, ...)")

    out_file = out_dir = None
    if output_path:
        dest = Path(clean_path(output_path)).expanduser()
        if len(targets) == 1 and dest.suffix.lower() in PHOTO_FORMATS:
            out_file = dest.resolve()
        else:
            out_dir = dest.resolve()

    saved, backups = [], []
    for i in targets:
        orig = Path(photos[i]["path"])
        ext = orig.suffix.lower() if orig.suffix.lower() in PHOTO_FORMATS else ".png"
        if replace_original:
            backup = orig.with_name(f"{orig.stem}.original{orig.suffix}")
            if not backup.exists():  # keep the first, untouched original
                shutil.copy2(orig, backup)
            backups.append(str(backup))
            dest = orig
        elif out_file:
            dest = out_file
        else:
            dest = _edited_beside(project, orig, ext, out_dir)
        _write_image(project.read_frame(i), dest)
        saved.append({"frame_index": i, "original": str(orig), "saved_to": str(dest)})

    outputs = project.manifest.setdefault("outputs", [])
    outputs.extend(s["saved_to"] for s in saved if s["saved_to"] not in outputs)
    project.log("save_photos", photos=len(saved), replace_original=replace_original)
    project.save()
    result = {"saved": saved, "unedited_photos_skipped": project.frame_count - len(edited)}
    if backups:
        result["original_backups"] = backups
    return result


def assemble(project_id: str, output_path: str | None = None, keep_audio: bool = True, fps: float | None = None,
             crf: int = 18, use_edited: bool = True, replace_original: bool = False) -> dict:
    project = load_project(project_id)
    if is_photos(project):
        return save_photos(project_id, output_path, replace_original)
    if replace_original and output_path:
        raise ProjectError("pass either output_path or replace_original, not both")
    rate = float(fps or project.fps)
    frames = [project.current_path(i) if use_edited else project.original_path(i) for i in range(project.frame_count)]

    user_video = _user_video_path(project)
    if replace_original:
        # Encode beside the original first; only swap once the new file is complete.
        out = user_video.with_name(f"{user_video.stem}.flowedits-tmp{user_video.suffix}")
    elif output_path:
        out = Path(clean_path(output_path)).resolve()
    else:
        out = _default_output(project)

    src = project.manifest["source"]
    extract = project.manifest["extract"]
    with_audio = keep_audio and src.get("has_audio") and abs(rate - project.fps) < 1e-6
    try:
        media.assemble_video(
            frames, out, fps=rate,
            audio_source=src["path"] if with_audio else None,
            audio_start=extract["start"] or None,
            audio_duration=project.frame_count / rate,
            crf=crf,
        )
    except Exception:
        if replace_original:
            out.unlink(missing_ok=True)
        raise

    backup = None
    if replace_original:
        backup = user_video.with_name(f"{user_video.stem}.original{user_video.suffix}")
        if not backup.exists():
            # First replacement: keep the untouched original. Later ones just overwrite the edited file.
            os.replace(user_video, backup)
        os.replace(out, user_video)
        out = user_video
        src["user_path"] = str(user_video)
        src["path"] = str(backup)  # audio for future assembles comes from the untouched original

    outputs = project.manifest.setdefault("outputs", [])
    if str(out) not in outputs:
        outputs.append(str(out))
    project.log("assemble", output=str(out), replace_original=replace_original)
    project.save()
    info = media.probe(out)
    result = {
        "output_path": str(out),
        "frames": project.frame_count,
        "edited_frames": len(project.edited_indices()) if use_edited else 0,
        "fps": rate,
        "duration": info["duration"],
        "size": [info["width"], info["height"]],
        "audio": bool(info["has_audio"]),
    }
    if backup:
        result["original_backup"] = str(backup)
    if keep_audio and src.get("has_audio") and not with_audio:
        result["note"] = "Audio dropped because fps differs from the extraction fps (it would be out of sync)."
    return result
