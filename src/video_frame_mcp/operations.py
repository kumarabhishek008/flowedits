"""The work behind each MCP tool. Plain functions so they can be tested without a client."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np

from . import edit, media
from .detect import TargetDetector, detect_in_frames, postprocess
from .workspace import (
    Project,
    ProjectError,
    create_project,
    list_images,
    load_image_bgra,
    load_project,
    parse_indices,
)

OPERATIONS = ("replace", "remove", "blur", "pixelate", "fill", "highlight")


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
    start_time: float | None = None,
    end_time: float | None = None,
    max_frames: int | None = None,
    max_width: int | None = None,
    image_format: str = "png",
    match_threshold: float = 0.7,
    detection_method: str = "auto",
) -> tuple[dict, bytes | None]:
    if image_format not in ("png", "jpg"):
        raise ProjectError("image_format must be png or jpg")
    info = media.probe(video_path)
    rate = float(fps or info["fps"] or 25.0)
    start = float(start_time or 0.0)
    duration = None
    if end_time is not None:
        if end_time <= start:
            raise ProjectError("end_time must be greater than start_time")
        duration = end_time - start

    project = create_project(info, Path(video_path).stem)
    try:
        count = media.extract_frames(
            info["path"], project.frames_dir, fps=rate, image_format=image_format,
            start=start or None, duration=duration, max_frames=max_frames, max_width=max_width,
        )
        if count == 0:
            raise ProjectError("no frames extracted; check start_time/end_time")
        first = cv2.imread(str(project.frames_dir / f"frame_000000.{image_format}"))
        project.manifest["extract"] = {
            "fps": rate, "start": start, "duration": round(count / rate, 3), "format": image_format,
            "width": int(first.shape[1]), "height": int(first.shape[0]),
        }
        project.manifest["frame_count"] = count
        project.log("split", frames=count, fps=rate)
        project.save()
    except Exception:
        shutil.rmtree(project.root, ignore_errors=True)
        raise

    result = {
        "project_id": project.id,
        "frame_count": count,
        "fps": rate,
        "frame_size": [project.manifest["extract"]["width"], project.manifest["extract"]["height"]],
        "frames_dir": str(project.frames_dir),
        "source": info,
    }
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
    target = load_image_bgra(target_image_path)
    target_copy = project.root / "target.png"
    cv2.imwrite(str(target_copy), target)

    detector = TargetDetector(target, method=method, threshold=match_threshold)
    raw = detect_in_frames(detector, lambda i: project.read_frame(i, edited=False), list(range(project.frame_count)))
    detections = postprocess([raw[i] for i in range(project.frame_count)], max_gap=max_gap, smooth_window=smooth_window)

    project.manifest["target"] = {
        "source_path": str(Path(target_image_path).expanduser()),
        "path": str(target_copy),
        "size": [int(target.shape[1]), int(target.shape[0])],
        "threshold": match_threshold,
        "method": method,
        "detections": detections,
    }
    project.log("locate_target", threshold=match_threshold, method=method)
    project.save()
    return target_summary(project), _first_match_preview(project)


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
    }
    if found:
        summary["example_detection"] = {"frame_index": found[0], **dets[found[0]]}
        summary["next_steps"] = (
            "The target image was located in the frames listed above (quad = 4 corners TL,TR,BR,BL in pixels). "
            "To change it in ALL those frames at once call apply_to_target: operation='replace' with an image "
            "you generated (it is warped onto the target's position/perspective in every frame), 'remove' to "
            "erase it, or 'blur'/'pixelate'/'fill'. Pass a folder as replacement_path to show different images "
            "over time. For frame-specific edits use export_target_regions -> edit the crops -> "
            "import_target_regions. Then call assemble_video."
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
    p = Path(replacement_path).expanduser()
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
) -> tuple[dict, bytes | None]:
    project = load_project(project_id)
    if operation not in OPERATIONS:
        raise ProjectError(f"operation must be one of {', '.join(OPERATIONS)}")
    if not project.manifest.get("target"):
        raise ProjectError("no target located yet; call locate_target first")
    if base not in ("original", "current"):
        raise ProjectError("base must be 'original' or 'current'")

    frames = [i for i in parse_indices(frame_indices, project.frame_count) if project.detection(i)]
    if not frames:
        raise ProjectError("none of the selected frames contain the target")

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


def export_target_regions(project_id: str, frame_indices=None, base: str = "original") -> dict:
    project = load_project(project_id)
    frames = [i for i in parse_indices(frame_indices, project.frame_count) if project.detection(i)]
    if not frames:
        raise ProjectError("no selected frames contain the target")
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
    applied, skipped = [], 0
    for key, digest in meta["hashes"].items():
        i = int(key)
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
    return {"frames_updated": len(applied), "frame_ranges": _ranges(applied), "unchanged_skipped": skipped}, preview


def update_frame(project_id: str, frame_index: int, image_path: str) -> dict:
    project = load_project(project_id)
    i = project.check_index(frame_index)
    img = load_image_bgra(image_path)
    h, w = project.manifest["extract"]["height"], project.manifest["extract"]["width"]
    resized = img.shape[:2] != (h, w)
    if resized:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    # Transparent pixels keep the current frame, so an overlay-only PNG works too.
    out = edit._composite(project.read_frame(i), img[:, :, :3], img[:, :, 3] / 255.0)
    path = project.write_edited(i, out)
    project.log("update_frame", frame=i)
    project.save()
    return {"frame_index": i, "saved_to": str(path), "resized_to_frame_size": resized}


def reset_frames(project_id: str, frame_indices=None) -> dict:
    project = load_project(project_id)
    indices = None if frame_indices is None else parse_indices(frame_indices, project.frame_count)
    removed = project.reset(indices)
    project.log("reset_frames", frames=removed)
    project.save()
    return {"frames_reverted": removed}


# ---- inspection ------------------------------------------------------------


def frame_info(project: Project, i: int) -> dict:
    det = project.detection(i)
    info = {
        "frame_index": i,
        "time": round(project.manifest["extract"]["start"] + i / project.fps, 3),
        "path": str(project.current_path(i)),
        "edited": project.is_edited(i),
        "has_target": det is not None,
    }
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
        "edited_frames": _ranges(project.edited_indices()),
        "edited_count": len(project.edited_indices()),
        "outputs": [str(p) for p in sorted(project.output_dir.glob("*"))] if project.output_dir.exists() else [],
        "history": [h["op"] for h in m.get("history", [])][-20:],
    }
    if m.get("target"):
        status["target"] = target_summary(project)
    return status


def list_frames(project_id: str, frame_indices=None, only_with_target: bool = False, limit: int = 100) -> dict:
    project = load_project(project_id)
    idx = parse_indices(frame_indices, project.frame_count)
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
                  columns: int = 4) -> tuple[dict, bytes]:
    project = load_project(project_id)
    idx = parse_indices(frame_indices, project.frame_count)
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


def assemble(project_id: str, output_path: str | None = None, keep_audio: bool = True, fps: float | None = None,
             crf: int = 18, use_edited: bool = True) -> dict:
    project = load_project(project_id)
    rate = float(fps or project.fps)
    frames = [project.current_path(i) if use_edited else project.original_path(i) for i in range(project.frame_count)]
    if output_path:
        out = Path(output_path).expanduser().resolve()
    else:
        stem = Path(project.manifest["source"]["path"]).stem
        out = project.output_dir / f"{stem}_edited.mp4"

    src = project.manifest["source"]
    extract = project.manifest["extract"]
    with_audio = keep_audio and src.get("has_audio") and abs(rate - project.fps) < 1e-6
    media.assemble_video(
        frames, out, fps=rate,
        audio_source=src["path"] if with_audio else None,
        audio_start=extract["start"] or None,
        audio_duration=project.frame_count / rate,
        crf=crf,
    )
    project.log("assemble", output=str(out))
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
    if keep_audio and src.get("has_audio") and not with_audio:
        result["note"] = "Audio dropped because fps differs from the extraction fps (it would be out of sync)."
    return result
