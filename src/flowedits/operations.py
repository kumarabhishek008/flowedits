"""The work behind each MCP tool. Plain functions so they can be tested without a client.

Long functions accept `job` (a jobs.Job) to report progress and support cancellation; the server
runs them in the background. Without a job they simply run to completion.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import edit, media, render
from .detect import TargetDetector, detect_parallel, detect_sequence, postprocess, scale_detection
from .paths import clean_path, resolve_media_path
from .workspace import (
    IMAGE_EXTS,
    Project,
    ProjectError,
    create_project,
    format_ranges,
    list_images,
    load_image_bgra,
    load_project,
    parse_indices,
    parse_time,
)

OPERATIONS = ("replace", "remove", "blur", "pixelate", "fill", "highlight")
DETECT_MAX_WIDTH = 1280      # detection runs on frames downscaled to this width; quads are scaled back
PARALLEL_MIN_FRAMES = 600    # below this one thread is fast enough
STRIDE_MIN_FRAMES = 900      # below this every frame gets a full search when the target is lost
LOST_STRIDE = 5              # full search every 5th frame while the target is missing...
LOST_MAX_STRIDE = 15         # ...widening to every 15th (0.5 s at 30 fps) when it stays away
MAX_EXPORT = 3000
QUALITY = {"draft": ("veryfast", 23), "standard": ("fast", 18), "high": ("medium", 16)}


class _Progress:
    """Thread-safe frame counter feeding a job's progress (raises Cancelled via job.update)."""

    def __init__(self, job, total: int, message: str, lo: float = 0.0, hi: float = 1.0):
        self.job, self.total, self.message, self.lo, self.hi = job, max(1, total), message, lo, hi
        self.done = 0
        self._lock = threading.Lock()

    def tick(self, n: int = 1) -> None:
        if self.job is None:
            return
        with self._lock:
            self.done += n
            done = self.done
        if done % 30 == 0 or done >= self.total:
            self.job.update(self.lo + (self.hi - self.lo) * done / self.total,
                            f"{self.message}: {done}/{self.total} frames")


def _encode(img: np.ndarray, max_width: int = 1024, quality: int = 85) -> bytes:
    if img.shape[1] > max_width:
        k = max_width / img.shape[1]
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ProjectError("failed to encode preview")
    return buf.tobytes()


def _select(project: Project, frame_indices=None, start_time=None, end_time=None, *, editing: bool) -> list[int]:
    """Frames picked by indices AND time range; editing tools are further limited to the edit window."""
    idx = parse_indices(frame_indices, project.frame_count)
    start, end = parse_time(start_time), parse_time(end_time)
    if (start is not None or end is not None) and project.is_photos:
        raise ProjectError("start_time/end_time apply to videos only; select photos with frame_indices")
    if start is not None or end is not None:
        span = set(project.frames_between(start, end))
        idx = [i for i in idx if i in span]
    if editing and project.edit_window_frames() is not None:
        idx = [i for i in idx if project.in_edit_window(i)]
    return idx


def _window_info(project: Project) -> dict | None:
    w = project.edit_window
    if w is None:
        return None
    frames = project.edit_window_frames()
    return {
        "start_time": w.get("start"),
        "end_time": w.get("end"),
        "frames": format_ranges(frames),
        "frame_count": len(frames),
        "rule": "Only these frames can be edited; every editing tool skips frames outside this time window.",
    }


def _no_frames_error(project: Project, what: str) -> ProjectError:
    msg = f"no selected frames {what}"
    if project.edit_window:
        w = project.edit_window
        msg += f" inside the edit window ({w.get('start')}s - {w.get('end')}s)"
    return ProjectError(msg)


def _parse_color(value: str) -> tuple[int, int, int]:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) != 6:
        raise ProjectError("fill_color must be hex like #RRGGBB")
    r, g, b = int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    return (b, g, r)


# ---- open ------------------------------------------------------------------


def split_video(
    video_path: str,
    target_image_path: str | None = None,
    fps: float | None = None,
    trim_start_time=None,
    trim_end_time=None,
    max_frames: int | None = None,
    max_width: int | None = None,
    match_threshold: float = 0.7,
    detection_method: str = "auto",
    edit_start_time=None,
    edit_end_time=None,
    job=None,
) -> tuple[dict, bytes | None]:
    """Open a video as a project (frames are decoded on demand, nothing is extracted)."""
    video = resolve_media_path(video_path, "any", allow_dir=True)
    if video.is_dir() or video.suffix.lower() in IMAGE_EXTS:
        if any(v is not None for v in (trim_start_time, trim_end_time, edit_start_time, edit_end_time)):
            raise ProjectError("time options apply to videos only; this is a photo (or folder of photos)")
        return open_photos(str(video), target_image_path, match_threshold, detection_method, job=job)

    info = media.probe(video)
    rate = float(fps or info["fps"] or 25.0)
    start = parse_time(trim_start_time) or 0.0
    end = parse_time(trim_end_time)
    if end is not None and end <= start:
        raise ProjectError("trim_end_time must be greater than trim_start_time")
    end = min(end if end is not None else info["duration"], info["duration"])
    if start >= end:
        raise ProjectError(f"trim_start_time is past the end of the video ({info['duration']}s)")
    native = abs(rate - info["fps"]) < 1e-3 and start == 0 and end == info["duration"] and info["frame_count"] > 0
    count = info["frame_count"] if native else int(round((end - start) * rate))
    if max_frames:
        count = min(count, int(max_frames))

    width, height = info["width"], info["height"]
    if max_width and width > max_width:
        width, height = media.even(max_width), media.even(height * max_width / width)

    project = create_project(info, video.stem)
    try:
        project.manifest["media_type"] = "video"
        project.manifest["extract"] = {"fps": rate, "start": start, "width": width, "height": height}
        project.manifest["frame_count"] = count
        # Container frame counts can be off by one or two at the end; make sure the last frame decodes.
        for _ in range(3):
            try:
                project.source.read(count - 1)
                break
            except ProjectError:
                count -= 1
                project.manifest["frame_count"] = count
                project._cache.pop("source", None)
        project.manifest["extract"]["duration"] = round(count / rate, 3)
        project.log("open_video", frames=count, fps=rate)
        if edit_start_time is not None or edit_end_time is not None:
            _set_window(project, edit_start_time, edit_end_time)
        project.save()
    except Exception:
        shutil.rmtree(project.root, ignore_errors=True)
        raise

    result = {
        "project_id": project.id,
        "media_type": "video",
        "frame_count": count,
        "fps": rate,
        "duration_seconds": round(count / rate, 3),
        "time_span": [round(t, 3) for t in project.time_span()],
        "frame_size": [width, height],
        "source": info,
    }
    if project.edit_window:
        result["edit_window"] = _window_info(project)
    preview = None
    if target_image_path:
        located, preview = locate_target(project.id, target_image_path, match_threshold=match_threshold,
                                         method=detection_method, job=job)
        result["target"] = located
    else:
        result["next_steps"] = (
            "No target image given. Call locate_target with the image to track, or use view_frame / "
            "contact_sheet to look at frames and export_frames + update_frame to edit whole frames."
        )
    return result, preview


def open_photos(
    photo_path: str,
    target_image_path: str | None = None,
    match_threshold: float = 0.7,
    detection_method: str = "auto",
    job=None,
) -> tuple[dict, bytes | None]:
    """A photo, or a folder of photos, as a project: each photo is one frame. Same edit tools as videos."""
    src = resolve_media_path(photo_path, "image", allow_dir=True)
    paths = list_images(src) if src.is_dir() else [src]
    if not paths:
        raise ProjectError(f"no photos found in {src}")

    project = create_project({"path": str(src), "kind": "photos", "photo_count": len(paths)}, src.stem)
    try:
        project.photos_dir.mkdir(parents=True, exist_ok=True)
        photos = []
        for n, path in enumerate(paths):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # applies EXIF orientation
            if img is None:
                raise ProjectError(f"could not read photo {path}")
            # Lossless working copy; the photo is re-encoded in its own format only when saved.
            cv2.imwrite(str(project.photos_dir / f"photo_{n:06d}.png"), img)
            photos.append({"path": str(path.resolve()), "width": int(img.shape[1]), "height": int(img.shape[0])})
            if job:
                job.update(0.2 * (n + 1) / len(paths), f"copying photos: {n + 1}/{len(paths)}")
        project.manifest.update(
            media_type="photos",
            photos=photos,
            frame_count=len(photos),
            extract={"fps": 1.0, "start": 0.0, "duration": float(len(photos)),
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
                                         method=detection_method, job=job)
        result["target"] = located
    return result, preview


# ---- detection -------------------------------------------------------------


def locate_target(
    project_id: str,
    target_image_path: str,
    match_threshold: float = 0.7,
    method: str = "auto",
    max_gap: int = 3,
    smooth_window: int = 3,
    job=None,
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
    progress = _Progress(job, len(searched), "locating target", 0.2 if project.is_photos else 0.0)

    def make_detector() -> TargetDetector:
        return TargetDetector(target, method=method, threshold=match_threshold)

    if project.is_photos:
        raw = detect_sequence(make_detector(), ((i, project.read_original(i)) for i in searched),
                              independent=True, on_frame=progress.tick)
    else:
        k = min(1.0, DETECT_MAX_WIDTH / project.width)
        det_size = (max(16, round(project.width * k)), max(16, round(project.height * k)))
        factor = project.width / det_size[0]
        first, count = searched[0], len(searched)  # windows are contiguous
        workers = 1 if count < PARALLEL_MIN_FRAMES else min(8, max(1, (os.cpu_count() or 4) - 2))
        stride = 1 if count <= STRIDE_MIN_FRAMES else LOST_STRIDE
        raw = detect_parallel(make_detector, lambda s, n: project.source.stream(s, n, size=det_size, gray=True), first, count,
                              workers=workers, stride=stride, max_stride=LOST_MAX_STRIDE if stride > 1 else 1,
                              on_frame=progress.tick)
        raw = {i: scale_detection(d, factor) for i, d in raw.items()}

    photos = project.is_photos
    # Separate photos are unrelated: never interpolate or smooth positions between them.
    detections = postprocess([raw.get(i) for i in range(project.frame_count)], max_gap=0 if photos else max_gap,
                             smooth_window=1 if photos else smooth_window)
    project.set_detections(detections)
    project.manifest["target"] = {
        "source_path": str(target_file),
        "path": str(target_copy),
        "size": [int(target.shape[1]), int(target.shape[0])],
        "threshold": match_threshold,
        "method": method,
        "searched_frames": format_ranges(searched),
    }
    project.log("locate_target", threshold=match_threshold, method=method)
    project.save()
    return target_summary(project), _first_match_preview(project)


def target_summary(project: Project) -> dict:
    dets = project.detections()
    found = project.target_frames()
    missing_best = max((d["score"] for d in dets if d and not d.get("found") and not d.get("skipped")), default=None)
    summary = {
        "target_image": project.manifest["target"]["path"],
        "frames_with_target": len(found),
        "frames_without_target": project.frame_count - len(found),
        "frame_ranges_with_target": format_ranges(found),
        "mean_score": round(float(np.mean([dets[i]["score"] for i in found])), 3) if found else None,
        "best_score_in_unmatched_frames": missing_best,
        "threshold": project.manifest["target"]["threshold"],
        "searched_frames": project.manifest["target"].get("searched_frames"),
    }
    if not project.is_photos and found:
        summary["time_ranges_with_target"] = _time_ranges(project, found)
    if project.edit_window:
        summary["edit_window"] = _window_info(project)
    if found:
        summary["example_detection"] = {"frame_index": found[0], **dets[found[0]]}
        finish = "save_photos" if project.is_photos else "render_video"
        summary["next_steps"] = (
            "The target was located in the frames listed above (quad = 4 corners TL,TR,BR,BL in pixels). To change "
            "it everywhere at once call apply_to_target: operation='replace' with ONE image you made (it follows "
            "the target's position/perspective in every frame), 'remove' to erase it, or 'blur'/'pixelate'/'fill'. "
            "A folder as replacement_path shows different images over time. For per-frame artwork use "
            f"export_target_regions -> edit crops -> import_target_regions. Then call {finish}."
        )
    else:
        summary["next_steps"] = (
            "Target not found. If best_score_in_unmatched_frames is close to the threshold, retry locate_target "
            "with a lower match_threshold (e.g. 0.55) or method='features' for rotated or perspective-distorted "
            "targets. The target image should be a tight crop of how it appears in the video/photo."
        )
    return summary


def _time_ranges(project: Project, frames: list[int]) -> list[list[float]]:
    out = []
    for part in format_ranges(frames).split(","):
        lo, _, hi = part.partition("-")
        a, b = int(lo), int(hi or lo)
        out.append([round(project.frame_time(a), 2), round(project.frame_time(b) + 1 / project.fps, 2)])
    return out[:50]


def _first_match_preview(project: Project) -> bytes | None:
    found = project.target_frames()
    if not found:
        return None
    i = found[0]
    return _encode(edit.annotate(project.read_original(i), project.detection(i)["quad"], f"target #{i}"))


# ---- edit window -------------------------------------------------------------


def _set_window(project: Project, start_time, end_time) -> None:
    start, end = parse_time(start_time), parse_time(end_time)
    if start is None and end is None:
        project.set_edit_window(None)
        return
    lo, hi = project.time_span()
    frames = project.frames_between(start, end)
    if not frames:
        raise ProjectError(f"edit window {start}s - {end}s contains no frames; the video spans "
                           f"{lo:.3f}s - {hi:.3f}s")
    project.set_edit_window({"start": start, "end": end})


def set_edit_window(project_id: str, start_time=None, end_time=None) -> dict:
    project = load_project(project_id)
    if project.is_photos:
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
                f"The target was not searched in frames {format_ranges(missing)} of this window. Call "
                "locate_target again so those frames can be edited too."
            )
    outside = [i for i in project.edited_indices() if not project.in_edit_window(i)]
    if outside:
        result["edited_frames_outside_window"] = format_ranges(outside)
        result["note"] = ("These frames were edited earlier and are outside the new window; they stay edited. "
                          "Use reset_frames to revert them.")
    return result


# ---- editing (recorded in the recipe; rendered later) -----------------------


def _preview(project: Project, frames: list[int]) -> bytes:
    i = frames[len(frames) // 2]
    return _encode(render.frame(project, i))


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
    if fit not in ("cover", "contain", "stretch"):
        raise ProjectError("fit must be cover, contain or stretch")

    frames = [i for i in _select(project, frame_indices, start_time, end_time, editing=True) if project.detection(i)]
    if not frames:
        raise _no_frames_error(project, "contain the target")

    op: dict = {"op": operation, "params": {"feather": feather, "padding": padding}}
    if operation == "replace":
        if not replacement_path:
            raise ProjectError("replace needs replacement_path (an image file, or a folder of images)")
        p = resolve_media_path(replacement_path, "image", allow_dir=True)
        files = list_images(p) if p.is_dir() else [p]
        if not files:
            raise ProjectError(f"no images found in {p}")
        op_id = project.manifest.get("next_op_id", 1)
        op["images"] = [project.store_asset(load_image_bgra(f), f"op{op_id}_{n}.png") for n, f in enumerate(files)]
        op["per_image"] = frames_per_image or max(1, -(-len(frames) // len(files)))
        op["order"] = format_ranges(frames)
        op["params"].update(fit=fit, opacity=opacity)
        op["source"] = str(p)
    elif operation in ("blur", "pixelate"):
        op["params"]["strength"] = strength
    elif operation == "fill":
        op["params"]["color"] = list(_parse_color(fill_color))

    if base == "original":
        project.remove_frames(frames)  # start from untouched frames: drop earlier edits there
    project.add_op(op, frames)
    project.log("apply_to_target", operation=operation, frames=len(frames))
    project.save()
    result = {
        "operation": operation,
        "frames_edited": len(frames),
        "frame_ranges": format_ranges(frames),
        "replacement_images": len(op.get("images", [])) or None,
        "next_steps": ("Recorded instantly; frames are rendered when you finish. Check the preview or "
                       "contact_sheet(only_edited=True), then call "
                       + ("save_photos." if project.is_photos else "render_video.")),
    }
    if not project.is_photos:
        result["time_range"] = [round(project.frame_time(frames[0]), 3),
                                round(project.frame_time(frames[-1]) + 1 / project.fps, 3)]
    return result, _preview(project, frames)


def export_target_regions(project_id: str, frame_indices=None, base: str = "original", start_time=None,
                          end_time=None, job=None) -> dict:
    project = load_project(project_id)
    frames = [i for i in _select(project, frame_indices, start_time, end_time, editing=True) if project.detection(i)]
    if not frames:
        raise _no_frames_error(project, "contain the target")
    if len(frames) > MAX_EXPORT:
        raise ProjectError(f"{len(frames)} frames selected; export at most {MAX_EXPORT} at a time (narrow with "
                           "frame_indices or start_time/end_time). For the same change everywhere use apply_to_target.")
    out_dir = project.regions_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Use one size for all crops so an editor can treat them as a uniform batch.
    tw, th = project.manifest["target"]["size"]
    sizes = [edit.quad_size(project.detection(i)["quad"]) for i in frames]
    w = max(64, min(int(np.median([s[0] for s in sizes])), 1024))
    h = max(1, int(round(w * th / tw)))

    progress = _Progress(job, len(frames), "exporting regions")
    hashes, quads = {}, {}
    for i, img in render.frames(project, frames, edited=(base == "current")):
        quad = project.detection(i)["quad"]
        path = out_dir / f"region_{i:06d}.png"
        cv2.imwrite(str(path), edit.extract_region(img, quad, (w, h)))
        hashes[str(i)] = hashlib.md5(path.read_bytes()).hexdigest()
        quads[str(i)] = quad
        progress.tick()
    project.manifest["regions"] = {"size": [w, h], "base": base, "hashes": hashes, "quads": quads}
    project.log("export_target_regions", frames=len(frames))
    project.save()
    return {
        "regions_dir": str(out_dir),
        "count": len(frames),
        "crop_size": [w, h],
        "filename_pattern": "region_<6-digit frame index>.png",
        "frame_ranges": format_ranges(frames),
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
    images, quads = {}, {}
    op_id = project.manifest.get("next_op_id", 1)
    for key, digest in meta["hashes"].items():
        i = int(key)
        if not project.in_edit_window(i):
            outside.append(i)  # the window changed after export
            continue
        path = project.regions_dir / f"region_{i:06d}.png"
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() == digest:
            skipped += 1
            continue
        images[key] = project.store_asset(load_image_bgra(path), f"op{op_id}_region_{i:06d}.png")
        quads[key] = meta.get("quads", {}).get(key) or project.detection(i)["quad"]
        applied.append(i)
    result = {"frames_updated": len(applied), "frame_ranges": format_ranges(applied), "unchanged_skipped": skipped}
    if outside:
        result["outside_edit_window_skipped"] = format_ranges(outside)
    if not applied:
        return result, None
    if meta.get("base") != "current":
        project.remove_frames(applied)
    project.add_op({"op": "paste_region", "params": {"feather": feather}, "images": images, "quads": quads}, applied)
    project.log("import_target_regions", frames=len(applied))
    project.save()
    return result, _preview(project, applied)


def update_frame(project_id: str, frame_index: int, image_path: str) -> dict:
    project = load_project(project_id)
    i = project.check_index(frame_index)
    if not project.in_edit_window(i):
        w = project.edit_window
        raise ProjectError(
            f"frame {i} ({project.frame_time(i):.3f}s) is outside the edit window "
            f"({w.get('start')}s - {w.get('end')}s); only frames {format_ranges(project.edit_window_frames())} "
            "can be edited"
        )
    img = load_image_bgra(resolve_media_path(image_path, "image"))
    w, h = project.frame_size(i)
    resized = img.shape[:2] != (h, w)
    if resized:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    op_id = project.manifest.get("next_op_id", 1)
    # Transparent pixels keep the current frame, so an overlay-only PNG works too.
    project.add_op({"op": "overlay_frame", "image": project.store_asset(img, f"op{op_id}_frame_{i:06d}.png")}, [i])
    project.log("update_frame", frame=i)
    project.save()
    return {"frame_index": i, "resized_to_frame_size": resized, "recorded": True}


def reset_frames(project_id: str, frame_indices=None, start_time=None, end_time=None) -> dict:
    project = load_project(project_id)
    # Reverting is always safe, so it is not limited to the edit window.
    selective = frame_indices is not None or start_time is not None or end_time is not None
    indices = _select(project, frame_indices, start_time, end_time, editing=False) if selective else \
        project.edited_indices()
    removed = project.remove_frames(indices)
    project.log("reset_frames", frames=removed)
    project.save()
    return {"frames_reverted": removed}


# ---- inspection ------------------------------------------------------------


def frame_info(project: Project, i: int) -> dict:
    det = project.detection(i)
    info: dict = {"frame_index": i}
    if project.is_photos:
        info["photo"] = project.manifest["photos"][i]["path"]
    else:
        info["time"] = round(project.frame_time(i), 3)
    info.update(edited=project.is_edited(i), has_target=det is not None)
    if project.edit_window:
        info["in_edit_window"] = project.in_edit_window(i)
    if det:
        info.update(quad=det["quad"], bbox=det["bbox"], score=det["score"], method=det.get("method"))
    return info


def project_status(project_id: str) -> dict:
    project = load_project(project_id)
    m = project.manifest
    edited = project.edited_indices()
    status = {
        "project_id": project.id,
        "media_type": m.get("media_type"),
        "source": m["source"],
        "frame_count": project.frame_count,
        "edit_window": _window_info(project),
        "edited_frames": format_ranges(edited),
        "edited_count": len(edited),
        "edits": [{"id": op["id"], "op": op["op"], "frames": op["frames"]} for op in project.ops][-30:],
        "outputs": m.get("outputs", []),
        "history": [h["op"] for h in m.get("history", [])][-20:],
    }
    if not project.is_photos:
        status["fps"] = project.fps
        status["time_span"] = [round(t, 3) for t in project.time_span()]
        status["frame_size"] = [project.width, project.height]
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
    img = render.frame(project, i, edited)
    project.frames_dir.mkdir(parents=True, exist_ok=True)
    path = project.frames_dir / f"frame_{i:06d}{'_edited' if edited and project.is_edited(i) else ''}.png"
    cv2.imwrite(str(path), img)
    det = project.detection(i)
    if annotate and det:
        img = edit.annotate(img, det["quad"], f"score {det['score']:.2f}")
    return {**frame_info(project, i), "saved_to": str(path)}, _encode(img, max_width=max_width)


def export_frames(project_id: str, frame_indices=None, start_time=None, end_time=None, edited: bool = True,
                  limit: int = 300, job=None) -> dict:
    """Write full frames as PNG files (e.g. to edit some by hand and send back with update_frame)."""
    project = load_project(project_id)
    if frame_indices is None and start_time is None and end_time is None:
        raise ProjectError("select frames with frame_indices or start_time/end_time")
    idx = _select(project, frame_indices, start_time, end_time, editing=False)
    if len(idx) > limit:
        raise ProjectError(f"{len(idx)} frames selected; export at most {limit} per call")
    project.frames_dir.mkdir(parents=True, exist_ok=True)
    progress = _Progress(job, len(idx), "exporting frames")
    paths = []
    for i, img in render.frames(project, idx, edited):
        path = project.frames_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(path), img)
        paths.append(str(path))
        progress.tick()
    return {"count": len(paths), "frames_dir": str(project.frames_dir), "files": paths[:100],
            "filename_pattern": "frame_<6-digit frame index>.png",
            "next_steps": "Edit a file and pass it to update_frame(project_id, frame_index, image_path)."}


def contact_sheet(project_id: str, count: int = 12, frame_indices=None, only_with_target: bool = False,
                  only_edited: bool = False, edited: bool = True, annotate: bool = False,
                  columns: int = 4, start_time=None, end_time=None) -> tuple[dict, bytes]:
    project = load_project(project_id)
    idx = _select(project, frame_indices, start_time, end_time, editing=False)
    if only_with_target:
        idx = [i for i in idx if project.detection(i)]
    if only_edited:
        edited_set = project.edited_set()
        idx = [i for i in idx if i in edited_set]
    if not idx:
        raise ProjectError("no frames match the selection")
    count = max(1, min(count, 36, len(idx)))
    picks = list(dict.fromkeys(idx[int(round(k))] for k in np.linspace(0, len(idx) - 1, count)))
    tiles = []
    for i, img in render.frames(project, picks, edited):
        det = project.detection(i)
        if annotate and det:
            img = edit.annotate(img, det["quad"])
        label = f"#{i}" if project.is_photos else f"#{i} {project.frame_time(i):.1f}s"
        tiles.append((img, label + (" *" if project.is_edited(i) else "")))
    sheet = edit.contact_sheet(tiles, columns=columns)
    return {"frames_shown": picks, "legend": "#index time, * = edited"}, _encode(sheet, max_width=1600)


# ---- output ----------------------------------------------------------------

KEEP_CONTAINERS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
PHOTO_FORMATS = {".jpg": [cv2.IMWRITE_JPEG_QUALITY, 95], ".jpeg": [cv2.IMWRITE_JPEG_QUALITY, 95],
                 ".webp": [cv2.IMWRITE_WEBP_QUALITY, 95], ".png": [], ".bmp": [], ".tif": [], ".tiff": []}


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


def _tmp_beside(dest: Path) -> Path:
    return dest.with_name(f".{dest.stem}.flowedits-tmp{dest.suffix}")


def assemble(project_id: str, output_path: str | None = None, keep_audio: bool = True, crf: int | None = None,
             use_edited: bool = True, replace_original: bool = False, quality: str = "standard",
             job=None) -> dict:
    """Render the video in one streaming pass: decode -> apply the recipe to edited frames -> encode."""
    project = load_project(project_id)
    if project.is_photos:
        return save_photos(project_id, output_path, replace_original, job=job)
    if replace_original and output_path:
        raise ProjectError("pass either output_path or replace_original, not both")
    if quality not in QUALITY:
        raise ProjectError(f"quality must be one of {', '.join(QUALITY)}")
    preset, default_crf = QUALITY[quality]

    user_video = _user_video_path(project)
    if replace_original:
        dest = user_video
    elif output_path:
        dest = Path(clean_path(output_path)).resolve()
    else:
        src = user_video
        dest = _edited_beside(project, src, src.suffix.lower() if src.suffix.lower() in KEEP_CONTAINERS else ".mp4")
    tmp = _tmp_beside(dest)

    src_info = project.manifest["source"]
    with_audio = keep_audio and src_info.get("has_audio")
    edited = project.edited_set() if use_edited else set()
    w, h = project.width, project.height
    progress = _Progress(job, project.frame_count, "rendering")
    encoder = media.Encoder(
        tmp, size=(w, h), fps=project.fps,
        audio_source=src_info["path"] if with_audio else None,
        audio_start=project.start or None,
        audio_duration=project.frame_count / project.fps,
        crf=crf if crf is not None else default_crf, preset=preset,
    )
    written = 0
    # Edited frames are rendered on a thread pool (OpenCV releases the GIL) and written in order;
    # the bounded queue keeps memory flat however long the video is.
    pending: deque = deque()
    workers = min(8, max(2, (os.cpu_count() or 4) - 2))

    def flush(limit: int) -> None:
        nonlocal written
        while len(pending) > limit:
            item = pending.popleft()
            encoder.write(item.result() if isinstance(item, Future) else item)
            written += 1
            progress.tick()

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, raw in enumerate(project.source.stream_raw()):
                if i in edited:
                    frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
                    pending.append(pool.submit(render.render, project, i, frame))
                else:
                    pending.append(raw)  # untouched frames pass straight through
                flush(workers * 3)
            flush(0)
        encoder.close()
    except BaseException:
        encoder.abort()
        tmp.unlink(missing_ok=True)
        raise

    backup = None
    if replace_original:
        backup = user_video.with_name(f"{user_video.stem}.original{user_video.suffix}")
        if not backup.exists():
            # First replacement: keep the untouched original. Later ones just overwrite the edited file.
            os.replace(user_video, backup)
        src_info["user_path"] = str(user_video)
        src_info["path"] = str(backup)  # future renders decode the untouched original
        project._cache.pop("source", None)
    os.replace(tmp, dest)

    outputs = project.manifest.setdefault("outputs", [])
    if str(dest) not in outputs:
        outputs.append(str(dest))
    project.log("render", output=str(dest), replace_original=replace_original)
    project.save()
    info = media.probe(dest)
    result = {
        "output_path": str(dest),
        "frames": written,
        "edited_frames": len(edited),
        "fps": project.fps,
        "duration": info["duration"],
        "size": [info["width"], info["height"]],
        "audio": bool(info["has_audio"]),
    }
    if backup:
        result["original_backup"] = str(backup)
    return result


def _write_image(img: np.ndarray, dest: Path) -> None:
    """Write via a temp file in the same folder so a failure never leaves a half-written photo."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_beside(dest)
    if not cv2.imwrite(str(tmp), img, PHOTO_FORMATS.get(dest.suffix.lower(), [])):
        tmp.unlink(missing_ok=True)
        raise ProjectError(f"failed to write {dest}")
    os.replace(tmp, dest)


def save_photos(project_id: str, output_path: str | None = None, replace_original: bool = False,
                frame_indices=None, job=None) -> dict:
    project = load_project(project_id)
    if not project.is_photos:
        raise ProjectError("this project is a video; use render_video")
    if replace_original and output_path:
        raise ProjectError("pass either output_path or replace_original, not both")
    photos = project.manifest["photos"]
    edited = project.edited_set()
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

    progress = _Progress(job, len(targets), "saving photos")
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
        _write_image(render.frame(project, i), dest)
        saved.append({"frame_index": i, "original": str(orig), "saved_to": str(dest)})
        progress.tick()

    outputs = project.manifest.setdefault("outputs", [])
    outputs.extend(s["saved_to"] for s in saved if s["saved_to"] not in outputs)
    project.log("save_photos", photos=len(saved), replace_original=replace_original)
    project.save()
    result = {"saved": saved, "unedited_photos_skipped": project.frame_count - len(edited)}
    if backups:
        result["original_backups"] = backups
    return result
