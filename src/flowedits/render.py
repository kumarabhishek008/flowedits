"""Apply a project's edit recipe to frames."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from . import edit
from .workspace import Project, parse_indices

TARGET_OPS = {"replace", "remove", "blur", "pixelate", "fill", "highlight", "paste_region"}


def _sequence_position(project: Project, op: dict, i: int) -> int:
    """Position of frame i in the op's original frame order (drives image sequences)."""
    cache = project._cache.setdefault("seq_pos", {})
    if op["id"] not in cache:
        order = parse_indices(op.get("order", op["frames"]), project.frame_count)
        cache[op["id"]] = {f: n for n, f in enumerate(order)}
    return cache[op["id"]].get(i, 0)


def apply_op(project: Project, op: dict, frame: np.ndarray, i: int) -> np.ndarray:
    kind = op["op"]
    if kind == "overlay_frame":
        img = project.asset(op["image"])
        h, w = frame.shape[:2]
        if img.shape[:2] != (h, w):
            import cv2

            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return edit._composite(frame, img[:, :, :3], img[:, :, 3] / 255.0)

    det = project.detection(i)
    if kind in TARGET_OPS and det is None:
        return frame  # target no longer detected here (e.g. after re-running locate_target)
    quad = det["quad"]
    p = op.get("params", {})
    pad, feather = p.get("padding", 4), p.get("feather", 2)
    # Each operation runs on a crop around the target; the margin covers what it touches outside the quad.
    if kind == "replace":
        images = op["images"]
        n = _sequence_position(project, op, i)
        img = project.asset(images[(n // max(1, op.get("per_image", 1))) % len(images)])
        return edit.on_region(frame, quad, pad + feather + 48, lambda roi, q: edit.replace(
            roi, q, img, fit=p.get("fit", "cover"), feather=feather, padding=pad, opacity=p.get("opacity", 1.0)))
    if kind == "remove":
        return edit.on_region(frame, quad, pad + 48, lambda roi, q: edit.remove(roi, q, padding=pad))
    if kind == "blur":
        strength = p.get("strength", 25)
        return edit.on_region(frame, quad, pad + 2 * strength + 8,
                              lambda roi, q: edit.blur(roi, q, strength=strength, padding=pad))
    if kind == "pixelate":
        return edit.on_region(frame, quad, pad + 2, lambda roi, q: edit.pixelate(
            roi, q, block=max(2, p.get("strength", 25) // 2), padding=pad))
    if kind == "fill":
        return edit.on_region(frame, quad, pad + 2, lambda roi, q: edit.fill(roi, q, tuple(p["color"]), padding=pad))
    if kind == "highlight":
        return edit.annotate(frame, quad)
    if kind == "paste_region":
        pq = op["quads"].get(str(i), quad)
        img = project.asset(op["images"][str(i)])
        return edit.on_region(frame, pq, feather + 4, lambda roi, q: edit.warp_onto(roi, q, img, feather=feather))
    raise ValueError(f"unknown edit operation {kind!r}")


def render(project: Project, i: int, frame: np.ndarray) -> np.ndarray:
    """Frame i with every edit that covers it applied, in order."""
    ops = project.ops_for(i)
    if not ops:
        return frame
    out = np.array(frame)  # writable copy; stream frames are read-only views
    for op in ops:
        out = apply_op(project, op, out, i)
    return out


def frame(project: Project, i: int, edited: bool = True) -> np.ndarray:
    original = project.read_original(i)
    return render(project, i, original) if edited else original


def frames(project: Project, indices: list[int], edited: bool = True) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (i, frame) for the given indices, decoding efficiently: seeks for a few, one pass for many."""
    indices = sorted(set(indices))
    if not indices:
        return
    span = indices[-1] - indices[0] + 1
    if project.is_photos or len(indices) <= 16 or span > 40 * len(indices):
        for i in indices:
            yield i, frame(project, i, edited)
        return
    wanted = set(indices)
    for i, raw in project.source.stream(indices[0], span):
        if i in wanted:
            yield i, render(project, i, raw) if edited else np.array(raw)
