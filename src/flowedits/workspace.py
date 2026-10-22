"""Project workspace: source, detections and the edit recipe of one video or photo set.

Videos are never extracted to disk. Frames are decoded on demand (ffmpeg pipes) and edits are
stored as an ordered recipe of operations that is applied while rendering, so long videos cost
no more disk than their output. Photos are copied once into the project as lossless PNGs.

Layout of one project:

    <workspace>/<project_id>/
        manifest.json      source info, settings, edit window, target metadata, edit recipe
        detections.json    per-frame target position (quad) — kept apart, it can be large
        photos/            lossless working copies (photo projects only)
        assets/            replacement images and pasted regions referenced by the recipe
        regions/           rectified crops of the target, exported for hand editing
        frames/            full frames exported for hand editing / viewing
        target.png         copy of the target image being tracked
        output/            renders that could not be written next to the source
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from . import media

def workspace_root() -> Path:
    """Where projects live; read on each call so FLOWEDITS_WORKSPACE can change at runtime (tests)."""
    return Path(os.environ.get("FLOWEDITS_WORKSPACE", Path.home() / ".flowedits" / "projects")).expanduser()

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class ProjectError(ValueError):
    pass


def format_ranges(indices) -> str:
    """[1,2,3,7,9,10] -> "1-3,7,9-10" (compact for LLM context and for storage)."""
    indices = sorted(indices)
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


# ---- frame sources ---------------------------------------------------------


class VideoSource:
    """Frames of a video decoded on demand. Frame i is at extract.start + i / fps on the source timeline."""

    def __init__(self, project: "Project"):
        m = project.manifest
        self.path = m["source"]["path"]
        self.fps = project.fps
        self.start = project.start
        self.count = project.frame_count
        self.size = (project.width, project.height)

    def _seek_time(self, i: int) -> float:
        # Land a quarter frame before frame i so accurate seeking returns exactly frame i.
        return max(0.0, self.start + (i - 0.25) / self.fps) if (i or self.start) else 0.0

    def read(self, i: int) -> np.ndarray:
        frame = media.read_frame_at(self.path, self._seek_time(i), size=self.size)
        if frame is None:
            raise ProjectError(f"could not decode frame {i} (past the end of the video?)")
        return frame

    def stream(self, first: int = 0, count: int | None = None, size: tuple[int, int] | None = None,
               gray: bool = False) -> Iterator[tuple[int, np.ndarray]]:
        count = self.count - first if count is None else min(count, self.count - first)
        if count <= 0:
            return
        with media.FrameStream(self.path, fps=self.fps, size=size or self.size, start=self._seek_time(first) or None,
                               max_frames=count, gray=gray) as s:
            for n, frame in enumerate(s):
                yield first + n, frame

    def stream_raw(self) -> Iterator[bytes]:
        with media.FrameStream(self.path, fps=self.fps, size=self.size, start=self._seek_time(0) or None,
                               max_frames=self.count) as s:
            yield from s.raw_frames()


class PhotoSource:
    def __init__(self, project: "Project"):
        self.dir = project.photos_dir
        self.count = project.frame_count

    def read(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self.dir / f"photo_{i:06d}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise ProjectError(f"could not read photo {i}")
        return img

    def stream(self, first: int = 0, count: int | None = None, size=None, gray: bool = False
               ) -> Iterator[tuple[int, np.ndarray]]:
        last = self.count if count is None else min(self.count, first + count)
        for i in range(first, last):
            yield i, self.read(i)


# ---- project ---------------------------------------------------------------


@dataclass
class Project:
    root: Path
    manifest: dict
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def id(self) -> str:
        return self.manifest["id"]

    @property
    def is_photos(self) -> bool:
        return self.manifest.get("media_type") == "photos"

    @property
    def frame_count(self) -> int:
        return int(self.manifest.get("frame_count", 0))

    @property
    def fps(self) -> float:
        return float(self.manifest["extract"]["fps"])

    @property
    def start(self) -> float:
        return float(self.manifest["extract"]["start"])

    @property
    def width(self) -> int:
        return int(self.manifest["extract"]["width"])

    @property
    def height(self) -> int:
        return int(self.manifest["extract"]["height"])

    def frame_size(self, i: int) -> tuple[int, int]:
        if self.is_photos:
            p = self.manifest["photos"][i]
            return p["width"], p["height"]
        return self.width, self.height

    # dirs
    @property
    def photos_dir(self) -> Path:
        return self.root / "photos"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def regions_dir(self) -> Path:
        return self.root / "regions"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def source(self) -> VideoSource | PhotoSource:
        if "source" not in self._cache:
            self._cache["source"] = PhotoSource(self) if self.is_photos else VideoSource(self)
        return self._cache["source"]

    def read_original(self, i: int) -> np.ndarray:
        return self.source.read(self.check_index(i))

    def check_index(self, i: int) -> int:
        if not 0 <= i < self.frame_count:
            raise ProjectError(f"frame_index {i} out of range 0..{self.frame_count - 1}")
        return i

    # ---- time -----------------------------------------------------------
    # Times are on the source video's timeline, so they stay meaningful when the
    # video was trimmed (frame 0 is then at extract.start, not at 0s).

    def frame_time(self, i: int) -> float:
        return self.start + i / self.fps

    def time_span(self) -> tuple[float, float]:
        return self.start, self.start + self.frame_count / self.fps

    def frames_between(self, start: float | None, end: float | None) -> list[int]:
        """Frames whose display interval overlaps [start, end]. None means open-ended."""
        lo = -np.inf if start is None else start
        hi = np.inf if end is None else end
        if hi <= lo:
            raise ProjectError(f"end time ({end}) must be after start time ({start})")
        t = self.start + np.arange(self.frame_count) / self.fps
        eps = 1e-6
        mask = (t + 1 / self.fps > lo + eps) & (t < hi - eps)
        return np.flatnonzero(mask).tolist()

    @property
    def edit_window(self) -> dict | None:
        return self.manifest.get("edit_window")

    def edit_window_frames(self) -> list[int] | None:
        """Frames editing tools may touch, or None when there is no window (all frames)."""
        w = self.edit_window
        if w is None:
            return None
        if "window_frames" not in self._cache:
            self._cache["window_frames"] = self.frames_between(w.get("start"), w.get("end"))
        return self._cache["window_frames"]

    def in_edit_window(self, i: int) -> bool:
        frames = self.edit_window_frames()
        if frames is None:
            return True
        if "window_set" not in self._cache:
            self._cache["window_set"] = set(frames)
        return i in self._cache["window_set"]

    def set_edit_window(self, window: dict | None) -> None:
        if window is None:
            self.manifest.pop("edit_window", None)
        else:
            self.manifest["edit_window"] = window
        self._cache.pop("window_frames", None)
        self._cache.pop("window_set", None)

    # ---- detections -----------------------------------------------------

    def detections(self) -> list[dict | None]:
        if "detections" not in self._cache:
            path = self.root / "detections.json"
            dets = json.loads(path.read_text()) if path.exists() else []
            self._cache["detections"] = dets if len(dets) == self.frame_count else [None] * self.frame_count
        return self._cache["detections"]

    def set_detections(self, detections: list[dict | None]) -> None:
        tmp = self.root / "detections.json.tmp"
        tmp.write_text(json.dumps(detections, separators=(",", ":")))
        tmp.replace(self.root / "detections.json")
        self._cache["detections"] = detections

    def detection(self, i: int) -> dict | None:
        det = self.detections()[i]
        return det if det and det.get("found") else None

    def target_frames(self) -> list[int]:
        return [i for i, d in enumerate(self.detections()) if d and d.get("found")]

    # ---- edit recipe ----------------------------------------------------
    # manifest["edits"] is an ordered list of operations, each with "frames" (a range string).
    # Rendering frame i applies, in order, every operation whose frames include i.

    @property
    def ops(self) -> list[dict]:
        return self.manifest.setdefault("edits", [])

    def _index(self) -> dict[int, list[dict]]:
        if "ops_index" not in self._cache:
            index: dict[int, list[dict]] = {}
            for op in self.ops:
                for i in parse_indices(op["frames"], self.frame_count):
                    index.setdefault(i, []).append(op)
            self._cache["ops_index"] = index
        return self._cache["ops_index"]

    def ops_for(self, i: int) -> list[dict]:
        return self._index().get(i, [])

    def edited_set(self) -> set[int]:
        return set(self._index())

    def edited_indices(self) -> list[int]:
        return sorted(self._index())

    def is_edited(self, i: int) -> bool:
        return i in self._index()

    def add_op(self, op: dict, frames: list[int]) -> dict:
        op = {"id": self.manifest.get("next_op_id", 1), **op, "frames": format_ranges(frames)}
        self.manifest["next_op_id"] = op["id"] + 1
        self.ops.append(op)
        self._cache.pop("ops_index", None)
        return op

    def remove_frames(self, frames) -> int:
        """Drop every edit on these frames (they render as the original again). Returns frames affected."""
        drop = set(frames)
        affected = drop & self.edited_set()
        kept = []
        for op in self.ops:
            left = [i for i in parse_indices(op["frames"], self.frame_count) if i not in drop]
            if left:
                op["frames"] = format_ranges(left)
                kept.append(op)
        self.manifest["edits"] = kept
        self._cache.pop("ops_index", None)
        return len(affected)

    def store_asset(self, image_bgra: np.ndarray, name: str) -> str:
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.assets_dir / name), image_bgra):
            raise ProjectError(f"failed to store asset {name}")
        return name

    def asset(self, name: str) -> np.ndarray:
        cache = self._cache.setdefault("assets", {})
        if name not in cache:
            cache[name] = load_image_bgra(self.assets_dir / name)
        return cache[name]

    # ---- persistence ----------------------------------------------------

    def save(self) -> None:
        self.manifest["updated_at"] = time.time()
        tmp = self.root / "manifest.json.tmp"
        tmp.write_text(json.dumps(self.manifest, indent=1))
        tmp.replace(self.root / "manifest.json")

    def log(self, op: str, **details) -> None:
        history = self.manifest.setdefault("history", [])
        history.append({"op": op, "at": time.time(), **details})
        del history[:-200]


def create_project(source: dict, name_hint: str = "") -> Project:
    slug = re.sub(r"[^a-z0-9]+", "-", name_hint.lower()).strip("-")[:24] or "media"
    project_id = f"{slug}-{uuid.uuid4().hex[:8]}"
    root = workspace_root() / project_id
    root.mkdir(parents=True)
    manifest = {
        "id": project_id,
        "version": 2,
        "created_at": time.time(),
        "source": source,
        "frame_count": 0,
        "edits": [],
        "history": [],
    }
    project = Project(root=root, manifest=manifest)
    project.save()
    return project


def load_project(project_id: str) -> Project:
    if not _ID_RE.match(project_id or ""):
        raise ProjectError(f"invalid project_id {project_id!r}")
    root = workspace_root() / project_id
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ProjectError(f"project {project_id!r} not found (workspace: {workspace_root()})")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 2:
        raise ProjectError(f"project {project_id!r} was made by an older flowedits; open the video again")
    return Project(root=root, manifest=manifest)


def list_projects() -> list[dict]:
    root = workspace_root()
    if not root.exists():
        return []
    out = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            m = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            {
                "project_id": m["id"],
                "media_type": m.get("media_type", "video"),
                "source": m.get("source", {}).get("user_path") or m.get("source", {}).get("path"),
                "frame_count": m.get("frame_count"),
                "has_target": bool(m.get("target")),
                "edits": len(m.get("edits", [])),
                "created_at": m.get("created_at"),
            }
        )
    return sorted(out, key=lambda p: p.get("created_at") or 0, reverse=True)


def delete_project(project_id: str) -> None:
    shutil.rmtree(load_project(project_id).root)


def parse_indices(spec, count: int) -> list[int]:
    """Accept None, an int, a list of ints, or a string like "0-20,35,40-".

    Ranges are inclusive. Out-of-range indices are dropped.
    """
    if spec is None or spec == "" or spec == "all":
        return list(range(count))
    if isinstance(spec, int):
        spec = [spec]
    if isinstance(spec, str):
        result: list[int] = []
        for part in spec.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                lo_i = int(lo) if lo else 0
                hi_i = int(hi) if hi else count - 1
                result.extend(range(max(0, lo_i), min(count - 1, hi_i) + 1))
            else:
                result.append(int(part))
        spec = result
    return sorted(i for i in set(int(x) for x in spec) if 0 <= i < count)


_TIME_RE = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)s?$")


def parse_time(value) -> float | None:
    """Seconds from a number or a string like "90", "90s", "1:30", "00:01:30.5"."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        m = _TIME_RE.match(str(value).strip())
        if not m:
            raise ProjectError(f"invalid time {value!r}; use seconds (90, 90.5, '90s') or 'MM:SS' / 'HH:MM:SS'")
        a, b, s = m.groups()
        # One colon -> MM:SS, two -> HH:MM:SS.
        hours, minutes = (int(a), int(b)) if b is not None else (0, int(a) if a else 0)
        seconds = hours * 3600 + minutes * 60 + float(s)
    if seconds < 0:
        raise ProjectError("times must not be negative")
    return seconds


def load_image_bgra(path: str | Path) -> np.ndarray:
    """Load any image as 8-bit BGRA (alpha=255 where the file has none)."""
    path = Path(path).expanduser()
    if not path.exists():
        raise ProjectError(f"image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ProjectError(f"could not decode image: {path}")
    if img.dtype != np.uint8:
        img = (img / (np.iinfo(img.dtype).max / 255.0)).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory).expanduser()
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)
