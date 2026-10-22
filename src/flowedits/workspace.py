"""Project workspace: where frames, edits, detections and outputs live on disk.

Layout of one project:

    <workspace>/<project_id>/
        manifest.json      source info, extraction settings, target detections
        frames/            original extracted frames (never modified)
        edited/            edited frames; when present they override frames/
        regions/           rectified crops of the target, exported for hand editing
        target.png         copy of the target image being tracked
        output/            assembled videos
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(
    os.environ.get("FLOWEDITS_WORKSPACE", Path.home() / ".flowedits" / "projects")
).expanduser()

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class ProjectError(ValueError):
    pass


@dataclass
class Project:
    root: Path
    manifest: dict

    @property
    def id(self) -> str:
        return self.manifest["id"]

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def edited_dir(self) -> Path:
        return self.root / "edited"

    @property
    def regions_dir(self) -> Path:
        return self.root / "regions"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def frame_count(self) -> int:
        return int(self.manifest.get("frame_count", 0))

    @property
    def ext(self) -> str:
        return self.manifest["extract"]["format"]

    @property
    def fps(self) -> float:
        return float(self.manifest["extract"]["fps"])

    def frame_name(self, i: int) -> str:
        return f"frame_{i:06d}.{self.ext}"

    def original_path(self, i: int) -> Path:
        return self.frames_dir / self.frame_name(i)

    def edited_path(self, i: int) -> Path:
        return self.edited_dir / self.frame_name(i)

    def current_path(self, i: int) -> Path:
        edited = self.edited_path(i)
        return edited if edited.exists() else self.original_path(i)

    def is_edited(self, i: int) -> bool:
        return self.edited_path(i).exists()

    def edited_indices(self) -> list[int]:
        if not self.edited_dir.exists():
            return []
        out = []
        for p in self.edited_dir.glob(f"frame_*.{self.ext}"):
            try:
                out.append(int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(out)

    def check_index(self, i: int) -> int:
        if not 0 <= i < self.frame_count:
            raise ProjectError(f"frame_index {i} out of range 0..{self.frame_count - 1}")
        return i

    # ---- frame IO -------------------------------------------------------

    def read_frame(self, i: int, edited: bool = True) -> np.ndarray:
        path = self.current_path(i) if edited else self.original_path(i)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ProjectError(f"could not read frame {i} at {path}")
        return img

    def write_edited(self, i: int, img: np.ndarray) -> Path:
        self.edited_dir.mkdir(parents=True, exist_ok=True)
        path = self.edited_path(i)
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if self.ext == "jpg" else []
        if not cv2.imwrite(str(path), img, params):
            raise ProjectError(f"failed to write {path}")
        return path

    def reset(self, indices: list[int] | None = None) -> int:
        targets = self.edited_indices() if indices is None else indices
        removed = 0
        for i in targets:
            p = self.edited_path(i)
            if p.exists():
                p.unlink()
                removed += 1
        return removed

    # ---- detections -----------------------------------------------------

    def detections(self) -> list[dict | None]:
        target = self.manifest.get("target")
        if not target:
            return [None] * self.frame_count
        return target["detections"]

    def detection(self, i: int) -> dict | None:
        det = self.detections()[i]
        return det if det and det.get("found") else None

    def target_frames(self) -> list[int]:
        return [i for i, d in enumerate(self.detections()) if d and d.get("found")]

    def save(self) -> None:
        self.manifest["updated_at"] = time.time()
        tmp = self.root / "manifest.json.tmp"
        tmp.write_text(json.dumps(self.manifest, indent=1))
        tmp.replace(self.root / "manifest.json")

    def log(self, op: str, **details) -> None:
        self.manifest.setdefault("history", []).append({"op": op, "at": time.time(), **details})


def create_project(source: dict, name_hint: str = "") -> Project:
    slug = re.sub(r"[^a-z0-9]+", "-", name_hint.lower()).strip("-")[:24] or "video"
    project_id = f"{slug}-{uuid.uuid4().hex[:8]}"
    root = WORKSPACE / project_id
    (root / "frames").mkdir(parents=True)
    manifest = {
        "id": project_id,
        "created_at": time.time(),
        "source": source,
        "frame_count": 0,
        "history": [],
    }
    project = Project(root=root, manifest=manifest)
    project.save()
    return project


def load_project(project_id: str) -> Project:
    if not _ID_RE.match(project_id or ""):
        raise ProjectError(f"invalid project_id {project_id!r}")
    root = WORKSPACE / project_id
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ProjectError(f"project {project_id!r} not found (workspace: {WORKSPACE})")
    return Project(root=root, manifest=json.loads(manifest_path.read_text()))


def list_projects() -> list[dict]:
    if not WORKSPACE.exists():
        return []
    out = []
    for manifest_path in sorted(WORKSPACE.glob("*/manifest.json")):
        try:
            m = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            {
                "project_id": m["id"],
                "source": m.get("source", {}).get("path"),
                "frame_count": m.get("frame_count"),
                "has_target": bool(m.get("target")),
                "created_at": m.get("created_at"),
            }
        )
    return sorted(out, key=lambda p: p.get("created_at") or 0, reverse=True)


def delete_project(project_id: str) -> None:
    project = load_project(project_id)
    shutil.rmtree(project.root)


def parse_indices(spec, count: int) -> list[int]:
    """Accept None, an int, a list of ints, or a string like "0-20,35,40-".

    Ranges are inclusive. Out-of-range indices are dropped.
    """
    if spec is None or spec == "" or spec == "all":
        return list(range(count))
    if isinstance(spec, int):
        spec = [spec]
    if isinstance(spec, str):
        result: set[int] = set()
        for part in spec.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                lo_i = int(lo) if lo else 0
                hi_i = int(hi) if hi else count - 1
                result.update(range(lo_i, hi_i + 1))
            else:
                result.add(int(part))
        spec = result
    return sorted(i for i in set(int(x) for x in spec) if 0 <= i < count)


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
