"""Turn whatever the user pasted into a real file path.

Users paste paths in many shapes: quoted, with shell-escaped spaces (terminal drag and drop),
as file:// URLs, with ~, or only a file name ("demo.mov"). macOS screen recordings also contain
a narrow no-break space (U+202F) before "PM" that never survives copy/paste as typed text.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlparse

from .workspace import IMAGE_EXTS, ProjectError

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp", ".ts",
              ".mts", ".m2ts", ".gif"}
KIND_EXTS = {"video": VIDEO_EXTS, "image": IMAGE_EXTS, "any": VIDEO_EXTS | IMAGE_EXTS}

SEARCH_DEPTH = 4
MAX_ENTRIES = 60_000  # bound the scan so a huge home folder can't stall a tool call
_SKIP_DIRS = {"Library", "node_modules", "__pycache__", "site-packages", "venv", ".venv", "Applications"}


def clean_path(raw: str) -> str:
    """Strip quotes, file:// and shell escapes; expand ~ and $VARS. Does not touch the filesystem."""
    s = str(raw).strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"`":
        s = s[1:-1].strip()
    if s.lower().startswith("file://"):
        s = unquote(urlparse(s).path)
    if os.sep == "/" and "\\" in s:
        s = re.sub(r"\\(.)", r"\1", s)  # "My\ Video.mov" -> "My Video.mov"
    return os.path.expandvars(os.path.expanduser(s))


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFC", name).replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", name).strip().casefold()


def search_roots() -> list[Path]:
    home = Path.home()
    roots = [Path(p).expanduser() for p in os.environ.get("FLOWEDITS_SEARCH_DIRS", "").split(os.pathsep) if p]
    roots += [home / d for d in ("Desktop", "Downloads", "Movies", "Documents", "Pictures")]
    roots += [home, Path.cwd()]
    seen, out = set(), []
    for r in roots:
        try:
            key = r.resolve()
        except OSError:
            continue
        if key not in seen and r.is_dir():
            seen.add(key)
            out.append(r)
    return out


def _walk(root: Path, depth: int, budget: list[int]):
    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for e in entries:
        budget[0] -= 1
        if budget[0] <= 0:
            return
        if e.name.startswith("."):
            continue
        try:
            if e.is_dir(follow_symlinks=False):
                if depth > 0 and e.name not in _SKIP_DIRS:
                    yield from _walk(Path(e.path), depth - 1, budget)
            elif e.is_file():
                yield Path(e.path)
        except OSError:
            continue


def find_media(query: str, kind: str = "video", limit: int = 20) -> list[Path]:
    """Files under the usual folders whose name matches `query`, best matches first, newest first."""
    exts = KIND_EXTS[kind]
    name = Path(clean_path(query)).name
    q = _norm(name)
    q_stem = _norm(Path(name).stem) if Path(name).suffix.lower() in exts else q
    exact, stem, partial = [], [], []
    budget = [MAX_ENTRIES]
    seen: set[Path] = set()
    for root in search_roots():
        # home and cwd are scanned shallowly; the named folders more deeply.
        depth = 1 if root in (Path.home(), Path.cwd()) else SEARCH_DEPTH
        for p in _walk(root, depth, budget):
            if p.suffix.lower() not in exts or p in seen:
                continue
            seen.add(p)
            n = _norm(p.name)
            if n == q:
                exact.append(p)
            elif _norm(p.stem) == q_stem:
                stem.append(p)
            elif len(q_stem) >= 3 and q_stem in n:
                partial.append(p)

    def newest(paths):
        return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)

    return (newest(exact) + newest(stem) + newest(partial))[:limit]


def resolve_media_path(raw: str, kind: str = "video", allow_dir: bool = False) -> Path:
    """Find the file the user meant, or raise ProjectError listing candidates / places searched."""
    if not raw or not str(raw).strip():
        raise ProjectError(f"no {kind} path given")
    p = Path(clean_path(raw))

    if p.exists():
        if p.is_dir() and not allow_dir:
            raise ProjectError(f"{p} is a folder, expected a {kind} file")
        return p.resolve()

    # Same folder, but the name differs only in unicode spacing/case (macOS screen recordings).
    if p.parent != Path(".") and p.parent.is_dir():
        matches = [c for c in p.parent.iterdir() if _norm(c.name) == _norm(p.name)]
        if len(matches) == 1:
            return matches[0].resolve()

    candidates = find_media(p.name, kind)
    exact = [c for c in candidates if _norm(c.name) == _norm(p.name)]
    if len(exact) == 1:
        return exact[0].resolve()
    if len(candidates) == 1:
        return candidates[0].resolve()
    if candidates:
        listed = "\n".join(f"  - {c}" for c in candidates[:10])
        raise ProjectError(f"{raw!r} not found as given; several {kind} files match, pass the full path of one:\n"
                           f"{listed}")
    roots = ", ".join(str(r) for r in search_roots())
    raise ProjectError(f"{kind} not found: {raw!r}. Searched by name in: {roots}. Pass the full path, or add folders "
                       f"to FLOWEDITS_SEARCH_DIRS.")
