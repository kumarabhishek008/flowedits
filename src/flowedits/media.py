"""ffmpeg wrappers: probe a video, split it into frames, join frames back into a video."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2


class FFmpegError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """System ffmpeg if available, otherwise the static build bundled with imageio-ffmpeg."""
    exe = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depends on environment
        raise FFmpegError("ffmpeg not found: install ffmpeg or the imageio-ffmpeg package") from exc


def run_ffmpeg(args: list[str]) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}")


def probe(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FFmpegError(f"video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FFmpegError(f"could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    # `ffmpeg -i` with no output exits non-zero but prints stream info to stderr.
    info = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True
    ).stderr
    duration = None
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    if not fps:
        m = re.search(r"(\d+(?:\.\d+)?)\s*fps", info)
        fps = float(m.group(1)) if m else 25.0
    if duration is None and fps:
        duration = frame_count / fps

    return {
        "path": str(path),
        "width": width,
        "height": height,
        "fps": round(fps, 4),
        "frame_count": frame_count,
        "duration": round(duration or 0.0, 3),
        "has_audio": bool(re.search(r"Stream #\S+.*Audio:", info)),
    }


def extract_frames(
    video: str | Path,
    out_dir: Path,
    *,
    fps: float,
    image_format: str = "png",
    start: float | None = None,
    duration: float | None = None,
    max_frames: int | None = None,
    max_width: int | None = None,
) -> int:
    """Split a video into constant-rate frames named frame_000000.<fmt>. Returns frame count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    args: list[str] = []
    if start:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(video)]
    if duration:
        args += ["-t", f"{duration:.3f}"]
    filters = [f"fps={fps}"]
    if max_width:
        filters.append(f"scale='min({int(max_width)},iw)':-2")
    args += ["-vf", ",".join(filters), "-an"]
    if max_frames:
        args += ["-frames:v", str(int(max_frames))]
    if image_format == "jpg":
        args += ["-q:v", "2"]
    args += ["-start_number", "0", str(out_dir / f"frame_%06d.{image_format}")]
    run_ffmpeg(args)
    return len(list(out_dir.glob(f"frame_*.{image_format}")))


def assemble_video(
    frame_paths: list[Path],
    output: Path,
    *,
    fps: float,
    audio_source: str | None = None,
    audio_start: float | None = None,
    audio_duration: float | None = None,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Join an ordered list of frame images into a video, optionally muxing audio from the source."""
    if not frame_paths:
        raise FFmpegError("no frames to assemble")
    ext = frame_paths[0].suffix
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()

    with tempfile.TemporaryDirectory(prefix="assemble-", dir=output.parent) as tmp:
        tmp_dir = Path(tmp)
        # ffmpeg's image2 demuxer needs one contiguous numbered sequence; mix edited and
        # original frames by symlinking them into place in order.
        for n, src in enumerate(frame_paths):
            link = tmp_dir / f"seq_{n:06d}{ext}"
            try:
                link.symlink_to(src.resolve())
            except OSError:
                shutil.copyfile(src, link)

        args = ["-framerate", f"{fps}", "-start_number", "0", "-i", str(tmp_dir / f"seq_%06d{ext}")]
        if audio_source:
            if audio_start:
                args += ["-ss", f"{audio_start:.3f}"]
            if audio_duration:
                args += ["-t", f"{audio_duration:.3f}"]
            args += ["-i", audio_source, "-map", "0:v:0", "-map", "1:a:0?"]

        if suffix == ".gif":
            args += ["-vf", "split[a][b];[a]palettegen[p];[b][p]paletteuse"]
        elif suffix == ".webm":
            args += ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-pix_fmt", "yuv420p"]
            if audio_source:
                args += ["-c:a", "libopus"]
        else:
            # yuv420p needs even dimensions.
            args += [
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            ]
            if suffix in (".mp4", ".mov", ".m4v"):
                args += ["-movflags", "+faststart"]
            if audio_source:
                args += ["-c:a", "aac", "-b:a", "192k"]
        if audio_source:
            args += ["-shortest"]
        args.append(str(output))
        run_ffmpeg(args)
    return output
