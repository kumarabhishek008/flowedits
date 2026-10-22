"""ffmpeg wrappers: probe a video, stream its frames, encode frames back into a video.

Nothing is extracted to disk: decoding and encoding run as ffmpeg pipes of raw BGR frames, so a
10-minute 30 fps video (18,000 frames) costs no more disk than the output file.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


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
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args]
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
    cap.release()

    # `ffmpeg -i` with no output exits non-zero but prints stream info to stderr.
    info = subprocess.run(
        [ffmpeg_exe(), "-nostdin", "-hide_banner", "-i", str(path)], capture_output=True, text=True
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

    # Decode one frame for the real (rotation-applied) size; container metadata can disagree.
    first = read_frame_at(path, 0.0)
    if first is None:
        raise FFmpegError(f"could not decode any frame from {path}")
    return {
        "path": str(path),
        "width": int(first.shape[1]),
        "height": int(first.shape[0]),
        "fps": round(fps, 4),
        "frame_count": frame_count,
        "duration": round(duration or 0.0, 3),
        "has_audio": bool(re.search(r"Stream #\S+.*Audio:", info)),
    }


def even(n: float) -> int:
    return max(2, int(round(n / 2)) * 2)


class FrameStream:
    """Iterate decoded frames (BGR uint8) of a video as a constant-rate sequence.

    Seeks with `start` (input-side, accurate), resamples to `fps`, scales to exactly `size`.
    Use as a context manager so the ffmpeg process is always cleaned up.
    """

    def __init__(self, src: str | Path, *, fps: float | None = None, size: tuple[int, int] | None = None,
                 start: float | None = None, max_frames: int | None = None, gray: bool = False,
                 threads: int | None = None):
        self.size = size
        self.channels = 1 if gray else 3
        filters = []
        if fps:
            filters.append(f"fps={fps}")
        if size:
            filters.append(f"scale={size[0]}:{size[1]}:flags=area")
        args = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error"]
        if threads:
            # Several streams decode in parallel during detection; cap each so they don't oversubscribe.
            args += ["-threads", str(threads), "-filter_threads", "1"]
        if start and start > 0:
            args += ["-ss", f"{start:.6f}"]
        args += ["-i", str(src), "-an", "-sn", "-map", "0:v:0"]
        if filters:
            args += ["-vf", ",".join(filters)]
        if max_frames:
            args += ["-frames:v", str(int(max_frames))]
        # Gray output (detection) is a third of the bytes and skips a colour conversion per frame.
        args += ["-f", "rawvideo", "-pix_fmt", "gray" if gray else "bgr24", "pipe:1"]
        if size is None:
            raise ValueError("FrameStream needs an explicit size")
        self._frame_bytes = size[0] * size[1] * self.channels
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=self._stderr,
                                      bufsize=self._frame_bytes)

    def read_raw(self) -> bytes | None:
        data = self._proc.stdout.read(self._frame_bytes)
        if len(data) < self._frame_bytes:
            self._finish()
            return None
        return data

    def raw_frames(self) -> Iterator[bytes]:
        while (data := self.read_raw()) is not None:
            yield data

    def __iter__(self) -> Iterator[np.ndarray]:
        w, h = self.size
        shape = (h, w) if self.channels == 1 else (h, w, 3)
        for data in self.raw_frames():
            yield np.frombuffer(data, np.uint8).reshape(shape)

    def _finish(self) -> None:
        rc = self._proc.wait()
        if rc not in (0, None):
            self._stderr.seek(0)
            raise FFmpegError(f"ffmpeg decode failed ({rc}): {self._stderr.read().decode(errors='replace')[-2000:]}")

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        self._proc.stdout.close()
        self._stderr.close()

    def __enter__(self) -> "FrameStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_frame_at(src: str | Path, t: float, *, size: tuple[int, int] | None = None) -> np.ndarray | None:
    """Single frame at time t (seconds). Size defaults to the decoded size."""
    if size is None:
        # Let ffmpeg pick the size and hand over a PNG; only used for probing.
        args = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error"]
        if t > 0:
            args += ["-ss", f"{t:.6f}"]
        args += ["-i", str(src), "-map", "0:v:0", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
        proc = subprocess.run(args, capture_output=True)
        if proc.returncode != 0 or not proc.stdout:
            return None
        return cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
    with FrameStream(src, size=size, start=t, max_frames=1) as stream:
        for frame in stream:
            return frame.copy()
    return None


def _output_args(suffix: str, crf: int, preset: str, with_audio: bool) -> list[str]:
    if suffix == ".gif":
        return ["-vf", "split[a][b];[a]palettegen[p];[b][p]paletteuse"]
    if suffix == ".webm":
        args = ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-row-mt", "1", "-pix_fmt", "yuv420p"]
        return args + (["-c:a", "libopus"] if with_audio else [])
    # yuv420p needs even dimensions.
    args = ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p"]
    if suffix in (".mp4", ".mov", ".m4v"):
        args += ["-movflags", "+faststart"]
    return args + (["-c:a", "aac", "-b:a", "192k"] if with_audio else [])


def _audio_input(audio_source: str | None, audio_start: float | None, audio_duration: float | None) -> list[str]:
    if not audio_source:
        return []
    args = []
    if audio_start:
        args += ["-ss", f"{audio_start:.6f}"]
    if audio_duration:
        args += ["-t", f"{audio_duration:.6f}"]
    return args + ["-i", audio_source]


class Encoder:
    """Pipe raw BGR frames into ffmpeg and encode them (optionally muxing audio from the source)."""

    def __init__(self, output: Path, *, size: tuple[int, int], fps: float, audio_source: str | None = None,
                 audio_start: float | None = None, audio_duration: float | None = None, crf: int = 18,
                 preset: str = "fast"):
        output.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        args = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "pipe:0"]
        args += _audio_input(audio_source, audio_start, audio_duration)
        args += ["-map", "0:v:0"] + (["-map", "1:a:0?"] if audio_source else [])
        args += _output_args(output.suffix.lower(), crf, preset, bool(audio_source))
        if audio_source:
            args += ["-shortest"]
        args.append(str(output))
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr,
                                      bufsize=w * h * 3)

    def _error(self, prefix: str) -> FFmpegError:
        self._proc.wait()
        self._stderr.seek(0)
        return FFmpegError(f"{prefix}: {self._stderr.read().decode(errors='replace')[-2000:]}")

    def write(self, frame: bytes | np.ndarray) -> None:
        try:
            self._proc.stdin.write(frame if isinstance(frame, (bytes, bytearray, memoryview)) else
                                   np.ascontiguousarray(frame).data)
        except BrokenPipeError:
            raise self._error("ffmpeg encoder stopped") from None

    def close(self) -> None:
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        if self._proc.wait() != 0:
            raise self._error("ffmpeg encode failed")
        self._stderr.close()

    def abort(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        self._stderr.close()


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
    """Join an ordered list of image files into a video (used to build inputs, e.g. in tests)."""
    if not frame_paths:
        raise FFmpegError("no frames to assemble")
    ext = frame_paths[0].suffix
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flowedits-assemble-") as tmp:
        tmp_dir = Path(tmp)
        # ffmpeg's image2 demuxer needs one contiguous numbered sequence.
        for n, src in enumerate(frame_paths):
            link = tmp_dir / f"seq_{n:06d}{ext}"
            try:
                link.symlink_to(src.resolve())
            except OSError:
                shutil.copyfile(src, link)
        args = ["-framerate", f"{fps}", "-start_number", "0", "-i", str(tmp_dir / f"seq_%06d{ext}")]
        args += _audio_input(audio_source, audio_start, audio_duration)
        if audio_source:
            args += ["-map", "0:v:0", "-map", "1:a:0?"]
        args += _output_args(output.suffix.lower(), crf, preset, bool(audio_source))
        if audio_source:
            args += ["-shortest"]
        run_ffmpeg(args + [str(output)])
    return output
