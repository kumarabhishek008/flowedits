"""Pieces that make long videos work: exact frame seeking, strided detection, parallel chunks, jobs."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from test_pipeline import env, make_logo  # noqa: F401  (fixture)

BITS = 9  # frame index encoded as 9 big black/white blocks -> up to 511 frames


def _coded_frame(i: int, w: int = 320, h: int = 96) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    block = w // BITS
    for b in range(BITS):
        if (i >> b) & 1:
            img[:, b * block:(b + 1) * block] = 255
    return img


def _decode_index(img: np.ndarray) -> int:
    h, w = img.shape[:2]
    block = w // BITS
    return sum(1 << b for b in range(BITS) if img[h // 2, b * block + block // 2].mean() > 127)


@pytest.fixture()
def coded_video(tmp_path):
    from flowedits import media

    d = tmp_path / "coded"
    d.mkdir()
    for i in range(300):
        cv2.imwrite(str(d / f"f_{i:04d}.png"), _coded_frame(i))
    video = tmp_path / "coded.mp4"
    media.assemble_video(sorted(d.glob("*.png")), video, fps=30)
    return video


def test_seek_and_stream_return_exact_frames(env, coded_video):  # noqa: F811
    ops = env[0]
    from flowedits.workspace import load_project

    result, _ = ops.split_video(str(coded_video))
    assert result["frame_count"] == 300
    project = load_project(result["project_id"])
    for i in (0, 1, 2, 29, 30, 149, 150, 151, 298, 299):
        assert _decode_index(project.read_original(i)) == i, i
    got = [(i, _decode_index(f)) for i, f in project.source.stream(137, 25)]
    assert all(i == j for i, j in got) and len(got) == 25
    raw = list(project.source.stream_raw())
    assert len(raw) == 300

    # Trimmed: frame 0 is at 2.0 s on the source timeline -> source frame 60.
    trimmed, _ = ops.split_video(str(coded_video), trim_start_time=2.0, trim_end_time=6.0)
    project = load_project(trimmed["project_id"])
    assert trimmed["frame_count"] == 120
    assert [_decode_index(project.read_original(i)) for i in (0, 1, 59, 119)] == [60, 61, 119, 179]


def _logo_frames(n: int, present: range) -> list[tuple[int, np.ndarray]]:
    logo = make_logo()
    rng = np.random.default_rng(1)
    bg = cv2.GaussianBlur(rng.integers(40, 200, (360, 640, 3), dtype=np.uint8), (0, 0), 5)
    frames = []
    for i in range(n):
        f = bg.copy()
        if i in present:
            x = 40 + (8 * i) % 440
            f[100:180, x:x + 120] = logo
        frames.append((i, f))
    return frames


def test_strided_detection_backfills_skipped_frames():
    from flowedits.detect import TargetDetector, detect_sequence

    frames = _logo_frames(40, range(7, 33))
    det = TargetDetector(cv2.cvtColor(make_logo(), cv2.COLOR_BGR2BGRA))
    results = detect_sequence(det, iter(frames), stride=5)
    found = sorted(i for i, r in results.items() if r["found"])
    assert found == list(range(7, 33))
    # While lost, most frames were skipped rather than fully searched.
    assert sum(1 for r in results.values() if r.get("skipped")) >= 8

    # A long absence widens the stride (up to 15) but the reappearance is still exact.
    frames = _logo_frames(80, range(53, 70))
    results = detect_sequence(TargetDetector(cv2.cvtColor(make_logo(), cv2.COLOR_BGR2BGRA)), iter(frames),
                              stride=5, max_stride=15)
    assert sorted(i for i, r in results.items() if r["found"]) == list(range(53, 70))
    full_searches_while_lost = sum(1 for i in range(53) if not results[i].get("skipped"))
    assert full_searches_while_lost <= 8  # stride 5 -> 10 -> 15 instead of 11 searches at a fixed 5


def test_parallel_chunks_match_sequential():
    from flowedits.detect import TargetDetector, detect_parallel, detect_sequence

    frames = _logo_frames(60, range(5, 50))
    target = cv2.cvtColor(make_logo(), cv2.COLOR_BGR2BGRA)
    seq = detect_sequence(TargetDetector(target), iter(frames))
    par = detect_parallel(lambda: TargetDetector(target), lambda s, n: iter(frames[s:s + n]), 0, 60,
                          workers=4, stride=1)
    assert sorted(par) == list(range(60))
    for i in range(60):
        assert par[i]["found"] == seq[i]["found"], i
        if seq[i]["found"]:
            assert np.allclose(par[i]["quad"], seq[i]["quad"], atol=1.5), i


def test_jobs_progress_busy_and_cancel():
    from flowedits.jobs import JobManager
    from flowedits.workspace import ProjectError

    jobs = JobManager()
    gate = threading.Event()

    def work(job):
        for n in range(1000):
            job.update(n / 1000, "working")
            gate.wait(0.01)
        return {"ok": True}

    job = jobs.start("p1", "render", work)
    assert not job.wait(0.05)
    with pytest.raises(ProjectError, match="busy"):
        jobs.ensure_idle("p1")
    jobs.ensure_idle("p2")  # other projects are unaffected
    assert job.to_dict()["status"] == "running"
    job.cancel()
    assert job.wait(5)
    assert job.status == "cancelled"
    jobs.ensure_idle("p1")

    done = jobs.start("p1", "quick", lambda j: ({"n": 1}, b"img"))
    assert done.wait(5) and done.status == "done" and done.result == {"n": 1} and done.preview == b"img"
    failing = jobs.start("p1", "bad", lambda j: 1 / 0)
    assert failing.wait(5) and failing.status == "error" and "ZeroDivisionError" in failing.error
    time.sleep(0)


@pytest.mark.parametrize("op", ["replace", "replace_alpha", "remove", "blur", "pixelate", "fill"])
def test_region_edit_matches_full_frame(op):
    """Editing only a crop around the target must give the same pixels as editing the whole frame."""
    from flowedits import edit

    rng = np.random.default_rng(3)
    frame = cv2.GaussianBlur(rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8), (0, 0), 3)
    rep = cv2.cvtColor(rng.integers(0, 255, (90, 140, 3), dtype=np.uint8), cv2.COLOR_BGR2BGRA)
    if op == "replace_alpha":
        rep[:, :30, 3] = 0  # partly transparent -> replace inpaints first
    fns = {
        "replace": (12, lambda f, q: edit.replace(f, q, rep, feather=2, padding=4)),
        "replace_alpha": (60, lambda f, q: edit.replace(f, q, rep, feather=2, padding=4)),
        "remove": (52, lambda f, q: edit.remove(f, q, padding=4)),
        "blur": (62, lambda f, q: edit.blur(f, q, strength=25, padding=4)),
        "pixelate": (6, lambda f, q: edit.pixelate(f, q, block=12, padding=4)),
        "fill": (6, lambda f, q: edit.fill(f, q, (0, 255, 0), padding=4)),
    }
    margin, fn = fns[op]
    quads = [
        [[400, 200], [560, 215], [555, 330], [395, 310]],   # perspective
        [[-30, 600], [90, 600], [90, 700], [-30, 700]],     # partly outside the frame
    ]
    for quad in quads:
        full = fn(frame.copy(), quad)
        region = edit.on_region(frame.copy(), quad, margin, fn)
        diff = np.abs(full.astype(int) - region.astype(int))
        assert diff.max() <= 2, (op, quad, diff.max())
