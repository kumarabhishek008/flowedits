"""End-to-end: synthetic video with a moving logo -> split -> locate -> edit -> assemble."""

from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np
import pytest

FPS = 24
N_FRAMES = 48
W, H = 640, 360


def make_logo() -> np.ndarray:
    logo = np.full((80, 120, 3), 255, np.uint8)
    cv2.rectangle(logo, (0, 0), (119, 79), (20, 20, 200), 6)
    cv2.circle(logo, (35, 40), 20, (200, 60, 20), -1)
    cv2.putText(logo, "AB", (62, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 120, 10), 3)
    return logo


def logo_pos(i: int) -> tuple[int, int]:
    return 40 + i * 9, 60 + int(40 * np.sin(i / 8))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOWEDITS_WORKSPACE", str(tmp_path / "ws"))
    from flowedits import media, operations, workspace

    importlib.reload(workspace)
    importlib.reload(operations)

    logo = make_logo()
    frames_dir = tmp_path / "src_frames"
    frames_dir.mkdir()
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(N_FRAMES):
        bg = np.dstack([(xx * 255 // W), (yy * 255 // H), np.full_like(xx, (i * 5) % 255)]).astype(np.uint8)
        x, y = logo_pos(i)
        bg[y:y + 80, x:x + 120] = logo
        cv2.imwrite(str(frames_dir / f"f_{i:04d}.png"), bg)

    audio = tmp_path / "tone.wav"
    media.run_ffmpeg(["-f", "lavfi", "-i", f"sine=frequency=440:duration={N_FRAMES / FPS}", str(audio)])
    video = tmp_path / "input.mp4"
    media.assemble_video(sorted(frames_dir.glob("*.png")), video, fps=FPS, audio_source=str(audio))

    logo_path = tmp_path / "logo.png"
    cv2.imwrite(str(logo_path), logo)
    replacement = np.zeros((80, 120, 3), np.uint8)
    replacement[:] = (255, 0, 255)  # magenta
    rep_path = tmp_path / "replacement.png"
    cv2.imwrite(str(rep_path), replacement)
    return operations, video, logo_path, rep_path, tmp_path


def test_split_locate_replace_assemble(env):
    ops, video, logo_path, rep_path, tmp_path = env

    result, preview = ops.split_video(str(video), str(logo_path))
    assert result["frame_count"] == N_FRAMES
    assert preview is not None
    target = result["target"]
    assert target["frames_with_target"] == N_FRAMES, target

    pid = result["project_id"]
    # Detected position should match where we drew the logo (h264 + smoothing: allow a few px).
    for i in (0, 20, 47):
        bbox = ops.list_frames(pid, [i])["frames"][0]["bbox"]
        x, y = logo_pos(i)
        assert abs(bbox[0] - x) <= 4 and abs(bbox[1] - y) <= 4, (i, bbox)

    edited, _ = ops.apply_to_target(pid, "replace", str(rep_path), padding=0, feather=0)
    assert edited["frames_edited"] == N_FRAMES

    _, jpeg = ops.view_frame(pid, 10)
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    x, y = logo_pos(10)
    center = frame[y + 40, x + 60].astype(int)
    assert abs(center[0] - 255) < 30 and center[1] < 40 and abs(center[2] - 255) < 30, center

    out = ops.assemble(pid, str(tmp_path / "out.mp4"))
    assert Path(out["output_path"]).exists()
    assert out["audio"] is True
    assert abs(out["duration"] - N_FRAMES / FPS) < 0.15


def test_remove_regions_and_reset(env):
    ops, video, logo_path, _, tmp_path = env
    result, _ = ops.split_video(str(video), str(logo_path), image_format="jpg", end_time=1.0)
    pid = result["project_id"]
    assert result["frame_count"] == FPS

    removed, _ = ops.apply_to_target(pid, "remove", frame_indices="0-5")
    assert removed["frames_edited"] == 6

    exported = ops.export_target_regions(pid, frame_indices=[3, 4])
    crop_path = Path(exported["regions_dir"]) / "region_000003.png"
    crop = cv2.imread(str(crop_path))
    crop[:] = (0, 255, 0)
    cv2.imwrite(str(crop_path), crop)
    imported, _ = ops.import_target_regions(pid)
    assert imported["frames_updated"] == 1 and imported["unchanged_skipped"] == 1

    sheet_info, _ = ops.contact_sheet(pid, only_edited=True)
    assert set(sheet_info["frames_shown"]) <= set(range(6))

    assert ops.reset_frames(pid)["frames_reverted"] == 6
    assert ops.project_status(pid)["edited_count"] == 0


def test_perspective_target_with_features(env):
    ops, _, logo_path, rep_path, tmp_path = env
    from flowedits import media

    # A larger, textured target viewed under changing perspective.
    rng = np.random.default_rng(0)
    poster = cv2.resize(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8), (320, 240), interpolation=cv2.INTER_NEAREST)
    cv2.putText(poster, "POSTER", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
    poster_path = tmp_path / "poster.png"
    cv2.imwrite(str(poster_path), poster)

    frames_dir = tmp_path / "persp"
    frames_dir.mkdir()
    src = np.float32([[0, 0], [320, 0], [320, 240], [0, 240]])
    for i in range(12):
        d = i * 3
        dst = np.float32([[150 + d, 60], [450 - d, 80 + d], [460, 300], [140, 290 - d]])
        frame = np.full((H, W, 3), 90, np.uint8)
        m = cv2.getPerspectiveTransform(src, dst)
        cv2.warpPerspective(poster, m, (W, H), frame, borderMode=cv2.BORDER_TRANSPARENT)
        cv2.imwrite(str(frames_dir / f"f_{i:04d}.png"), frame)
    video = tmp_path / "persp.mp4"
    media.assemble_video(sorted(frames_dir.glob("*.png")), video, fps=FPS)

    result, _ = ops.split_video(str(video), str(poster_path), detection_method="features")
    assert result["target"]["frames_with_target"] == 12, result["target"]
    det = ops.list_frames(result["project_id"], [0])["frames"][0]
    assert np.allclose(det["quad"][0], [150, 60], atol=6), det["quad"]
