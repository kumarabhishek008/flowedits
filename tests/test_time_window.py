"""Edits limited to a time period: frames outside the window must never change."""

from __future__ import annotations

import pytest

from test_pipeline import FPS, N_FRAMES, env  # noqa: F401  (fixture)


def test_parse_time():
    from flowedits.workspace import ProjectError, parse_time

    assert parse_time(None) is None
    assert parse_time(12) == 12.0
    assert parse_time("12.5") == 12.5
    assert parse_time("12.5s") == 12.5
    assert parse_time("1:30") == 90.0
    assert parse_time("00:01:30.5") == 90.5
    assert parse_time("1:02:03") == 3723.0
    with pytest.raises(ProjectError):
        parse_time("soon")
    with pytest.raises(ProjectError):
        parse_time(-1)


def test_edit_window_limits_detection_and_edits(env):  # noqa: F811
    ops, video, logo_path, rep_path, tmp_path = env
    from flowedits.workspace import ProjectError, load_project

    # 48 frames at 24 fps = 2 s. Window 0.5 s - 1.25 s covers frames 12..29.
    result, _ = ops.split_video(str(video), str(logo_path), edit_start_time="0.5s", edit_end_time="0:01.25")
    pid = result["project_id"]
    assert result["frame_count"] == N_FRAMES
    assert result["edit_window"]["frames"] == "12-29"
    assert result["target"]["frame_ranges_with_target"] == "12-29"

    applied, _ = ops.apply_to_target(pid, "replace", str(rep_path))
    assert applied["frame_ranges"] == "12-29"
    project = load_project(pid)
    assert project.edited_indices() == list(range(12, 30))

    # A per-call range narrows further but can never escape the window.
    ops.reset_frames(pid)
    applied, _ = ops.apply_to_target(pid, "blur", start_time=0, end_time=0.75)
    assert applied["frame_ranges"] == "12-17"
    with pytest.raises(ProjectError, match="edit window"):
        ops.apply_to_target(pid, "blur", start_time=1.5)
    with pytest.raises(ProjectError, match="outside the edit window"):
        ops.update_frame(pid, 5, str(rep_path))

    # Region round trip only covers window frames.
    exported = ops.export_target_regions(pid)
    assert exported["frame_ranges"] == "12-29"

    # The final video is still full length, with untouched frames outside the window.
    out = ops.assemble(pid, str(tmp_path / "windowed.mp4"))
    assert abs(out["duration"] - N_FRAMES / FPS) < 0.15
    project = load_project(pid)
    assert project.edited_indices() == list(range(12, 18))
    for i in (0, 11, 30, 47):
        assert not project.is_edited(i)


def test_set_edit_window_widen_and_clear(env):  # noqa: F811
    ops, video, logo_path, _, _ = env

    result, _ = ops.split_video(str(video), str(logo_path), edit_end_time=1.0)
    pid = result["project_id"]
    assert result["target"]["frame_ranges_with_target"] == "0-23"
    ops.apply_to_target(pid, "fill")

    narrowed = ops.set_edit_window(pid, 0.5, 1.0)
    assert narrowed["edit_window"]["frames"] == "12-23"
    assert narrowed["edited_frames_outside_window"] == "0-11"

    widened = ops.set_edit_window(pid, None, None)
    assert widened["edit_window"].startswith("cleared")
    assert "24-47" in widened["warning"]  # target never searched there

    relocated, _ = ops.locate_target(pid, str(logo_path))
    assert relocated["frame_ranges_with_target"] == "0-47"

    frames = ops.list_frames(pid, start_time="1s", end_time="1.5s")
    assert [f["frame_index"] for f in frames["frames"]] == list(range(24, 36))

    with pytest.raises(Exception, match="contains no frames"):
        ops.set_edit_window(pid, 5, 6)


def test_window_uses_source_timeline_when_trimmed(env):  # noqa: F811
    ops, video, logo_path, _, _ = env

    # Extract 0.5 s - 2 s only: frame 0 is at 0.5 s. A window of 1.0 s - 1.5 s is frames 12..23.
    result, _ = ops.split_video(str(video), str(logo_path), trim_start_time=0.5, edit_start_time=1.0,
                                edit_end_time=1.5)
    assert result["time_span"] == [0.5, 2.0]
    assert result["edit_window"]["frames"] == "12-23"
    info = ops.list_frames(result["project_id"], [12])["frames"][0]
    assert info["time"] == 1.0 and info["in_edit_window"] is True
