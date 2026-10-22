"""Pasted paths are found, and the edited video lands next to the original."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from test_pipeline import FPS, N_FRAMES, env  # noqa: F401  (fixture)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "Movies" / "clips").mkdir(parents=True)
    (home / "Downloads").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("FLOWEDITS_SEARCH_DIRS", raising=False)
    monkeypatch.chdir(tmp_path)
    return home


def test_pasted_path_forms(home):
    from flowedits.paths import resolve_media_path

    video = home / "Movies" / "My Trip.mov"
    video.write_bytes(b"x")
    expected = video.resolve()

    assert resolve_media_path(str(video)) == expected
    assert resolve_media_path(f'"{video}"') == expected
    assert resolve_media_path(f"'{video}'  ") == expected
    assert resolve_media_path(str(video).replace(" ", "\\ ")) == expected  # terminal drag and drop
    assert resolve_media_path("file://" + str(video).replace(" ", "%20")) == expected
    assert resolve_media_path("~/Movies/My Trip.mov") == expected
    assert resolve_media_path("My Trip.mov") == expected  # bare name: searched
    assert resolve_media_path("my trip") == expected  # stem, any case


def test_macos_screen_recording_name(home):
    from flowedits.paths import resolve_media_path

    # Finder names contain U+202F before "PM"; pasted text has a normal space.
    real = home / "Desktop" / "Screen Recording 2026-10-22 at 11.52.03 PM.mov"
    real.parent.mkdir()
    real.write_bytes(b"x")
    typed = str(home / "Desktop" / "Screen Recording 2026-10-22 at 11.52.03 PM.mov")
    assert resolve_media_path(typed) == real.resolve()
    assert resolve_media_path("Screen Recording 2026-10-22 at 11.52.03 PM.mov") == real.resolve()


def test_search_dirs_ambiguity_and_not_found(home, tmp_path, monkeypatch):
    from flowedits.paths import find_media, resolve_media_path
    from flowedits.workspace import ProjectError

    extra = tmp_path / "projects" / "shoot"
    extra.mkdir(parents=True)
    (extra / "intro.mp4").write_bytes(b"x")
    monkeypatch.setenv("FLOWEDITS_SEARCH_DIRS", str(tmp_path / "projects"))
    assert resolve_media_path("intro.mp4") == (extra / "intro.mp4").resolve()

    (home / "Downloads" / "intro.mp4").write_bytes(b"x")
    with pytest.raises(ProjectError, match="several video files match"):
        resolve_media_path("intro.mp4")
    assert len(find_media("intro")) == 2

    with pytest.raises(ProjectError, match="video not found"):
        resolve_media_path("nope.mp4")
    with pytest.raises(ProjectError, match="is a folder"):
        resolve_media_path(str(home / "Movies"))
    # Images are not returned when looking for a video.
    (home / "Downloads" / "intro.png").write_bytes(b"x")
    assert all(p.suffix == ".mp4" for p in find_media("intro"))


def test_split_by_bare_name_and_output_next_to_source(env, home):  # noqa: F811
    ops, video, logo_path, rep_path, tmp_path = env
    clip = home / "Movies" / "clips" / "Holiday Clip.mp4"
    shutil.copy(video, clip)
    (home / "Pictures").mkdir()
    shutil.copy(logo_path, home / "Pictures" / "old logo.png")

    result, _ = ops.split_video("holiday clip.mp4", "~/Pictures/old\\ logo.png")
    pid = result["project_id"]
    assert result["source"]["path"] == str(clip.resolve())
    assert result["target"]["frames_with_target"] == N_FRAMES

    ops.apply_to_target(pid, "replace", f'"{rep_path}"')
    first = ops.assemble(pid)
    assert first["output_path"] == str(clip.parent.resolve() / "Holiday Clip_edited.mp4")
    assert Path(first["output_path"]).exists()

    # Re-assembling overwrites its own output, but never a file someone else put there.
    again = ops.assemble(pid)
    assert again["output_path"] == first["output_path"]
    foreign_project, _ = ops.split_video(str(clip))
    third = ops.assemble(foreign_project["project_id"])
    assert third["output_path"].endswith("Holiday Clip_edited_2.mp4")


def test_replace_original_keeps_backup(env, home):  # noqa: F811
    ops, video, logo_path, rep_path, _ = env
    clip = home / "Movies" / "ad.mp4"
    shutil.copy(video, clip)
    original_bytes = clip.read_bytes()

    result, _ = ops.split_video(str(clip), str(logo_path))
    pid = result["project_id"]
    ops.apply_to_target(pid, "fill")
    out = ops.assemble(pid, replace_original=True)

    backup = home / "Movies" / "ad.original.mp4"
    assert out["output_path"] == str(clip.resolve())
    assert out["original_backup"] == str(backup.resolve())
    assert backup.read_bytes() == original_bytes
    assert clip.read_bytes() != original_bytes
    assert abs(out["duration"] - N_FRAMES / FPS) < 0.15 and out["audio"]

    # A second replace keeps the first (true) original as the backup.
    ops.apply_to_target(pid, "blur")
    ops.assemble(pid, replace_original=True)
    assert backup.read_bytes() == original_bytes
    assert not list(home.glob("Movies/*flowedits-tmp*"))
