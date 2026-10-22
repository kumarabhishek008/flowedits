"""Photos use the same workflow as videos: open -> locate -> edit -> save next to the original."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from test_paths import home  # noqa: F401  (fixture)
from test_pipeline import env, make_logo  # noqa: F401  (fixture)


def _photo(w: int, h: int, logo_at: tuple[int, int] | None, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(40, 200, (h, w, 3), dtype=np.uint8), (0, 0), 6)
    if logo_at:
        logo = make_logo()
        logo = cv2.resize(logo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        x, y = logo_at
        img[y:y + logo.shape[0], x:x + logo.shape[1]] = logo
    return img


def test_single_photo_replace_and_save(env, home):  # noqa: F811
    ops, _, logo_path, rep_path, _ = env
    photo = home / "Pictures" / "Shop Front.jpg"
    photo.parent.mkdir()
    cv2.imwrite(str(photo), _photo(800, 600, (300, 200)), [cv2.IMWRITE_JPEG_QUALITY, 95])

    # split_video accepts a photo too, found by bare name.
    result, preview = ops.split_video("shop front.jpg", str(logo_path))
    assert result["media_type"] == "photos" and result["photo_count"] == 1
    assert result["target"]["frames_with_target"] == 1 and preview is not None
    pid = result["project_id"]
    x, y, w, h = ops.list_frames(pid)["frames"][0]["bbox"]
    assert abs(x - 300) <= 3 and abs(y - 200) <= 3 and abs(w - 120) <= 4

    ops.apply_to_target(pid, "replace", str(rep_path), padding=0, feather=0)
    saved = ops.assemble(pid)  # assemble_video on a photo project saves photos
    out = home / "Pictures" / "Shop Front_edited.jpg"
    assert saved["saved"][0]["saved_to"] == str(out.resolve())
    img = cv2.imread(str(out))
    assert img.shape[:2] == (600, 800)
    b, g, r = img[260, 360].astype(int)  # centre of the old logo is now magenta
    assert b > 200 and g < 50 and r > 200


def test_photo_folder_mixed_sizes(env, home):  # noqa: F811
    ops, _, logo_path, _, _ = env
    folder = home / "Pictures" / "batch"
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / "a.png"), _photo(640, 480, (50, 60), seed=1))
    cv2.imwrite(str(folder / "b.jpg"), _photo(1200, 900, (700, 500), scale=1.5, seed=2))
    cv2.imwrite(str(folder / "c.png"), _photo(500, 500, None, seed=3))  # no logo

    result, _ = ops.open_photos(str(folder), str(logo_path))
    pid = result["project_id"]
    assert result["photo_count"] == 3
    # Unrelated photos: no interpolation fills c.png in from its neighbours.
    assert result["target"]["frame_ranges_with_target"] == "0-1"
    b_box = ops.list_frames(pid, [1])["frames"][0]["bbox"]
    assert abs(b_box[0] - 700) <= 4 and abs(b_box[2] - 180) <= 6

    ops.apply_to_target(pid, "blur")
    saved = ops.save_photos(pid)
    names = sorted(s["saved_to"].rsplit("/", 1)[1] for s in saved["saved"])
    assert names == ["a_edited.png", "b_edited.jpg"]
    assert saved["unedited_photos_skipped"] == 1
    assert cv2.imread(str(folder / "b_edited.jpg")).shape[:2] == (900, 1200)

    # update_frame resizes to that photo's own size, not the first photo's.
    replacement = home / "full.png"
    cv2.imwrite(str(replacement), np.zeros((10, 10, 3), np.uint8))
    assert ops.update_frame(pid, 2, str(replacement))["resized_to_frame_size"]
    assert ops.view_frame(pid, 2)[0]["photo"].endswith("c.png")

    out_dir = home / "exports"
    saved = ops.save_photos(pid, output_path=str(out_dir))
    assert sorted(p.name for p in out_dir.iterdir()) == ["a_edited.png", "b_edited.jpg", "c_edited.png"]


def test_photo_replace_original_and_video_only_options(env, home):  # noqa: F811
    ops, _, logo_path, _, _ = env
    from flowedits.workspace import ProjectError

    photo = home / "Downloads" / "flyer.png"
    cv2.imwrite(str(photo), _photo(640, 480, (100, 100)))
    original = photo.read_bytes()

    result, _ = ops.open_photos(str(photo), str(logo_path))
    pid = result["project_id"]
    with pytest.raises(ProjectError, match="no edited photos"):
        ops.save_photos(pid)
    with pytest.raises(ProjectError, match="videos only"):
        ops.set_edit_window(pid, 0, 1)
    with pytest.raises(ProjectError, match="videos only"):
        ops.apply_to_target(pid, "fill", start_time=0)
    with pytest.raises(ProjectError, match="videos only"):
        ops.split_video(str(photo), edit_start_time=1)

    ops.apply_to_target(pid, "fill", fill_color="#00ff00")
    out = ops.save_photos(pid, replace_original=True)
    backup = home / "Downloads" / "flyer.original.png"
    assert out["original_backups"] == [str(backup)]
    assert backup.read_bytes() == original and photo.read_bytes() != original
    assert not list((home / "Downloads").glob("*flowedits-tmp*"))

    # Saving again keeps the first original as the backup.
    ops.apply_to_target(pid, "blur")
    ops.save_photos(pid, replace_original=True)
    assert backup.read_bytes() == original
