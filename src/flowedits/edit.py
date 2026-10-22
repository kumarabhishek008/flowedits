"""Pixel operations applied to the target region (a quad) of a frame."""

from __future__ import annotations

import cv2
import numpy as np


def _quad(quad) -> np.ndarray:
    return np.float32(quad).reshape(4, 2)


def quad_size(quad) -> tuple[int, int]:
    """Approximate width/height of a quad (mean of opposite edges)."""
    q = _quad(quad)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return max(1, int(round(w))), max(1, int(round(h)))


def expand_quad(quad, pixels: float) -> np.ndarray:
    """Grow a quad outward from its centroid by roughly `pixels` on each side."""
    q = _quad(quad)
    if pixels <= 0:
        return q
    w, h = quad_size(q)
    center = q.mean(axis=0)
    scale = np.float32([1 + 2 * pixels / w, 1 + 2 * pixels / h])
    # Scale along the quad's own axes so rotated/perspective quads grow evenly.
    x_axis = (q[1] - q[0] + q[2] - q[3]) / 2
    y_axis = (q[3] - q[0] + q[2] - q[1]) / 2
    x_axis /= np.linalg.norm(x_axis) or 1
    y_axis /= np.linalg.norm(y_axis) or 1
    rel = q - center
    u, v = rel @ x_axis, rel @ y_axis
    return (center + np.outer(u * scale[0], x_axis) + np.outer(v * scale[1], y_axis)).astype(np.float32)


def on_region(frame: np.ndarray, quad, margin: int, fn) -> np.ndarray:
    """Run fn(roi, shifted_quad) on just the area around the quad and paste the result back.

    Targets are usually small next to the frame, so editing a cropped region instead of the whole
    1080p/4K frame is many times faster. `margin` must cover everything fn touches outside the quad
    (padding, feathering, blur radius, inpainting neighbourhood). Modifies `frame` in place.
    """
    q = _quad(quad)
    x, y, w, h = cv2.boundingRect(q)
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(fw, x + w + margin), min(fh, y + h + margin)
    if x1 <= x0 or y1 <= y0:
        return frame  # entirely outside the frame
    roi = np.ascontiguousarray(frame[y0:y1, x0:x1])
    frame[y0:y1, x0:x1] = fn(roi, q - np.float32([x0, y0]))
    return frame


def region_mask(shape: tuple[int, ...], quad, padding: int = 0) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(_quad(quad)).astype(np.int32), 255)
    if padding > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * padding + 1, 2 * padding + 1))
        mask = cv2.dilate(mask, kernel)
    return mask


def _composite(frame: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha.astype(np.float32)[..., None]
    out = frame.astype(np.float32) * (1 - a) + color.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _feather(alpha: np.ndarray, feather: int) -> np.ndarray:
    """Soften the edge inward so no dark border bleeds out of the warped image."""
    alpha = alpha.astype(np.float32) / 255.0
    if feather <= 0:
        return alpha
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather + 1, 2 * feather + 1))
    eroded = cv2.erode(alpha, kernel)
    return cv2.GaussianBlur(eroded, (0, 0), sigmaX=max(0.5, feather / 2))


def remove(frame: np.ndarray, quad, padding: int = 4, radius: int = 6) -> np.ndarray:
    """Erase the region and fill it from its surroundings (OpenCV inpainting)."""
    mask = region_mask(frame.shape, quad, padding)
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return frame.copy()
    # Inpaint only a padded window around the region; much faster on big frames.
    m = max(radius * 3, 16)
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(frame.shape[1], x + w + m), min(frame.shape[0], y + h + m)
    out = frame.copy()
    out[y0:y1, x0:x1] = cv2.inpaint(frame[y0:y1, x0:x1], mask[y0:y1, x0:x1], radius, cv2.INPAINT_TELEA)
    return out


def fill(frame: np.ndarray, quad, color: tuple[int, int, int], padding: int = 0) -> np.ndarray:
    out = frame.copy()
    out[region_mask(frame.shape, quad, padding) > 0] = color
    return out


def blur(frame: np.ndarray, quad, strength: int = 25, padding: int = 4) -> np.ndarray:
    mask = region_mask(frame.shape, quad, padding)
    sigma = max(1.0, strength / 2)
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma)
    return _composite(frame, blurred, cv2.GaussianBlur(mask, (0, 0), 2) / 255.0)


def pixelate(frame: np.ndarray, quad, block: int = 16, padding: int = 4) -> np.ndarray:
    mask = region_mask(frame.shape, quad, padding)
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return frame.copy()
    roi = frame[y:y + h, x:x + w]
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
    mosaic = frame.copy()
    mosaic[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = frame.copy()
    out[mask > 0] = mosaic[mask > 0]
    return out


def fit_to_aspect(image_bgra: np.ndarray, aspect: float, fit: str) -> np.ndarray:
    """Make `image_bgra` match width/height `aspect`.

    stretch: unchanged (the warp distorts it), cover: center crop, contain: transparent padding.
    """
    h, w = image_bgra.shape[:2]
    if fit == "stretch" or abs(w / h - aspect) < 1e-3:
        return image_bgra
    if fit == "cover":
        if w / h > aspect:
            nw = int(round(h * aspect))
            x0 = (w - nw) // 2
            return image_bgra[:, x0:x0 + nw]
        nh = int(round(w / aspect))
        y0 = (h - nh) // 2
        return image_bgra[y0:y0 + nh]
    if fit == "contain":
        if w / h > aspect:
            nh = int(round(w / aspect))
            canvas = np.zeros((nh, w, 4), np.uint8)
            y0 = (nh - h) // 2
            canvas[y0:y0 + h] = image_bgra
        else:
            nw = int(round(h * aspect))
            canvas = np.zeros((h, nw, 4), np.uint8)
            x0 = (nw - w) // 2
            canvas[:, x0:x0 + w] = image_bgra
        return canvas
    raise ValueError("fit must be stretch, cover or contain")


def warp_onto(frame: np.ndarray, quad, image_bgra: np.ndarray, feather: int = 2, opacity: float = 1.0) -> np.ndarray:
    """Perspective-warp an RGBA image onto the quad and alpha-composite it."""
    ih, iw = image_bgra.shape[:2]
    src = np.float32([[0, 0], [iw, 0], [iw, ih], [0, ih]])
    m = cv2.getPerspectiveTransform(src, _quad(quad))
    fh, fw = frame.shape[:2]
    warped = cv2.warpPerspective(
        image_bgra, m, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    alpha = _feather(warped[:, :, 3], feather) * float(np.clip(opacity, 0, 1))
    return _composite(frame, warped[:, :, :3], alpha)


def replace(
    frame: np.ndarray,
    quad,
    image_bgra: np.ndarray,
    fit: str = "cover",
    feather: int = 2,
    clear_first: bool | None = None,
    padding: int = 3,
    opacity: float = 1.0,
) -> np.ndarray:
    """Swap the target for `image_bgra`, following the target's position and perspective.

    When the replacement leaves parts of the region uncovered (transparency, `contain`
    padding, opacity < 1) the original target is inpainted first so it doesn't show through.
    """
    w, h = quad_size(quad)
    fitted = fit_to_aspect(image_bgra, w / h, fit)
    if clear_first is None:
        clear_first = opacity < 1.0 or bool((fitted[:, :, 3] < 250).any())
    base = remove(frame, quad, padding=padding) if clear_first else frame
    # Cover the target's edge pixels too, so feathering never reveals the old border.
    return warp_onto(base, expand_quad(quad, padding if clear_first else max(padding, feather)), fitted,
                     feather=feather, opacity=opacity)


def extract_region(frame: np.ndarray, quad, size: tuple[int, int] | None = None) -> np.ndarray:
    """Rectify the quad into an upright image (undoing perspective)."""
    w, h = size or quad_size(quad)
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(_quad(quad), dst)
    return cv2.warpPerspective(frame, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def annotate(frame: np.ndarray, quad, label: str | None = None) -> np.ndarray:
    out = frame.copy()
    thickness = max(2, frame.shape[1] // 400)
    cv2.polylines(out, [np.round(_quad(quad)).astype(np.int32)], True, (0, 255, 0), thickness, cv2.LINE_AA)
    if label:
        x, y = np.round(_quad(quad)[0]).astype(int)
        cv2.putText(out, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thickness / 2 + 0.3,
                    (0, 255, 0), thickness, cv2.LINE_AA)
    return out


def contact_sheet(tiles: list[tuple[np.ndarray, str]], columns: int = 4, tile_width: int = 320) -> np.ndarray:
    columns = max(1, min(columns, len(tiles)))
    resized = []
    for img, label in tiles:
        k = tile_width / img.shape[1]
        t = cv2.resize(img, (tile_width, max(1, int(img.shape[0] * k))), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (len(label) * 9 + 8, 20), (0, 0, 0), -1)
        cv2.putText(t, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        resized.append(t)
    tile_h = max(t.shape[0] for t in resized)
    rows = (len(resized) + columns - 1) // columns
    sheet = np.full((rows * tile_h + (rows - 1) * 4, columns * tile_width + (columns - 1) * 4, 3), 32, np.uint8)
    for n, t in enumerate(resized):
        r, c = divmod(n, columns)
        y, x = r * (tile_h + 4), c * (tile_width + 4)
        sheet[y:y + t.shape[0], x:x + t.shape[1]] = t
    return sheet
