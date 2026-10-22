"""Find a target image (logo, product, poster, face crop...) inside every video frame.

Two strategies, combined in "auto" mode:

* template  - multi-scale normalized cross-correlation. Fast and precise when the target
              appears axis-aligned (overlays, logos, screen recordings).
* features  - SIFT keypoints + RANSAC homography. Handles rotation, perspective and
              partial occlusion (a sign filmed by a moving camera, a product in hand).

Both produce a quad (4 corners TL, TR, BR, BL in frame pixels). Every candidate is scored
the same way: warp the quad back to the target's shape and correlate with the target,
so `score` means the same thing regardless of the method that found it.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

import cv2
import numpy as np

TEMPLATE_WORK_WIDTH = 640    # frames are downscaled to this width for the full template sweep
FEATURE_WORK_WIDTH = 1280    # and to this width for keypoint detection
TARGET_MAX_DIM = 512         # the target image is downscaled to at most this size
SCALE_STEPS = 20             # full sweep: ~20% apart; the first refinement covers the gaps
TEMPLATE_CACHE = 256


def _resize(img: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return img
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=interp)


def quad_to_bbox(quad: np.ndarray) -> list[int]:
    x, y, w, h = cv2.boundingRect(np.asarray(quad, dtype=np.float32))
    return [int(x), int(y), int(w), int(h)]


class TargetDetector:
    """Finds one target image per frame. Keeps tracking state between consecutive frames:
    while the target is tracked only a region around its last position is searched."""

    def __init__(self, target_bgra: np.ndarray, method: str = "auto", threshold: float = 0.7):
        if method not in ("auto", "template", "features"):
            raise ValueError("method must be auto, template or features")
        self.method = method
        self.threshold = threshold

        factor = min(1.0, TARGET_MAX_DIM / max(target_bgra.shape[:2]))
        target_bgra = _resize(target_bgra, factor)
        self.gray = cv2.cvtColor(target_bgra[:, :, :3], cv2.COLOR_BGR2GRAY)
        alpha = target_bgra[:, :, 3]
        # Only use a mask when the target is meaningfully transparent (e.g. a cut-out logo).
        self.mask = ((alpha > 127).astype(np.uint8) * 255) if (alpha < 128).mean() > 0.05 else None
        self.h, self.w = self.gray.shape
        self.corners = np.float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]])

        self._box: tuple[float, float, float, float] | None = None  # last match (x, y, w, h) in frame pixels
        self._last_method: str | None = None
        self._templates: dict[tuple[int, int], tuple[np.ndarray, np.ndarray | None]] = {}
        self._sift = None
        self._kp_t = None
        self._des_t = None
        self._kp_scale = 1.0
        self._matcher = None

    # ---- tracking state --------------------------------------------------

    @property
    def tracking(self) -> bool:
        return self._box is not None

    def reset(self) -> None:
        self._box = None
        self._last_method = None

    def get_state(self):
        return self._box, self._last_method

    def set_state(self, state) -> None:
        self._box, self._last_method = state

    # ---- public ---------------------------------------------------------

    def detect(self, frame: np.ndarray) -> dict:
        """`frame` is BGR or already grayscale (video detection streams gray frames)."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        use_template = self.method in ("auto", "template")
        use_features = self.method in ("auto", "features")
        best: dict | None = None

        def consider(candidate: dict | None) -> bool:
            nonlocal best
            if candidate and (best is None or candidate["score"] > best["score"]):
                best = candidate
            return best is not None and best["score"] >= self.threshold

        done = False
        if self._box is not None:
            # Cheap first: search near the last position, with the method that found it last.
            order = ["features", "template"] if self._last_method == "features" else ["template", "features"]
            for m in order:
                if m == "template" and use_template:
                    done = consider(self._track_template(gray))
                elif m == "features" and use_features:
                    done = consider(self._features(gray, roi=self._roi(gray, 1.0)))
                if done:
                    break
        if not done and use_template:
            done = consider(self._template(gray))
        if not done and use_features:
            consider(self._features(gray))

        if best is None:
            self.reset()
            return {"found": False, "score": 0.0}
        found = best["score"] >= self.threshold
        quad = best["quad"]
        if found:
            self._box = tuple(float(v) for v in cv2.boundingRect(np.float32(quad)))
            self._last_method = best["method"]
        else:
            self.reset()
        return {
            "found": bool(found),
            "score": round(float(best["score"]), 4),
            "method": best["method"],
            "quad": np.round(quad, 2).tolist(),
            "bbox": quad_to_bbox(quad),
        }

    # ---- scoring --------------------------------------------------------

    def _ncc(self, a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float:
        res = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED, mask=mask)
        if mask is not None:
            res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        return float(res.max())

    def verify(self, gray: np.ndarray, quad: np.ndarray) -> float:
        """Correlation between the target and the frame region under `quad` (rectified)."""
        m = cv2.getPerspectiveTransform(np.float32(quad), self.corners)
        warped = cv2.warpPerspective(gray, m, (self.w, self.h), flags=cv2.INTER_LINEAR)
        return self._ncc(warped, self.gray, self.mask)

    # ---- template matching ---------------------------------------------

    def _template_at(self, tw: int, th: int) -> tuple[np.ndarray, np.ndarray | None]:
        key = (tw, th)
        cached = self._templates.get(key)
        if cached is None:
            if len(self._templates) >= TEMPLATE_CACHE:
                self._templates.clear()
            s = tw / self.w
            tmpl = cv2.resize(self.gray, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            mask = None
            if self.mask is not None:
                mask = cv2.resize(self.mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            cached = self._templates[key] = (tmpl, mask)
        return cached

    def _match_scales(self, img: np.ndarray, widths) -> tuple[float, tuple[int, int], int, int] | None:
        ih, iw = img.shape
        best = None
        for tw in widths:
            tw_i = int(round(tw))
            th_i = int(round(tw * self.h / self.w))
            if tw_i < 8 or th_i < 8 or tw_i > iw or th_i > ih:
                continue
            tmpl, mask = self._template_at(tw_i, th_i)
            res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
            if mask is not None:
                res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
            _, score, _, loc = cv2.minMaxLoc(res)
            if best is None or score > best[0]:
                best = (score, loc, tw_i, th_i)
        return best

    def _refine(self, gray: np.ndarray, fx: float, fy: float, ftw: float, stages, slack: float) -> dict:
        """Re-match at full resolution around (fx, fy) with progressively finer scale steps."""
        fh, fw = gray.shape
        fth = ftw * self.h / self.w
        for spread, steps in stages:
            margin = int(max(4, slack + (spread + 0.04) * ftw))
            x0, y0 = max(0, int(fx) - margin), max(0, int(fy) - margin)
            x1, y1 = min(fw, int(fx + ftw) + margin + 1), min(fh, int(fy + fth) + margin + 1)
            refined = self._match_scales(gray[y0:y1, x0:x1], np.linspace(ftw * (1 - spread), ftw * (1 + spread), steps))
            if refined is not None:
                _, (rx, ry), rtw, rth = refined
                fx, fy, ftw, fth = x0 + rx, y0 + ry, rtw, rth
        quad = np.float32([[fx, fy], [fx + ftw, fy], [fx + ftw, fy + fth], [fx, fy + fth]])
        return {"method": "template", "quad": quad, "score": self.verify(gray, quad)}

    def _roi(self, gray: np.ndarray, grow: float) -> tuple[int, int, int, int]:
        x, y, w, h = self._box
        m = max(grow * max(w, h), 24)
        fh, fw = gray.shape
        return max(0, int(x - m)), max(0, int(y - m)), min(fw, int(x + w + m)), min(fh, int(y + h + m))

    def _track_template(self, gray: np.ndarray) -> dict | None:
        """Search a region around the last match at nearby sizes (the target moves a little per frame)."""
        x0, y0, x1, y1 = self._roi(gray, 0.6)
        w = self._box[2]
        best = self._match_scales(gray[y0:y1, x0:x1], np.geomspace(w * 0.9, w * 1.11, 5))
        if best is None:
            return None
        _, (rx, ry), tw, _ = best
        return self._refine(gray, x0 + rx, y0 + ry, tw, ((0.025, 5),), slack=2)

    def _template(self, gray: np.ndarray) -> dict | None:
        """Full search: all positions and sizes on a downscaled frame, then refine at full resolution."""
        fw = gray.shape[1]
        k = min(1.0, TEMPLATE_WORK_WIDTH / fw)
        small = _resize(gray, k)
        sh, sw = small.shape
        min_w = max(12.0, 0.02 * sw)
        max_w = min(float(sw), sh * self.w / self.h)
        if max_w < min_w:
            return None
        best = self._match_scales(small, np.geomspace(min_w, max_w, SCALE_STEPS))
        if best is None:
            return None
        _, (x, y), tw, _ = best
        # The sweep's sizes are ~20% apart: refine +-11% in 2.75% steps, then +-1.5% in 0.5% steps.
        return self._refine(gray, x / k, y / k, tw / k, ((0.11, 9), (0.015, 7)), slack=2 / k)

    # ---- feature matching ----------------------------------------------

    def _init_features(self) -> None:
        if self._sift is not None:
            return
        self._sift = cv2.SIFT_create(nfeatures=4000)
        # Small targets yield few keypoints; upsample them for description.
        self._kp_scale = max(1.0, 300 / max(self.w, self.h))
        tmpl = _resize(self.gray, self._kp_scale)
        mask = _resize(self.mask, self._kp_scale) if self.mask is not None else None
        self._kp_t, self._des_t = self._sift.detectAndCompute(tmpl, mask)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    def _features(self, gray: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> dict | None:
        self._init_features()
        if self._des_t is None or len(self._kp_t) < 6:
            return None
        x0, y0 = (roi[0], roi[1]) if roi else (0, 0)
        region = gray[roi[1]:roi[3], roi[0]:roi[2]] if roi else gray
        if region.shape[0] < 16 or region.shape[1] < 16:
            return None
        k = min(1.0, FEATURE_WORK_WIDTH / region.shape[1])
        small = _resize(region, k)
        kp_f, des_f = self._sift.detectAndCompute(small, None)
        if des_f is None or len(kp_f) < 6:
            return None
        pairs = self._matcher.knnMatch(self._des_t, des_f, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 8:
            return None
        src = np.float32([self._kp_t[m.queryIdx].pt for m in good]) / self._kp_scale
        dst = np.float32([kp_f[m.trainIdx].pt for m in good]) / k + np.float32([x0, y0])
        h_mat, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if h_mat is None or inliers is None or int(inliers.sum()) < 8:
            return None
        quad = cv2.perspectiveTransform(self.corners.reshape(-1, 1, 2), h_mat).reshape(4, 2)
        if not cv2.isContourConvex(quad.astype(np.float32)) or cv2.contourArea(quad) < 64:
            return None
        return {"method": "features", "quad": quad, "score": self.verify(gray, quad)}


def scale_detection(det: dict, factor: float) -> dict:
    """Map a detection made on a resized frame back to full-resolution coordinates."""
    if "quad" not in det or factor == 1.0:
        return det
    quad = np.float32(det["quad"]) * factor
    return {**det, "quad": np.round(quad, 2).tolist(), "bbox": quad_to_bbox(quad)}


def detect_sequence(
    detector: TargetDetector,
    frames: Iterable[tuple[int, np.ndarray]],
    *,
    stride: int = 1,
    max_stride: int | None = None,
    independent: bool = False,
    on_frame: Callable[[], None] | None = None,
) -> dict[int, dict]:
    """Detect over consecutive frames.

    While the target is tracked every frame is checked (cheap: a region near its last position).
    While it is lost, only every `stride`-th frame gets the expensive full search, widening up to
    `max_stride` the longer it stays away. When it is found again the skipped frames are searched
    backwards from there, so the exact first frame of every appearance is still found.
    `independent` (unrelated photos) disables carrying position hints between frames.
    """
    max_stride = max(stride, max_stride or stride)
    results: dict[int, dict] = {}
    skipped: deque[tuple[int, np.ndarray]] = deque(maxlen=max(0, max_stride - 1))
    current = stride
    misses = 0          # consecutive failed full searches
    since_full = stride  # frames since the last full search; starts "due" so frame one is searched
    prev = None
    for i, frame in frames:
        if independent or (prev is not None and i != prev + 1):
            detector.reset()
            skipped.clear()
            since_full, current, misses = stride, stride, 0
        prev = i
        tracking = detector.tracking
        if tracking or since_full >= current:
            det = detector.detect(frame)
            since_full = 1  # a failed tracked search also falls back to the full search
            if det["found"]:
                misses, current = 0, stride
                if not tracking and skipped:
                    forward_state = detector.get_state()
                    for j, back in reversed(skipped):
                        earlier = detector.detect(back)
                        if not earlier["found"]:
                            break
                        results[j] = earlier
                    detector.set_state(forward_state)
            else:
                misses += 1
                # Every 3 misses in a row, search less often (the backfill keeps the result exact).
                current = min(max_stride, stride * (1 + misses // 3))
            skipped.clear()
            results[i] = det
        else:
            results[i] = {"found": False, "score": 0.0, "skipped": True}
            skipped.append((i, frame))
            since_full += 1
        if on_frame:
            on_frame()
    return results


def detect_parallel(
    make_detector: Callable[[], TargetDetector],
    open_chunk: Callable[[int, int], Iterable[tuple[int, np.ndarray]]],
    first: int,
    count: int,
    *,
    workers: int,
    stride: int,
    max_stride: int | None = None,
    on_frame: Callable[[], None] | None = None,
) -> dict[int, dict]:
    """Detect over [first, first+count) with a pool of threads working through small chunks.

    Many chunks per thread keep the load even: a stretch where the target is missing (slow full
    searches) is spread over threads instead of stalling one. OpenCV and the ffmpeg pipes release
    the GIL, so threads scale without the risks of extra processes (which could write to the MCP
    stdio channel).
    """
    workers = max(1, min(workers, count))
    size = max(120, min(600, -(-count // (workers * 4))))
    chunks = [(s, min(size, first + count - s)) for s in range(first, first + count, size)]

    def run(chunk):
        s, n = chunk
        return detect_sequence(make_detector(), open_chunk(s, n), stride=stride, max_stride=max_stride,
                               on_frame=on_frame)

    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(run, chunks):
            results.update(part)
    return results


def postprocess(detections: list[dict | None], max_gap: int = 3, smooth_window: int = 3) -> list[dict | None]:
    """Fill short gaps between detections by interpolation and smooth corner jitter."""
    n = len(detections)
    out = [dict(d) if d else None for d in detections]
    found = [i for i, d in enumerate(out) if d and d.get("found")]

    if max_gap > 0:
        for a, b in zip(found, found[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                qa, qb = np.float32(out[a]["quad"]), np.float32(out[b]["quad"])
                for j in range(a + 1, b):
                    t = (j - a) / (b - a)
                    quad = qa * (1 - t) + qb * t
                    out[j] = {
                        "found": True,
                        "score": round(min(out[a]["score"], out[b]["score"]), 4),
                        "method": "interpolated",
                        "quad": np.round(quad, 2).tolist(),
                        "bbox": quad_to_bbox(quad),
                    }

    if smooth_window > 1:
        half = smooth_window // 2
        quads = {i: np.float32(d["quad"]) for i, d in enumerate(out) if d and d.get("found")}
        for i in quads:
            # Symmetric window inside the same continuous run, so moving targets don't lag at run edges.
            r = 0
            while r < half and (i - r - 1) in quads and (i + r + 1) in quads:
                r += 1
            if r:
                quad = np.mean([quads[j] for j in range(i - r, i + r + 1)], axis=0)
                out[i]["quad"] = np.round(quad, 2).tolist()
                out[i]["bbox"] = quad_to_bbox(quad)
    return out
