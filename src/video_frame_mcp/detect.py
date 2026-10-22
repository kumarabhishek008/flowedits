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

from typing import Callable

import cv2
import numpy as np

TEMPLATE_WORK_WIDTH = 640    # frames are downscaled to this width for the coarse template sweep
FEATURE_WORK_WIDTH = 1280    # and to this width for keypoint detection
TARGET_MAX_DIM = 512         # the target image is downscaled to at most this size
SCALE_STEPS = 32


def _resize(img: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return img
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=interp)


def quad_to_bbox(quad: np.ndarray) -> list[int]:
    x, y, w, h = cv2.boundingRect(np.asarray(quad, dtype=np.float32))
    return [int(x), int(y), int(w), int(h)]


class TargetDetector:
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

        self._prev_width: float | None = None  # last matched target width in work pixels
        self._sift = None
        self._kp_t = None
        self._des_t = None
        self._kp_scale = 1.0
        self._matcher = None

    # ---- public ---------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        best: dict | None = None

        if self.method in ("auto", "template"):
            best = self._template(gray)
        if self.method == "features" or (
            self.method == "auto" and (best is None or best["score"] < self.threshold)
        ):
            feat = self._features(gray)
            if feat and (best is None or feat["score"] > best["score"]):
                best = feat

        if best is None:
            self._prev_width = None
            return {"found": False, "score": 0.0}

        found = best["score"] >= self.threshold
        quad = best["quad"]
        if found and best["method"] == "template":
            self._prev_width = best["work_width"]
        elif not found:
            self._prev_width = None
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
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        return float(res.max())

    def verify(self, gray: np.ndarray, quad: np.ndarray) -> float:
        """Correlation between the target and the frame region under `quad` (rectified)."""
        m = cv2.getPerspectiveTransform(np.float32(quad), self.corners)
        warped = cv2.warpPerspective(gray, m, (self.w, self.h), flags=cv2.INTER_LINEAR)
        return self._ncc(warped, self.gray, self.mask)

    # ---- template matching ---------------------------------------------

    def _match_scales(self, img: np.ndarray, widths) -> tuple[float, tuple[int, int], int, int] | None:
        ih, iw = img.shape
        best = None
        for tw in widths:
            s = tw / self.w
            tw_i, th_i = int(round(self.w * s)), int(round(self.h * s))
            if tw_i < 8 or th_i < 8 or tw_i > iw or th_i > ih:
                continue
            tmpl = cv2.resize(self.gray, (tw_i, th_i), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            mask = None
            if self.mask is not None:
                mask = cv2.resize(self.mask, (tw_i, th_i), interpolation=cv2.INTER_NEAREST)
            res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
            res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
            _, score, _, loc = cv2.minMaxLoc(res)
            if best is None or score > best[0]:
                best = (score, loc, tw_i, th_i)
        return best

    def _template(self, gray: np.ndarray) -> dict | None:
        fh, fw = gray.shape
        k = min(1.0, TEMPLATE_WORK_WIDTH / fw)
        small = _resize(gray, k)
        sh, sw = small.shape
        aspect = self.w / self.h

        best = None
        if self._prev_width:
            # Tracking: the target rarely changes size much between consecutive frames.
            best = self._match_scales(small, np.geomspace(self._prev_width * 0.85, self._prev_width * 1.18, 7))
        if best is None or best[0] < self.threshold:
            min_w = max(12.0, 0.02 * sw)
            max_w = min(float(sw), sh * aspect)
            if max_w < min_w:
                return None
            sweep = self._match_scales(small, np.geomspace(min_w, max_w, SCALE_STEPS))
            if sweep and (best is None or sweep[0] > best[0]):
                best = sweep
        if best is None:
            return None

        _, (x, y), tw, th = best
        # Refine at full resolution in a small window around the coarse hit.
        fx, fy, ftw = x / k, y / k, tw / k
        fth = ftw / aspect
        margin = int(max(4, 2 / k + 0.06 * ftw))
        x0, y0 = max(0, int(fx) - margin), max(0, int(fy) - margin)
        x1, y1 = min(fw, int(fx + ftw) + margin + 1), min(fh, int(fy + fth) + margin + 1)
        window = gray[y0:y1, x0:x1]
        refined = self._match_scales(window, np.linspace(ftw * 0.97, ftw * 1.03, 5))
        if refined is not None:
            _, (rx, ry), rtw, rth = refined
            fx, fy, ftw, fth = x0 + rx, y0 + ry, rtw, rth
        quad = np.float32([[fx, fy], [fx + ftw, fy], [fx + ftw, fy + fth], [fx, fy + fth]])
        return {"method": "template", "quad": quad, "score": self.verify(gray, quad), "work_width": tw}

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

    def _features(self, gray: np.ndarray) -> dict | None:
        self._init_features()
        if self._des_t is None or len(self._kp_t) < 6:
            return None
        k = min(1.0, FEATURE_WORK_WIDTH / gray.shape[1])
        small = _resize(gray, k)
        kp_f, des_f = self._sift.detectAndCompute(small, None)
        if des_f is None or len(kp_f) < 6:
            return None
        pairs = self._matcher.knnMatch(self._des_t, des_f, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 8:
            return None
        src = np.float32([self._kp_t[m.queryIdx].pt for m in good]) / self._kp_scale
        dst = np.float32([kp_f[m.trainIdx].pt for m in good]) / k
        h_mat, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if h_mat is None or inliers is None or int(inliers.sum()) < 8:
            return None
        quad = cv2.perspectiveTransform(self.corners.reshape(-1, 1, 2), h_mat).reshape(4, 2)
        if not cv2.isContourConvex(quad.astype(np.float32)) or cv2.contourArea(quad) < 64:
            return None
        return {"method": "features", "quad": quad, "score": self.verify(gray, quad)}


def detect_in_frames(
    detector: TargetDetector,
    read_frame: Callable[[int], np.ndarray],
    indices: list[int],
) -> dict[int, dict]:
    results: dict[int, dict] = {}
    prev = None
    for i in indices:
        if prev is not None and i != prev + 1:
            detector._prev_width = None  # not consecutive; drop tracking state
        results[i] = detector.detect(read_frame(i))
        prev = i
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
