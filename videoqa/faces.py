"""Finding faces and mouths (YuNet), and guessing how a face looks (male/female)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .common import models_dir

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
GENDER_FILE = "genderage.onnx"


class FaceDetector:
    def __init__(self) -> None:
        path = models_dir() / YUNET_FILE
        self.ok = path.exists()
        self._size = (0, 0)
        if self.ok:
            self.det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.75, 0.3, 20)

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """Return N x 15 array: x, y, w, h, 5 landmarks (x, y), score."""
        if not self.ok:
            return np.zeros((0, 15), np.float32)
        h, w = bgr.shape[:2]
        if (w, h) != self._size:
            self.det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.det.detect(bgr)
        return faces if faces is not None else np.zeros((0, 15), np.float32)


class GenderModel:
    """Small face-attribute model. Returns the probability that a face looks male."""

    def __init__(self) -> None:
        path = models_dir() / GENDER_FILE
        self.ok = path.exists()
        if self.ok:
            import onnxruntime as ort

            self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
            shape = self.sess.get_inputs()[0].shape
            self.size = int(shape[2]) if isinstance(shape[2], int) else 96

    def p_male(self, bgr_full: np.ndarray, box: np.ndarray) -> float | None:
        if not self.ok:
            return None
        x, y, w, h = [float(v) for v in box]
        cx, cy = x + w / 2, y + h / 2
        s = self.size / (max(w, h) * 1.5)
        m = np.array([[s, 0, self.size / 2 - cx * s], [0, s, self.size / 2 - cy * s]], np.float32)
        crop = cv2.warpAffine(bgr_full, m, (self.size, self.size), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(crop, 1.0, (self.size, self.size), (0.0, 0.0, 0.0), swapRB=True)
        pred = self.sess.run(None, {self.inp: blob})[0][0]
        g = pred[:2].astype(np.float64)
        e = np.exp(g - g.max())
        return float(e[1] / e.sum())


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _roi(gray: np.ndarray, cx: float, cy: float, half_w: float, up: float, down: float) -> np.ndarray | None:
    h, w = gray.shape
    x0, x1 = int(cx - half_w), int(cx + half_w)
    y0, y1 = int(cy - up), int(cy + down)
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 - x0 < 6 or y1 - y0 < 6:
        return None
    return cv2.resize(gray[y0:y1, x0:x1], (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)


@dataclass
class Track:
    tid: int
    first_idx: int
    last_idx: int
    box: np.ndarray
    idxs: list = field(default_factory=list)
    mouth: list = field(default_factory=list)  # mouth movement (head movement removed)
    sizes: list = field(default_factory=list)
    genders: list = field(default_factory=list)  # (time, p_male)
    _mouth_roi: np.ndarray | None = None
    _eye_roi: np.ndarray | None = None
    _last_gender_t: float = -99.0


class FaceTracker:
    """Follows each face from frame to frame and records how much its mouth moves."""

    def __init__(self) -> None:
        self.det = FaceDetector()
        self.gender = GenderModel()
        self.tracks: list[Track] = []
        self._next = 0

    @property
    def ok(self) -> bool:
        return self.det.ok

    DETECT_EVERY = 3
    DETECT_W = 320

    def update(self, idx: int, t: float, small_bgr: np.ndarray, small_gray: np.ndarray,
               full_bgr: np.ndarray, scale_to_full: float) -> None:
        # finding faces is the slow part: do it on a smaller picture every few frames,
        # and re-use the last face positions in between (faces barely move in 0.1 s)
        if idx % self.DETECT_EVERY == 0 or not hasattr(self, "_last_faces"):
            h, w = small_bgr.shape[:2]
            k = self.DETECT_W / w if w > self.DETECT_W else 1.0
            img = cv2.resize(small_bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else small_bgr
            faces = self.det.detect(img).copy()
            if faces.size:
                faces[:, :14] /= k
            self._last_faces = faces
        faces = self._last_faces
        active = [tr for tr in self.tracks if tr.last_idx >= idx - 5]
        used: set[int] = set()
        for f in faces:
            box = f[:4]
            best, best_iou = None, 0.3
            for tr in active:
                if tr.tid in used:
                    continue
                v = _iou(box, tr.box)
                if v > best_iou:
                    best, best_iou = tr, v
            if best is None:
                best = Track(self._next, idx, idx, box)
                self._next += 1
                self.tracks.append(best)
            used.add(best.tid)
            continuous = best.last_idx == idx - 1
            best.box, best.last_idx = box, idx

            lm = f[4:14].reshape(5, 2)
            mouth_l, mouth_r = lm[3], lm[4]
            mw = float(np.linalg.norm(mouth_r - mouth_l))
            mcx, mcy = (mouth_l + mouth_r) / 2
            mouth = _roi(small_gray, mcx, mcy, 0.85 * mw, 0.45 * mw, 0.75 * mw) if mw > 4 else None
            ecx, ecy = (lm[0] + lm[1]) / 2
            eye = _roi(small_gray, ecx, ecy, 0.85 * mw, 0.45 * mw, 0.75 * mw) if mw > 4 else None
            mv = np.nan
            if continuous and mouth is not None and best._mouth_roi is not None:
                m_diff = float(np.mean(np.abs(mouth - best._mouth_roi)))
                e_diff = 0.0
                if eye is not None and best._eye_roi is not None:
                    e_diff = float(np.mean(np.abs(eye - best._eye_roi)))
                mv = max(0.0, m_diff - 0.7 * e_diff)
            best._mouth_roi, best._eye_roi = mouth, eye
            best.idxs.append(idx)
            best.mouth.append(mv)
            best.sizes.append(float(box[2]))

            if self.gender.ok and t - best._last_gender_t >= 0.5 and box[2] >= 40 and f[14] >= 0.85:
                p = self.gender.p_male(full_bgr, box * scale_to_full)
                if p is not None:
                    best.genders.append((t, p))
                best._last_gender_t = t

    def finished_tracks(self, min_frames: int) -> list[Track]:
        return [t for t in self.tracks if len(t.idxs) >= min_frames]
