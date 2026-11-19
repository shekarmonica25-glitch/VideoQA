"""Shared helpers: the Defect record, time formatting, run merging, file locations."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

# Defect type names (these appear in the log file)
MISSING_IMAGE = "MISSING_IMAGE"
FROZEN_VIDEO = "FROZEN_VIDEO"
VIDEO_BLUR = "VIDEO_BLUR"
FRAME_FRAGMENT = "FRAME_FRAGMENT"
AUDIO_DELAY = "AUDIO_DELAY"
AUDIO_OVERLAP = "AUDIO_OVERLAP"
DOUBLE_BGM = "DOUBLE_BGM"
AUDIO_LAG = "AUDIO_LAG"
NO_AUDIO = "NO_AUDIO"
GENDER_MISMATCH = "GENDER_MISMATCH"
LANGUAGE_CHANGE = "LANGUAGE_CHANGE"
INFO = "INFO"

ProgressFn = Callable[[str], None]


@dataclass
class Defect:
    kind: str
    start: float
    end: float | None
    message: str
    confidence: float
    review: bool = False  # True = "please check" (a guess, not a sure thing)

    def time_label(self) -> str:
        if self.end is None or self.end - self.start < 0.001:
            return f"[{fmt_ts(self.start)}]"
        return f"[{fmt_ts(self.start)} - {fmt_ts(self.end)}]"


def fmt_ts(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx_exclusive) for each run of True values."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def merge_intervals(iv: Iterable[tuple[float, float]], max_gap: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(iv):
        if out and s - out[-1][1] <= max_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def app_dir() -> Path:
    """Folder that holds the exe (frozen) or the project root (dev)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    return app_dir() / "models"


def no_window_flags() -> int:
    """Stop black console windows flashing when the GUI exe runs ffmpeg."""
    if os.name == "nt":
        return subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    return 0


def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))
