"""Picture checks: missing image, frozen picture, blur, frame fragmentation.
Also collects face/mouth tracks used later by the audio-vs-picture checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .common import (FRAME_FRAGMENT, FROZEN_VIDEO, MISSING_IMAGE, VIDEO_BLUR, Defect,
                     ProgressFn, clamp01, merge_intervals, runs)
from .faces import FaceTracker, Track
from .media import MediaInfo

WORK_W = 640


@dataclass
class VideoData:
    fps: float
    times: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    p98: np.ndarray
    sharp: np.ndarray
    block: np.ndarray
    color_err: np.ndarray
    bottom_flat: np.ndarray
    d_prev: np.ndarray
    d_skip: np.ndarray  # difference between frame i-1 and i+1 (stored at i)
    changed: np.ndarray | None = None  # share of pixels that changed vs previous frame
    tracks: list[Track] = field(default_factory=list)
    faces_ok: bool = False
    gender_ok: bool = False


def _blockiness(gray: np.ndarray) -> float:
    """How strongly the picture shows an 8x8 block grid (1.0 = no grid)."""
    h, w = gray.shape
    if w < 32 or h < 32:
        return 1.0
    dx = cv2.absdiff(gray[:, 1:], gray[:, :-1])
    dy = cv2.absdiff(gray[1:, :], gray[:-1, :])
    cx = cv2.reduce(dx, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32F).ravel()
    cy = cv2.reduce(dy, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).ravel()

    def ratio(c: np.ndarray) -> float:
        b = c[7::8]
        mask = np.ones(c.size, bool)
        mask[7::8] = False
        inner = c[mask].mean()
        return float(b.mean() / (inner + 1.0))

    return (ratio(cx) + ratio(cy)) / 2


def scan_video(info: MediaInfo, progress: ProgressFn, use_faces: bool = True) -> VideoData:
    cap = cv2.VideoCapture(info.path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video.")
    fps = info.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(info.duration * fps) or 1
    tracker = FaceTracker() if use_faces else None

    cols: dict[str, list] = {k: [] for k in
                             ("t", "mean", "std", "p98", "sharp", "block", "color", "bflat", "dprev", "dskip",
                              "changed")}
    prev2 = prev1 = prev_gs = None
    idx = 0
    last_pct = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        h, w = frame.shape[:2]
        scale = WORK_W / w if w > WORK_W else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        gs = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gfull = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tiny = cv2.resize(gs, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)

        m, s = cv2.meanStdDev(gs)
        cols["t"].append(t)
        cols["mean"].append(float(m[0][0]))
        cols["std"].append(float(s[0][0]))
        cols["p98"].append(float(np.percentile(tiny, 98)))
        # sharpness = strength of the crispest edges (top 0.5% of pixels). Flat cartoons with crisp
        # outlines still score high; a truly blurred picture has no crisp edges anywhere.
        lap = cv2.convertScaleAbs(cv2.Laplacian(gs, cv2.CV_16S, ksize=3))
        hist = cv2.calcHist([lap], [0], None, [256], [0, 256]).ravel()
        cum = np.cumsum(hist)
        cols["sharp"].append(float(np.searchsorted(cum, 0.995 * cum[-1])))
        # share of pixels that visibly changed since the last frame (for 'frozen' detection)
        if prev_gs is not None:
            cols["changed"].append(float(cv2.countNonZero(cv2.threshold(cv2.absdiff(gs, prev_gs), 12, 255,
                                                                         cv2.THRESH_BINARY)[1])) / gs.size)
        else:
            cols["changed"].append(1.0)
        prev_gs = gs
        cols["block"].append(_blockiness(gfull) if idx % 2 == 0 else (cols["block"][-1] if cols["block"] else 1.0))

        c160 = cv2.resize(small, (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)
        b, g, r = c160[..., 0], c160[..., 1], c160[..., 2]
        green = (g > 90) & (r < 40) & (b < 40)
        magenta = (r > 150) & (b > 150) & (g < 50)
        cols["color"].append(float(np.mean(green | magenta)))

        hh = gs.shape[0]
        bottom = gs[int(hh * 0.78):]
        top = gs[: int(hh * 0.5)]
        cols["bflat"].append(bool(cv2.meanStdDev(bottom)[1][0][0] < 2.0 and cv2.meanStdDev(top)[1][0][0] > 12.0))

        cols["dprev"].append(float(np.mean(np.abs(tiny - prev1))) if prev1 is not None else 0.0)
        cols["dskip"].append(0.0)
        if prev2 is not None:
            cols["dskip"][-2] = float(np.mean(np.abs(tiny - prev2)))
        prev2, prev1 = prev1, tiny

        if tracker is not None and tracker.ok:
            tracker.update(idx, t, small, gs, frame, 1.0 / scale)

        idx += 1
        pct = int(100 * idx / total)
        if pct != last_pct and pct % 5 == 0:
            progress(f"  Watching the pictures... {min(pct, 100)}%")
            last_pct = pct
    cap.release()

    arr = {k: np.array(v) for k, v in cols.items()}
    return VideoData(
        fps=fps, times=arr["t"], mean=arr["mean"], std=arr["std"], p98=arr["p98"], sharp=arr["sharp"],
        block=arr["block"], color_err=arr["color"], bottom_flat=arr["bflat"].astype(bool),
        d_prev=arr["dprev"], d_skip=arr["dskip"], changed=arr["changed"],
        tracks=tracker.finished_tracks(int(fps)) if tracker is not None and tracker.ok else [],
        faces_ok=bool(tracker is not None and tracker.ok),
        gender_ok=bool(tracker is not None and tracker.gender.ok),
    )


def _span(vd: VideoData, a: int, b: int) -> tuple[float, float]:
    return float(vd.times[a]), float(vd.times[min(b, len(vd.times) - 1)]) if b > a + 1 else float(vd.times[a])


def find_video_defects(vd: VideoData, decode_errs: list[tuple[float, str]]) -> list[Defect]:
    out: list[Defect] = []
    n = len(vd.times)
    if n == 0:
        return [Defect(MISSING_IMAGE, 0, None, "No picture frames could be read from the video", 1.0)]
    fps = vd.fps
    dur = vd.times[-1] + 1 / fps

    # 1) Missing image: black or single-colour (blank) frames
    black = (vd.mean < 18) & (vd.p98 < 40)
    blank = (vd.std < 3.0) & ~black
    for mask, what in ((black, "Black screen"), (blank, "Blank single-colour screen")):
        for a, b in runs(mask):
            length = (b - a) / fps
            if length < 0.2:
                continue
            s, e = vd.times[a], vd.times[b - 1] + 1 / fps
            edge = s < 0.6 or e > dur - 0.6
            note = " (at start/end - may be an intended fade)" if edge else ""
            conf = clamp01(0.6 + 0.3 * min(length, 2) / 2) * (0.75 if edge else 1.0)
            out.append(Defect(MISSING_IMAGE, s, e, f"{what} for {length:.1f}s{note}", conf, review=edge))

    # 2) Frozen picture: nothing changes for a long time
    # frozen = practically no pixel changes at all (a person standing still still moves a little)
    if vd.changed is not None:
        still = (vd.changed < 0.0005) & ~black & ~blank
    else:
        still = (vd.d_prev < 0.25) & ~black & ~blank
    still[0] = False
    for a, b in runs(still):
        length = (b - a + 1) / fps
        if length >= 3.0:
            s, e = vd.times[max(a - 1, 0)], vd.times[b - 1] + 1 / fps
            conf = clamp01(0.5 + 0.1 * length)
            out.append(Defect(FROZEN_VIDEO, s, e,
                              f"Picture is frozen (no change) for {length:.1f}s - missing image or stuck video "
                              f"(ignore if it is an intended still slide)", conf, review=length < 6))

    # 3) Blur: sharpness falls far below this video's normal level
    valid = ~black & ~blank
    if valid.sum() > 10:
        med = float(np.median(vd.sharp[valid]))
        blurry = valid & (vd.sharp < 0.40 * med)
        for a, b in runs(blurry):
            length = (b - a) / fps
            if length < 0.5:
                continue
            drop = 1 - float(np.median(vd.sharp[a:b])) / (med + 1e-9)
            out.append(Defect(VIDEO_BLUR, vd.times[a], vd.times[b - 1] + 1 / fps,
                              f"Blurry picture: sharpness {drop * 100:.0f}% below this video's normal level",
                              clamp01(0.5 + 0.5 * drop), review=drop < 0.8))

    # 4) Frame fragmentation
    frag: list[tuple[float, float, str, float]] = []
    # 4a) one broken frame between two normal ones
    for i in range(1, n - 1):
        d1, d2, dsk = vd.d_prev[i], vd.d_prev[i + 1], vd.d_skip[i]
        lo = min(d1, d2)
        if lo > 18 and dsk < 0.35 * lo:
            frag.append((vd.times[i], vd.times[i] + 1 / fps, "Single broken/odd frame (differs from frames on both sides)",
                         clamp01(0.6 + (lo - 18) / 60)))
    # 4b) blocky squares (compression damage)
    if n > 10:
        bmed = float(np.median(vd.block))
        thr = max(1.6, bmed * 1.5)
        for a, b in runs(vd.block > thr):
            peak = float(vd.block[a:b].max())
            frag.append((vd.times[a], vd.times[b - 1] + 1 / fps, f"Blocky squares in picture (block score {peak:.1f} vs normal {bmed:.1f})",
                         clamp01(0.5 + (peak - thr) / 2)))
    # 4c) green / pink corruption patches
    cmed = float(np.median(vd.color_err))
    for a, b in runs((vd.color_err > 0.04) & (vd.color_err > cmed * 4 + 0.01)):
        if (b - a) / fps > 3:
            continue  # long-lasting green is probably real content (e.g. green screen)
        frag.append((vd.times[a], vd.times[b - 1] + 1 / fps,
                     f"Green/pink corrupted patch covering {vd.color_err[a:b].max() * 100:.0f}% of picture", 0.8))
    # 4d) bottom part of the picture suddenly missing (half-drawn frame)
    if vd.bottom_flat.mean() < 0.2:
        for a, b in runs(vd.bottom_flat):
            if (b - a) / fps <= 1.0:
                frag.append((vd.times[a], vd.times[b - 1] + 1 / fps, "Bottom part of the picture is missing/flat (half-drawn frame)", 0.7))
    # 4e) decoder complaints
    for t, msg in decode_errs:
        frag.append((t, t, f"Decoder error: {msg}", 0.9))

    frag.sort()
    merged: list[list] = []
    for s, e, msg, c in frag:
        if merged and s - merged[-1][1] <= 0.5:
            m = merged[-1]
            m[1] = max(m[1], e)
            m[3] = max(m[3], c)
            if msg not in m[2] and len(m[2]) < 3:
                m[2].append(msg)
            m[4] += 1
        else:
            merged.append([s, e, [msg], c, 1])
    for s, e, msgs, c, cnt in merged:
        extra = f" (+{cnt - len(msgs)} more)" if cnt > len(msgs) else ""
        out.append(Defect(FRAME_FRAGMENT, s, e if e - s > 0.05 else None, "; ".join(msgs) + extra, c))
    return out
