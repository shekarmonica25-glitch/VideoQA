"""Runs every check on one video and writes the log file."""

from __future__ import annotations

import csv
import time
import traceback
from datetime import datetime
from pathlib import Path

from . import __version__
from .common import Defect, ProgressFn, fmt_ts
from .media import load_audio, probe, safe_name, decode_errors

ALL_CHECKS = {
    "missing_image": "Missing image / black / frozen",
    "blur": "Video blur",
    "fragment": "Frame fragmentation",
    "no_audio": "No audio (incl. at the end)",
    "audio_lag": "Audio lag (gaps, clicks, stutter)",
    "overlap": "Audio overlap (two voices)",
    "double_bgm": "Double BGM",
    "language": "Audio language change",
    "delay": "Audio delay (lip sync)",
    "gender": "Wrong voice gender",
}

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v", ".webm", ".flv", ".mpg", ".mpeg", ".ts", ".mts"}


def analyze(path: str, checks: set[str], progress: ProgressFn) -> tuple[list[Defect], dict]:
    t_start = time.time()
    notes: dict = {"warnings": [], "timings": []}
    outer = progress
    stage = {"name": None, "t": t_start}

    def progress(msg: str) -> None:  # also records how long each step took
        if not msg.startswith("  "):
            now = time.time()
            if stage["name"]:
                notes["timings"].append((stage["name"], now - stage["t"]))
            stage["name"], stage["t"] = msg.rstrip("."), now
        outer(msg)

    progress("Reading the video label...")
    info = probe(path)
    notes["info"] = info
    defects: list[Defect] = []

    vd = None
    if info.has_video:
        from .video_checks import find_video_defects, scan_video

        need_faces = bool(checks & {"delay", "gender"})
        progress("Watching the pictures...")
        vd = scan_video(info, progress, use_faces=need_faces)
        if need_faces and not vd.faces_ok:
            notes["warnings"].append("Face model missing - lip sync and gender checks skipped.")
        errs: list = []
        if "fragment" in checks:
            progress("Asking the decoder about broken frames...")
            errs = decode_errors(path)
        vdef = find_video_defects(vd, errs)
        keep = {"MISSING_IMAGE": "missing_image", "FROZEN_VIDEO": "missing_image",
                "VIDEO_BLUR": "blur", "FRAME_FRAGMENT": "fragment"}
        defects += [d for d in vdef if keep.get(d.kind) in checks]
        notes["faces_found"] = len(vd.tracks)
    else:
        notes["warnings"].append("No picture track found.")

    progress("Pulling out the sound...")
    audio = load_audio(info)
    from . import audio_checks as ac

    if "no_audio" in checks:
        defects += ac.check_no_audio(info, None if audio is None else _ad(audio, notes, progress), vd)
    if audio is not None:
        ad = _ad(audio, notes, progress)
        steps = [
            ("audio_lag", "Listening for gaps, clicks and stutter...", lambda: ac.check_audio_lag(ad)),
            ("overlap", "Listening for two voices at once...", lambda: ac.check_overlap(ad)),
            ("double_bgm", "Listening for two background songs...", lambda: ac.check_double_bgm(ad, progress)),
        ]
        for key, msg, fn in steps:
            if key in checks:
                progress(msg)
                defects += _safe(fn, key, notes)
        if "language" in checks:
            progress("Finding the language of each speech piece (AI)...")
            res = _safe(lambda: ac.check_language(ad, progress), "language", notes)
            if res:
                d, main = res
                defects += d
                notes["main_language"] = main
        if vd is not None and vd.tracks:
            if "delay" in checks:
                progress("Comparing lips with voice (lip sync)...")
                defects += _safe(lambda: ac.check_lipsync(ad, vd, info), "delay", notes)
            if "gender" in checks or "no_audio" in checks:
                progress("Comparing faces with voices...")
                gd = _safe(lambda: ac.check_gender_and_silent_mouth(ad, vd), "gender", notes)
                defects += [d for d in gd if (d.kind == "GENDER_MISMATCH" and "gender" in checks)
                            or (d.kind == "NO_AUDIO" and "no_audio" in checks)]
        elif vd is not None and checks & {"delay", "gender"}:
            notes["warnings"].append("No clear face found - lip sync and voice-gender checks had nothing to compare.")
        notes["speech_seconds"] = sum(e - s for s, e in ad.speech)

    defects.sort(key=lambda d: (d.start, d.kind))
    progress("Done")
    notes["timings"] = [t for t in notes["timings"] if t[1] >= 0.5]
    notes["seconds"] = time.time() - t_start
    return defects, notes


def _ad(audio, notes, progress):
    if "_ad" not in notes:
        from .audio_core import build_audio_data

        progress("Measuring the sound (voice, pitch, music)...")
        notes["_ad"] = build_audio_data(audio)
    return notes["_ad"]


def _safe(fn, key, notes):
    try:
        return fn()
    except Exception as e:  # one broken check must not stop the others
        notes["warnings"].append(f"Check '{key}' failed: {e}")
        notes.setdefault("errors", []).append(traceback.format_exc())
        return []


def write_report(path: str, defects: list[Defect], notes: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"{safe_name(path)}_QA_{stamp}"
    log, csvp = base.with_suffix(".log"), base.with_suffix(".csv")
    info = notes.get("info")
    width = max([len(d.time_label()) for d in defects] + [30])
    lines = [
        "=== Video QA Report ===",
        f"File     : {path}",
        f"Scanned  : {datetime.now():%Y-%m-%d %H:%M:%S}   (VideoQA {__version__}, took {notes.get('seconds', 0):.0f}s)",
    ]
    if info:
        lines.append(f"Duration : {fmt_ts(info.video_duration or info.duration)}   "
                     f"{info.width}x{info.height} @ {info.fps:.2f} fps, video={info.video_codec or '-'}, audio={info.audio_codec or 'none'}")
    if "main_language" in notes:
        lines.append(f"Language : {notes['main_language']} (main spoken language)")
    if "faces_found" in notes:
        lines.append(f"Faces    : {notes['faces_found']} face track(s) found")
    for w in notes.get("warnings", []):
        lines.append(f"WARNING  : {w}")
    lines.append("")
    if not defects:
        lines.append("No defects found.")
    for d in defects:
        tag = "  (please check)" if d.review else ""
        lines.append(f"{d.time_label():<{width}}  {d.kind:<16} {d.message}   (confidence {d.confidence:.2f}){tag}")
    lines.append("")
    lines.append(f"Total defects: {len(defects)}")
    by_kind: dict[str, int] = {}
    for d in defects:
        by_kind[d.kind] = by_kind.get(d.kind, 0) + 1
    for k, v in sorted(by_kind.items()):
        lines.append(f"  {k:<16} {v}")
    if notes.get("timings"):
        lines += ["", "Time spent per step:"] + [f"  {s:6.0f}s  {n}" for n, s in notes["timings"]]
    if notes.get("errors"):
        lines += ["", "--- technical error details ---"] + notes["errors"]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csvp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "start", "end", "start_seconds", "end_seconds", "defect", "details", "confidence", "please_check"])
        for d in defects:
            w.writerow([path, fmt_ts(d.start), fmt_ts(d.end) if d.end is not None else "", f"{d.start:.3f}",
                        f"{d.end:.3f}" if d.end is not None else "", d.kind, d.message, f"{d.confidence:.2f}",
                        "yes" if d.review else "no"])
    return log, csvp


def collect_videos(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out += [str(f) for f in sorted(pp.rglob("*")) if f.suffix.lower() in VIDEO_EXT]
        elif pp.is_file():
            out.append(str(pp))
    return out
