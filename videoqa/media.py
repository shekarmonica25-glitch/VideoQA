"""Opening the video: reading its label (probe), pulling out the sound, and asking
FFmpeg whether any part of the picture was broken while decoding."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import app_dir, no_window_flags

SR = 16000  # all audio analysis runs at 16 kHz mono


def ffmpeg_exe() -> str:
    bundled = app_dir() / "ffmpeg" / "ffmpeg.exe"
    if bundled.exists():
        return str(bundled)
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError("FFmpeg was not found. Put ffmpeg.exe in the 'ffmpeg' folder next to VideoQA.exe.")


@dataclass
class MediaInfo:
    path: str
    duration: float
    has_video: bool
    has_audio: bool
    fps: float = 25.0
    width: int = 0
    height: int = 0
    video_start: float = 0.0
    video_duration: float = 0.0
    audio_start: float = 0.0
    audio_duration: float = 0.0  # filled in after decoding the audio
    audio_codec: str = ""
    video_codec: str = ""


def probe(path: str) -> MediaInfo:
    import av

    with av.open(path) as c:
        duration = (c.duration or 0) / 1_000_000
        info = MediaInfo(path=path, duration=duration, has_video=False, has_audio=False)
        if c.streams.video:
            vs = c.streams.video[0]
            info.has_video = True
            rate = vs.average_rate or vs.guessed_rate or 25
            info.fps = float(rate) if float(rate) > 0 else 25.0
            info.width = vs.codec_context.width
            info.height = vs.codec_context.height
            info.video_codec = vs.codec_context.name
            if vs.start_time is not None:
                info.video_start = float(vs.start_time * vs.time_base)
            if vs.duration is not None:
                info.video_duration = float(vs.duration * vs.time_base)
        if c.streams.audio:
            st = c.streams.audio[0]
            info.has_audio = True
            info.audio_codec = st.codec_context.name
            if st.start_time is not None:
                info.audio_start = float(st.start_time * st.time_base)
        if not info.video_duration:
            info.video_duration = duration
    return info


def load_audio(info: MediaInfo) -> np.ndarray | None:
    """Decode the first audio track to 16 kHz mono float32, aligned to the video timeline."""
    if not info.has_audio:
        return None
    cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-i", info.path, "-map", "0:a:0",
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True, creationflags=no_window_flags())
    audio = np.frombuffer(p.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        return None
    offset = info.audio_start - info.video_start
    if offset > 0:
        audio = np.concatenate([np.zeros(int(round(offset * SR)), np.float32), audio])
    elif offset < 0:
        audio = audio[int(round(-offset * SR)):]
    info.audio_duration = audio.size / SR
    return audio


_ERR_RE = re.compile(
    r"error while decoding|concealing|corrupt|invalid nal|missing picture|"
    r"cabac decode|left block unavailable|top block unavailable|out of range|"
    r"decode_slice_header error|no frame!|mmco: unref|non-existing pps|"
    r"illegal|damaged|broken",
    re.IGNORECASE,
)
_PTS_RE = re.compile(r"pts_time:\s*([-\d.]+)")


def decode_errors(path: str) -> list[tuple[float, str]]:
    """Decode the whole video with FFmpeg and note every decoding complaint and when it happened."""
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "info", "-threads", "1",
           "-i", path, "-map", "0:v:0", "-vf", "showinfo", "-f", "null", "-"]
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         creationflags=no_window_flags(), text=True, errors="replace")
    last_t = 0.0
    found: list[tuple[float, str]] = []
    assert p.stderr is not None
    for line in p.stderr:
        m = _PTS_RE.search(line)
        if m:
            try:
                last_t = float(m.group(1))
            except ValueError:
                pass
            continue
        if _ERR_RE.search(line) and "Parsed_showinfo" not in line:
            found.append((last_t, line.strip()[:160]))
    p.wait()
    return found


def safe_name(path: str) -> str:
    return re.sub(r"[^\w\-. ]", "_", Path(path).stem)
