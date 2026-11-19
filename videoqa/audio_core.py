"""Basic sound measurements shared by all sound checks:
loudness over time, the spectrogram, where people are talking (VAD) and voice pitch."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .media import SR

N_FFT = 1024
HOP = 256  # 16 ms


def frame_rms_db(x: np.ndarray, hop: int = 160, win: int = 320) -> np.ndarray:
    """Loudness every 10 ms, in dB (0 dB = full scale)."""
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    n = 1 + (x.size - win) // hop
    fr = np.lib.stride_tricks.as_strided(x, shape=(n, win), strides=(x.strides[0] * hop, x.strides[0]))
    rms = np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)
    return (20 * np.log10(rms)).astype(np.float32)


def stft_mag(x: np.ndarray) -> np.ndarray:
    if x.size < N_FFT:
        x = np.pad(x, (0, N_FFT - x.size))
    n = 1 + (x.size - N_FFT) // HOP
    fr = np.lib.stride_tricks.as_strided(x, shape=(n, N_FFT), strides=(x.strides[0] * HOP, x.strides[0]))
    win = np.hanning(N_FFT).astype(np.float32)
    out = np.empty((n, N_FFT // 2 + 1), np.float32)
    for i in range(0, n, 4096):
        out[i:i + 4096] = np.abs(np.fft.rfft(fr[i:i + 4096] * win, axis=1))
    return out


def speech_segments(x: np.ndarray) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(threshold=0.5, min_speech_duration_ms=200, min_silence_duration_ms=250, speech_pad_ms=60)
    ts = get_speech_timestamps(x, opts)
    return [(d["start"] / SR, d["end"] / SR) for d in ts]


@dataclass
class PitchTrack:
    idx: np.ndarray        # spectrogram frame indices analysed
    f1: np.ndarray         # voice pitch (Hz)
    s1: np.ndarray         # how clearly voiced the frame is (0..1)


def voice_pitch(x: np.ndarray, frame_idx: np.ndarray) -> PitchTrack:
    """Voice pitch per frame by autocorrelation. Picks the SHORTEST strong repeat period,
    which avoids 'ghost' notes 2-3x too low (a common mistake when music plays underneath)."""
    lo, hi = int(SR / 400), int(SR / 70)
    win = np.hanning(N_FFT).astype(np.float32)
    wac = np.fft.irfft(np.abs(np.fft.rfft(win, 2 * N_FFT)) ** 2)[: hi + 2]
    wac = wac / wac[0]
    f1s, s1s = [], []
    offs = np.arange(N_FFT)
    for i in range(0, frame_idx.size, 2048):
        fi = frame_idx[i:i + 2048]
        idx = np.minimum(fi[:, None] * HOP + offs[None, :], x.size - 1)
        fr = x[idx]
        fr = (fr - fr.mean(axis=1, keepdims=True)) * win
        r = np.fft.irfft(np.abs(np.fft.rfft(fr, 2 * N_FFT, axis=1)) ** 2, axis=1)[:, : hi + 2]
        r = r / (r[:, :1] + 1e-12) / np.maximum(wac, 1e-3)
        seg = r[:, lo - 1: hi + 2]
        peak = (seg[:, 1:-1] >= seg[:, :-2]) & (seg[:, 1:-1] >= seg[:, 2:])
        core = seg[:, 1:-1]
        best = np.where(peak, core, -1).max(axis=1)
        good = peak & (core >= 0.88 * best[:, None])
        k = good.argmax(axis=1)
        lag = lo + k
        f1s.append(SR / lag)
        s1s.append(np.clip(core[np.arange(k.size), k], 0, 1))
    cat = lambda L: np.concatenate(L) if L else np.zeros(0)
    return PitchTrack(frame_idx, cat(f1s), cat(s1s))


@dataclass
class AudioData:
    x: np.ndarray
    duration: float
    rms_db: np.ndarray                 # every 10 ms
    spec: np.ndarray                   # every 16 ms
    spec_t: np.ndarray
    speech: list[tuple[float, float]]
    speech_mask: np.ndarray            # per spectrogram frame
    pitch: PitchTrack | None = None
    extra: dict = field(default_factory=dict)

    def rms_at(self, t0: float, t1: float) -> np.ndarray:
        return self.rms_db[int(t0 * 100):max(int(t1 * 100), int(t0 * 100) + 1)]


def build_audio_data(x: np.ndarray) -> AudioData:
    spec = stft_mag(x)
    spec_t = (np.arange(spec.shape[0]) * HOP + N_FFT / 2) / SR
    segs = speech_segments(x)
    mask = np.zeros(spec.shape[0], bool)
    for s, e in segs:
        mask[(spec_t >= s) & (spec_t <= e)] = True
    rms = frame_rms_db(x)
    ad = AudioData(x=x, duration=x.size / SR, rms_db=rms, spec=spec, spec_t=spec_t, speech=segs, speech_mask=mask)
    # pitch only where people talk and it is loud enough
    loud = np.interp(spec_t, np.arange(rms.size) * 0.01 + 0.01, rms) > -45
    fi = np.flatnonzero(mask & loud)
    if fi.size:
        ad.pitch = voice_pitch(x, fi)
    return ad
