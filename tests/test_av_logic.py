"""Checks the lip-sync and voice-gender logic with a simulated face track
(a 'mouth' that moves exactly with the voice, shifted in time)."""

import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from videoqa import audio_core, audio_checks as ac  # noqa: E402
from videoqa.faces import Track  # noqa: E402
from videoqa.media import MediaInfo  # noqa: E402
from videoqa.video_checks import VideoData  # noqa: E402

SR, FPS = 16000, 25
ASSETS = Path(__file__).resolve().parent / "assets"


def wav(n):
    with wave.open(str(ASSETS / f"{n}.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def build(delay_s: float, face_p_male: float, voice: str):
    x = np.zeros(SR * 14, np.float32)
    sp = np.concatenate([wav(voice), np.zeros(SR // 2, np.float32), wav(voice)])
    x[SR: SR + sp.size] = sp[: x.size - SR]
    ad = audio_core.build_audio_data(x)
    n = int(14 * FPS)
    times = np.arange(n) / FPS
    # mouth movement = change of speech loudness, happening delay_s BEFORE the sound
    env = ac._speech_envelope(ad, times + delay_s)
    mouth = np.abs(np.diff(env, prepend=env[0])) * 8 + np.random.default_rng(1).normal(0, 0.05, n).clip(0)
    tr = Track(0, 0, n - 1, np.array([100, 100, 120, 120.0]))
    tr.idxs = list(range(n))
    tr.mouth = list(mouth)
    tr.sizes = [120.0] * n
    tr.genders = [(t, face_p_male) for t in np.arange(0, 14, 0.5)]
    z = np.zeros(n)
    vd = VideoData(fps=FPS, times=times, mean=z + 100, std=z + 40, p98=z + 200, sharp=z + 500, block=z + 1,
                   color_err=z, bottom_flat=z.astype(bool), d_prev=z + 3, d_skip=z + 3, tracks=[tr],
                   faces_ok=True, gender_ok=True)
    info = MediaInfo(path="sim", duration=14, has_video=True, has_audio=True, fps=FPS)
    return ad, vd, info


ok = True
for delay in (0.0, 0.2, -0.16):
    ad, vd, info = build(delay, 0.1, "female1")
    d = ac.check_lipsync(ad, vd, info)
    print(f"voice delay {delay * 1000:+.0f} ms ->", [x.message for x in d] or "no delay found")
    if (abs(delay) > 0.1) != bool(d):
        ok = False

for face, voice in ((0.9, "female1"), (0.1, "female1"), (0.1, "male1"), (0.9, "male1")):
    ad, vd, info = build(0.0, face, voice)
    d = [x for x in ac.check_gender_and_silent_mouth(ad, vd) if x.kind == "GENDER_MISMATCH"]
    should = (face > 0.5) != (voice.startswith("male"))
    print(f"face p_male={face} voice={voice} ->", [x.message for x in d] or "match")
    if should != bool(d):
        ok = False
print("ALL OK" if ok else "SOME CHECKS FAILED")
