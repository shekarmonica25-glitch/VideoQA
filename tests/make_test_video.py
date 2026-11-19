"""Builds tests/out/test_defects.mp4: a 50 s video with planted mistakes, so we can check
that VideoQA finds them. Prints the list of planted mistakes."""

import subprocess
import wave
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
SR = 16000
FPS = 25
W, H = 1280, 720
VID_LEN = 50.0
AUD_LEN = 46.0
rng = np.random.default_rng(7)


def wav(name: str) -> np.ndarray:
    with wave.open(str(ASSETS / f"{name}.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def music(bpm: float, seconds: float, root_hz: float, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    beat = 60 / bpm
    k = np.arange(int(0.25 * SR)) / SR
    kick = np.sin(2 * np.pi * (55 + 60 * np.exp(-k * 30)) * k) * np.exp(-k * 14)
    hat = r.normal(0, 1, int(0.05 * SR)) * np.exp(-np.arange(int(0.05 * SR)) / SR * 80)
    hat = np.diff(hat, prepend=0)
    b = 0.0
    while b < seconds:
        i = int(b * SR)
        out[i:i + kick.size] += kick[: max(0, n - i)] * 0.8
        j = int((b + beat / 2) * SR)
        if j < n:
            out[j:j + hat.size] += hat[: max(0, n - j)] * 0.25
        b += beat
    chord = sum(np.sin(2 * np.pi * root_hz * m * t) for m in (1, 1.26, 1.5)) / 3
    out += 0.15 * chord * (0.6 + 0.4 * np.sin(2 * np.pi * t / (beat * 4)))
    return out.astype(np.float32)


def place(dst: np.ndarray, src: np.ndarray, at: float, gain: float = 1.0) -> None:
    i = int(at * SR)
    m = min(src.size, dst.size - i)
    dst[i:i + m] += src[:m] * gain


def build_audio() -> np.ndarray:
    a = np.zeros(int(AUD_LEN * SR), np.float32)
    place(a, music(120, AUD_LEN, 130.8, 1), 0, 0.10)
    place(a, wav("male1"), 1.0)
    f1 = wav("female1").copy()
    i = int(3.5 * SR)
    f1[i:i + int(0.12 * SR)] = 0  # dropout at 11.5 s
    place(a, f1, 8.0)
    m2 = wav("male2")
    i, L = int(1.5 * SR), int(0.2 * SR)
    m2 = np.concatenate([m2[:i + L], m2[i:i + L], m2[i + L:]])  # stutter at ~17.7 s
    place(a, m2, 16.0)
    a[int(22.0 * SR)] += 0.9  # click at 22.0 s
    a[int(22.0 * SR) + 1] -= 0.7
    place(a, wav("female2"), 24.0)  # overlap 24.3 - ~28.5
    place(a, wav("male1"), 24.3)
    place(a, music(87, 8, 196.0, 2), 31.0, 0.12)  # double BGM 31-39
    place(a, wav("female3"), 40.0)
    return np.clip(a, -1, 1)


def base_canvas() -> np.ndarray:
    c = np.full((H + 200, W + 600, 3), 235, np.uint8)
    for _ in range(260):
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, c.shape[1])), int(rng.integers(0, c.shape[0]))
        if rng.random() < 0.5:
            cv2.rectangle(c, (x, y), (x + int(rng.integers(20, 160)), y + int(rng.integers(20, 120))), col, -1)
        else:
            cv2.circle(c, (x, y), int(rng.integers(10, 70)), col, -1)
    for k in range(40):
        cv2.putText(c, f"VideoQA test {k}", (int(rng.integers(0, c.shape[1] - 300)), int(rng.integers(30, c.shape[0]))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    return c


def build_frames():
    canvas = base_canvas()
    frozen = None
    for n in range(int(VID_LEN * FPS)):
        t = n / FPS
        x = int(10 * t) % 600
        y = int(60 + 40 * np.sin(t / 3))
        f = canvas[y:y + H, x:x + W].copy()
        cv2.putText(f, f"t={t:05.2f}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 200), 3)
        if 5.0 <= t < 6.5:
            f[:] = 0
        elif 12.0 <= t < 13.5:
            f = cv2.GaussianBlur(f, (0, 0), 7)
        elif n == int(15.0 * FPS):
            blocks = f[H // 2:].reshape(H // 16, 8, W // 8, 8, 3).swapaxes(1, 2).reshape(-1, 8, 8, 3).copy()
            rng.shuffle(blocks)
            f[H // 2:] = blocks.reshape(H // 16, W // 8, 8, 8, 3).swapaxes(1, 2).reshape(H // 2, W, 3)
        elif 17.0 <= t < 17.12:
            f[int(H * 0.6):] = (0, 135, 0)
        elif 20.0 <= t < 24.0:
            if frozen is None:
                frozen = f.copy()
            f = frozen
        elif 27.0 <= t < 28.0:
            small = cv2.resize(f, (W // 16, H // 16), interpolation=cv2.INTER_AREA)
            f = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        elif 33.0 <= t < 33.6:
            f[:] = 128
        yield f


def main():
    audio = build_audio()
    wav_path = OUT / "test_audio.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    out = OUT / "test_defects.mp4"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(wav_path), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-ar", "48000", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in build_frames():
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    print("Wrote", out)
    print("""Planted mistakes:
  05.0-06.5  MISSING_IMAGE   black screen
  11.5       AUDIO_LAG       120 ms dropout inside female speech
  12.0-13.5  VIDEO_BLUR      blurred
  15.0       FRAME_FRAGMENT  one scrambled frame
  17.0-17.1  FRAME_FRAGMENT  green patch
  ~17.7      AUDIO_LAG       stutter (200 ms repeated)
  20.0-24.0  FROZEN_VIDEO    frozen picture
  22.0       AUDIO_LAG       click
  24.3-28.5  AUDIO_OVERLAP   male + female talking together
  27.0-28.0  FRAME_FRAGMENT  blocky squares
  31.0-39.0  DOUBLE_BGM      second song (87 BPM over 120 BPM)
  33.0-33.6  MISSING_IMAGE   blank grey screen
  46.0-50.0  NO_AUDIO        sound ends 4 s before the picture""")


if __name__ == "__main__":
    main()
