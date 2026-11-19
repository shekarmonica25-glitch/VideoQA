"""Sound checks: no audio, audio lag (gaps/clicks/stutter), overlap, double BGM, language change.
Plus the checks that compare sound with the picture: audio delay (lip sync), wrong voice gender,
and mouth moving with no voice."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

from .audio_core import HOP, AudioData
from .common import (AUDIO_DELAY, AUDIO_LAG, AUDIO_OVERLAP, DOUBLE_BGM, GENDER_MISMATCH, LANGUAGE_CHANGE,
                     NO_AUDIO, Defect, ProgressFn, clamp01, merge_intervals, models_dir, runs)
from .media import SR, MediaInfo
from .video_checks import VideoData

SILENT_DB = -55.0


# ---------------------------------------------------------------- 9. no audio
def check_no_audio(info: MediaInfo, ad: AudioData | None, vd: VideoData | None) -> list[Defect]:
    out: list[Defect] = []
    vdur = info.video_duration or info.duration
    if ad is None:
        return [Defect(NO_AUDIO, 0.0, vdur, "Video has no sound track at all", 1.0)]

    def moving(t0: float, t1: float) -> float:
        if vd is None or vd.times.size == 0:
            return 0.0
        m = (vd.times >= t0) & (vd.times <= t1)
        return float(np.mean(vd.d_prev[m])) if m.any() else 0.0

    if vdur - ad.duration > 0.5:
        out.append(Defect(NO_AUDIO, ad.duration, vdur,
                          f"Sound track ends {vdur - ad.duration:.1f}s before the picture ends", 0.97))

    loud = ad.rms_db > SILENT_DB
    if loud.any():
        last = (np.flatnonzero(loud)[-1] + 1) * 0.01
        end = min(vdur, ad.duration)
        tail = end - last
        if tail >= 2.0:
            mv = moving(last, end)
            if mv > 0.8:
                out.append(Defect(NO_AUDIO, last, end, f"Silent for the last {tail:.1f}s while the picture is still moving", 0.9))
            else:
                out.append(Defect(NO_AUDIO, last, end, f"Silent for the last {tail:.1f}s (picture is still - may be an intended end card)",
                                  0.6, review=True))
        first = np.flatnonzero(loud)[0] * 0.01
        # silent holes in the middle of the video
        for a, b in runs(~loud):
            s, e = a * 0.01, b * 0.01
            if s <= first or e >= last or e - s < 2.0:
                continue
            mv = moving(s, e)
            if mv > 0.8:
                out.append(Defect(NO_AUDIO, s, e, f"No sound for {e - s:.1f}s in the middle of the video while the picture moves", 0.8))
    else:
        out.append(Defect(NO_AUDIO, 0.0, vdur, "Sound track is completely silent", 1.0))
    return out


# ---------------------------------------------------------------- 5. audio lag
def check_audio_lag(ad: AudioData) -> list[Defect]:
    out: list[Defect] = []
    rms = ad.rms_db
    x = ad.x

    # 5a) dropouts: the voice is cut off within a few ms, then comes back (music may continue)
    from scipy.signal import butter, sosfilt

    sos = butter(4, [300, 3400], btype="band", fs=SR, output="sos")
    y = np.concatenate([sosfilt(sos, x[i:i + SR * 60]).astype(np.float32) for i in range(0, x.size, SR * 60)])
    hop, win = 32, 64  # 2 ms steps
    n = (y.size - win) // hop
    if n > 50:
        fr = np.lib.stride_tricks.as_strided(y, shape=(n, win), strides=(y.strides[0] * hop, y.strides[0]))
        env = 10 * np.log10(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)
        et = (np.arange(n) * hop + win / 2) / SR
        in_speech = np.zeros(n, bool)
        for s, e in ad.speech:
            in_speech[(et >= s) & (et <= e)] = True
        loud_ref = np.percentile(env[in_speech], 70) if in_speech.any() else env.max()
        k = 10  # 20 ms: a cut-off happens within this time (natural word endings fade slower)
        drop = np.flatnonzero(in_speech[:-k] & (env[:-k] > loud_ref - 12) & (env[:-k] - env[k:] > 20))
        last_end = -1.0
        for i in drop:
            if et[i] < last_end:
                continue
            top = env[i]
            j = i + k
            while j < n - k and env[j] < top - 20 and et[j] - et[i] < 0.8:
                j += 1
            gap = et[j] - et[i]
            # long enough not to be a normal 'p/t/k' pause, deep, and the voice snaps back
            if (0.17 <= gap < 0.8 and j < n - 12 and env[i + k:j].mean() < top - 35
                    and env[min(j + 12, n - 1)] > top - 12 and in_speech[j]):
                out.append(Defect(AUDIO_LAG, float(et[i]), float(et[j]),
                                  f"Voice cuts out suddenly for {gap * 1000:.0f} ms in the middle of speech (dropout/lag)",
                                  0.6, review=True))
                last_end = et[j] + 0.1

    # 5b) clicks / pops: one sharp spike much bigger than everything around it
    d = np.abs(np.diff(x))
    nfr = d.size // 160
    if nfr > 20:
        fr_med = np.median(d[:nfr * 160].reshape(nfr, 160), axis=1)
        ref = np.repeat(median_filter(fr_med, size=21, mode="nearest"), 160)
        cand = np.flatnonzero((d[:nfr * 160] > 6 * (ref + 1e-4)) & (d[:nfr * 160] > 0.08))
        clicks: list[list] = []
        last = -10**9
        for c in cand:
            if c - last < 12:
                continue
            lo, hi = max(0, c - 12), min(d.size, c + 12)
            c = lo + int(np.argmax(d[lo:hi]))
            last = c
            around = np.concatenate([d[max(0, c - 200):max(0, c - 10)], d[c + 10:c + 200]])
            if around.size < 100:
                continue
            if d[c] > 5 * np.percentile(around, 95) + 0.03:
                t = c / SR
                if clicks and t - clicks[-1][1] < 0.3:
                    clicks[-1][1] = t
                    clicks[-1][2] += 1
                else:
                    clicks.append([t, t, 1])
        for s, e, cnt in clicks:
            out.append(Defect(AUDIO_LAG, s, e if e - s > 0.01 else None,
                              "Click/pop noise in the sound" + (f" ({cnt} clicks)" if cnt > 1 else ""), 0.75))

    # 5c) stutter: a short piece of speech repeated straight after itself
    S = np.log1p(ad.spec[:, 4:200] * 10)
    S = S - S.mean(axis=1, keepdims=True)
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-9
    energy = np.interp(ad.spec_t, np.arange(rms.size) * 0.01, rms)
    ok_frame = ad.speech_mask & (energy > -40)
    found: list[tuple[float, float, float]] = []
    for n in range(5, 26):  # 80 ms .. 400 ms
        if S.shape[0] <= 2 * n:
            break
        sim = np.einsum("ij,ij->i", S[:-n], S[n:])
        avg = uniform_filter1d(sim, n, origin=-(n // 2))
        half = max(2, n // 2)
        sim_half = np.einsum("ij,ij->i", S[:-half], S[half:])
        avg_half = uniform_filter1d(sim_half, n, origin=-(n // 2))[: avg.size]
        okm = ok_frame[: avg.size] & ok_frame[n: n + avg.size]
        normal = np.percentile(avg[okm], 99) if okm.sum() > 100 else 0.75
        hit = okm & (avg > max(0.8, normal + 0.06)) & (avg_half[: avg.size] < 0.5)
        for a, b in runs(hit):
            found.append((float(ad.spec_t[a]), float(ad.spec_t[min(b + 2 * n, ad.spec_t.size - 1)]), n * HOP / SR))
    for s, e in merge_intervals([(f[0], f[1]) for f in found], 0.2):
        piece = [f[2] for f in found if f[0] >= s and f[0] <= e]
        out.append(Defect(AUDIO_LAG, s, e, f"Sound stutters: a {np.median(piece) * 1000:.0f} ms piece repeats itself",
                          0.7, review=True))
    return out


# ---------------------------------------------------------------- 2. overlap
SEG_FILE = "pyannote_segmentation_3.onnx"
SEG_STEP, SEG_OFFSET = 270, 990  # samples per output frame, first frame centre


def overlap_probability(x: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """pyannote segmentation model: probability that 2 people talk at once, over time."""
    path = models_dir() / SEG_FILE
    if not path.exists():
        return None
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    win, hop = 10 * SR, 5 * SR
    n_out = 1 + max(0, x.size - SEG_OFFSET) // SEG_STEP
    acc = np.zeros(n_out)
    cnt = np.zeros(n_out)
    starts = list(range(0, max(1, x.size - win + 1), hop))
    if starts[-1] + win < x.size:
        starts.append(max(0, x.size - win))
    for st in starts:
        seg = x[st:st + win]
        if seg.size < win:
            seg = np.pad(seg, (0, win - seg.size))
        logits = sess.run(None, {"input_values": seg[None, None, :].astype(np.float32)})[0][0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        ov = p[:, 4:7].sum(axis=1)
        o0 = st // SEG_STEP
        m = min(ov.size, n_out - o0)
        if m <= 0:
            continue
        # trust the middle of each window more than its edges
        w = np.hanning(ov.size + 2)[1:-1][:m] + 1e-3
        acc[o0:o0 + m] += ov[:m] * w
        cnt[o0:o0 + m] += w
    prob = acc / np.maximum(cnt, 1e-9)
    t = (np.arange(n_out) * SEG_STEP + SEG_OFFSET) / SR
    return t, prob


def check_overlap(ad: AudioData) -> list[Defect]:
    res = overlap_probability(ad.x)
    if res is None:
        return []
    t, prob = res
    sm = uniform_filter1d(prob, 9)
    out = []
    for a, b in runs(sm > 0.5):
        s, e = float(t[a]), float(t[min(b, t.size - 1)])
        if e - s < 0.4:
            continue
        pk = float(sm[a:b].mean())
        out.append(Defect(AUDIO_OVERLAP, s, e, f"Two or more voices talking at the same time ({e - s:.1f}s)",
                          clamp01(pk), review=pk < 0.75))
    merged: list[Defect] = []
    for d in out:
        if merged and d.start - (merged[-1].end or 0) < 0.8:
            merged[-1].end = d.end
            merged[-1].message = f"Two or more voices talking at the same time ({d.end - merged[-1].start:.1f}s)"
        else:
            merged.append(d)
    return merged


# ---------------------------------------------------------------- 3. double BGM
def check_double_bgm(ad: AudioData, progress: ProgressFn) -> list[Defect]:
    out: list[Defect] = []
    fps = SR / HOP
    # beats are measured in the deep bass (30-160 Hz: kick drum, bass) where most voices do not reach,
    # so the rhythm of speech is not mistaken for a second song
    logS = np.log1p(ad.spec[:, 2:11] * 10)
    flux = np.maximum(0, np.diff(logS, axis=0)).sum(axis=1)
    flux = np.concatenate([[0], flux])
    win, step = int(8 * fps), int(2 * fps)
    lag_lo, lag_hi = int(0.3 * fps), int(1.5 * fps)
    clash_win: list[tuple[float, float, float, float]] = []
    beat_period: list[float] = []
    for start in range(0, max(1, flux.size - win), step):
        seg = flux[start:start + win]
        if seg.size < win:
            break
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[seg.size - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        part = ac[lag_lo:lag_hi]
        pk, props = find_peaks(part, height=0.12, prominence=0.05)
        if pk.size == 0:
            continue
        order = np.argsort(props["peak_heights"])[::-1]
        l1 = (pk[order[0]] + lag_lo)
        h1 = props["peak_heights"][order[0]]
        beat_period.append(l1 / fps)
        if h1 < 0.2:
            continue
        for j in order[1:]:
            l2 = pk[j] + lag_lo
            h2 = props["peak_heights"][j]
            if h2 < max(0.12, 0.3 * h1):
                break
            r = l2 / l1
            related = any(abs(r - k) / k < 0.05 for k in (1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 1, 4 / 3, 3 / 2, 2, 5 / 2, 3, 4))
            if not related:
                clash_win.append((start / fps, (start + win) / fps, 60 * fps / l1, 60 * fps / l2))
                break
    # need at least 2 windows in a row to call it
    groups = merge_intervals([(a, b) for a, b, _, _ in clash_win], 0.1)
    for s, e in groups:
        members = [c for c in clash_win if c[0] >= s and c[1] <= e]
        if len(members) >= 2:
            t1 = np.median([m[2] for m in members])
            t2 = np.median([m[3] for m in members])
            t1, t2 = (t1 * 2 if t1 < 60 else t1), (t2 * 2 if t2 < 60 else t2)
            out.append(Defect(DOUBLE_BGM, s, e, f"Two different music beats at once (~{t1:.0f} BPM and ~{t2:.0f} BPM)",
                              clamp01(0.5 + 0.05 * len(members)), review=True))

    # same music layered twice with a small time shift -> 'echo' peak in the cepstrum
    n = 1 << 16  # ~4 s
    hop = n // 2
    qlo, qhi = int(0.03 * SR), int(1.5 * SR)
    beats = np.array(beat_period) if beat_period else np.zeros(0)
    hits: list[tuple[float, float, float, float]] = []
    x = ad.x
    w = np.hanning(n).astype(np.float32)
    for st in range(0, max(1, x.size - n), hop):
        seg = x[st:st + n]
        if seg.size < n or np.sqrt(np.mean(seg ** 2)) < 0.01:
            continue
        spec = np.abs(np.fft.rfft(seg * w)) ** 2
        ceps = np.fft.irfft(np.log(spec + 1e-10))[qlo:qhi]
        mad = np.median(np.abs(ceps)) + 1e-9
        k = int(np.argmax(ceps))
        z = ceps[k] / mad
        d = (k + qlo) / SR
        if z > 25:
            if beats.size and np.any(np.abs((d / beats) - np.round(d / beats)) < 0.03):
                continue  # matches the song's own beat -> normal loop, not a duplicate
            hits.append((st / SR, (st + n) / SR, d, z))
    used = [False] * len(hits)
    for i, h in enumerate(hits):
        if used[i]:
            continue
        grp = [j for j in range(i, len(hits)) if abs(hits[j][2] - h[2]) < 0.003]
        if len(grp) >= 2:
            for j in grp:
                used[j] = True
            s, e = hits[grp[0]][0], hits[grp[-1]][1]
            out.append(Defect(DOUBLE_BGM, s, e, f"Same music/sound layered twice, {h[2] * 1000:.0f} ms apart (echo effect)",
                              clamp01(0.55 + 0.05 * len(grp)), review=True))
    return out


# ---------------------------------------------------------------- 10. language change
LANG_NAMES = {
    "en": "English", "hi": "Hindi", "ur": "Urdu", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi", "or": "Odia", "as": "Assamese",
    "ne": "Nepali", "si": "Sinhala", "sa": "Sanskrit", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ar": "Arabic", "tr": "Turkish", "id": "Indonesian", "ms": "Malay", "th": "Thai", "vi": "Vietnamese",
    "nl": "Dutch", "pl": "Polish", "uk": "Ukrainian", "fa": "Persian", "sw": "Swahili",
}
SAME_GROUP = [{"hi", "ur"}, {"no", "nn"}, {"ms", "id"}]


def lang_name(c: str) -> str:
    return LANG_NAMES.get(c, c)


def _same(a: str, b: str) -> bool:
    return a == b or any(a in g and b in g for g in SAME_GROUP)


def speech_chunks(segs: list[tuple[float, float]], target: float = 10.0) -> list[tuple[float, float]]:
    chunks: list[list[float]] = []
    for s, e in segs:
        while e - s > target * 1.5:
            chunks.append([s, s + target])
            s += target
        if chunks and s - chunks[-1][1] < 1.0 and e - chunks[-1][0] <= target * 1.3:
            chunks[-1][1] = e
        else:
            chunks.append([s, e])
    return [(a, b) for a, b in chunks if b - a >= 2.0]


_WHISPER = None


def _whisper():
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel

        path = models_dir() / "whisper-small"
        if not (path / "model.bin").exists():
            return None
        # graphics card if NVIDIA CUDA libraries are installed, otherwise CPU
        # (float32 is ~3x faster than int8 on CPUs without VNNI instructions)
        try:
            m = WhisperModel(str(path), device="cuda", compute_type="float16")
            m.detect_language(audio=np.zeros(SR, np.float32))
            _WHISPER = m
        except Exception:
            _WHISPER = WhisperModel(str(path), device="cpu", compute_type="float32")
    return _WHISPER


def check_language(ad: AudioData, progress: ProgressFn) -> tuple[list[Defect], str]:
    model = _whisper()
    if model is None:
        return [], "(language model missing)"
    chunks = speech_chunks(ad.speech)
    if not chunks:
        return [], "(no speech found)"
    res = []
    for i, (s, e) in enumerate(chunks):
        progress(f"  Listening for language... piece {i + 1}/{len(chunks)}")
        lang, prob, allp = model.detect_language(audio=ad.x[int(s * SR):int(e * SR)])
        res.append((s, e, lang, prob, dict(allp)))
    score: dict[str, float] = {}
    for s, e, lang, prob, allp in res:
        for k, v in allp.items():
            score[k] = score.get(k, 0.0) + v * (e - s)
    main = max(score, key=score.get)
    flagged = []
    for s, e, lang, prob, allp in res:
        main_p = sum(v for k, v in allp.items() if _same(k, main))
        if not _same(lang, main) and prob >= 0.6 and main_p < 0.25:
            flagged.append((s, e, lang, prob))
    out = []
    i = 0
    while i < len(flagged):
        s, e, lang, prob = flagged[i]
        probs = [prob]
        while i + 1 < len(flagged) and flagged[i + 1][2] == lang and flagged[i + 1][0] - e < 2.0:
            i += 1
            e = flagged[i][1]
            probs.append(flagged[i][3])
        out.append(Defect(LANGUAGE_CHANGE, s, e, f"Main language is {lang_name(main)}, but here it sounds like {lang_name(lang)}",
                          float(np.mean(probs)), review=float(np.mean(probs)) < 0.8))
        i += 1
    return out, lang_name(main)


# ---------------------------------------------------------------- picture + sound helpers
def _speech_envelope(ad: AudioData, times: np.ndarray) -> np.ndarray:
    """Speech-band loudness sampled at each video frame time."""
    lo, hi = int(300 / (SR / 1024)), int(3000 / (SR / 1024))
    band = np.log1p(ad.spec[:, lo:hi].sum(axis=1))
    return np.interp(times, ad.spec_t, band)


def _track_signal(tr, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    sig = np.full(n_frames, np.nan)
    idx = np.array(tr.idxs)
    sig[idx] = np.array(tr.mouth, dtype=float)
    present = ~np.isnan(sig)
    return sig, present


def _znorm(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-9)


# ---------------------------------------------------------------- 1. audio delay (lip sync)
def check_lipsync(ad: AudioData, vd: VideoData, info: MediaInfo) -> list[Defect]:
    out: list[Defect] = []
    off = info.audio_start - info.video_start
    if abs(off) > 0.045:
        out.append(Defect(AUDIO_DELAY, 0.0, None,
                          f"File says sound starts {abs(off) * 1000:.0f} ms {'after' if off > 0 else 'before'} the picture "
                          f"(container offset) - some players/editors ignore this", 0.6, review=True))
    if not vd.tracks:
        return out
    fps = vd.fps
    n = vd.times.size
    env = _speech_envelope(ad, vd.times)
    env_d = np.concatenate([[0], np.abs(np.diff(env))])
    speech = np.zeros(n, bool)
    for s, e in ad.speech:
        speech[(vd.times >= s) & (vd.times <= e)] = True
    maxlag = int(round(0.5 * fps))
    W, step = int(6 * fps), int(2 * fps)
    tol = max(2, int(round(0.1 * fps)))  # ~100 ms
    # Only report a delay when the evidence is strong and repeated: one window alone is too noisy.
    # Windows of each face are grouped into ~30 s blocks; their match curves are averaged.
    for tr in vd.tracks:
        sig, present = _track_signal(tr, n)
        block: list[tuple[float, float, np.ndarray]] = []

        def flush():
            if len(block) >= 4:
                tot = np.mean([b[2] for b in block], axis=0)
                k = int(np.argmax(tot))
                lags = np.array([int(np.argmax(b[2])) for b in block])
                agree = float(np.mean(np.abs(lags - k) <= 2))
                peak, sharp = float(tot[k]), float(tot[k] - np.median(tot))
                lag = (k - maxlag) / fps
                if abs(k - maxlag) >= tol and peak > 0.3 and sharp > 0.2 and agree >= 0.6:
                    late = "late" if lag > 0 else "early"
                    out.append(Defect(AUDIO_DELAY, block[0][0], block[-1][1],
                                      f"Voice is ~{abs(lag) * 1000:.0f} ms {late} compared to the lips",
                                      clamp01(0.4 + peak * 0.6), review=peak < 0.45))
            block.clear()

        for st in range(tr.first_idx, max(tr.first_idx + 1, tr.last_idx - W), step):
            sl = slice(st, st + W)
            if st + W > n:
                break
            if present[sl].mean() < 0.85 or speech[sl].mean() < 0.4:
                continue
            m = sig[sl].copy()
            m[np.isnan(m)] = np.nanmedian(m)
            if np.std(m) < 0.3:
                continue
            m = _znorm(uniform_filter1d(m, 3))
            a = _znorm(uniform_filter1d(env_d[sl], 3))
            cc = np.array([np.mean(m[max(0, -k):W - max(0, k)] * a[max(0, k):W - max(0, -k)]) for k in range(-maxlag, maxlag + 1)])
            t0, t1 = float(vd.times[st]), float(vd.times[min(st + W, n - 1)])
            if block and (t0 - block[-1][0] > 2.5 * step / fps or t0 - block[0][0] > 30):
                flush()
            block.append((t0, t1, cc))
        flush()
    out.sort(key=lambda d: d.start)
    return out


# ---------------------------------------------------------------- 8. wrong voice gender & mouth without voice
def _voice_p_male(ad: AudioData, s: float, e: float) -> tuple[float, float] | None:
    p = ad.pitch
    if p is None:
        return None
    t = ad.spec_t[p.idx]
    inside = (t >= s) & (t <= e)
    m = inside & (p.s1 >= 0.5)  # clearly voiced frames only
    if m.sum() < 25 or m.sum() < 0.3 * inside.sum():
        return None
    f = p.f1[m]
    f0 = float(np.median(f))
    # share of voiced moments that are low (male range); music bass notes only cause a few
    low = float(np.mean(f < 150))
    p_male = float(np.clip((low - 0.2) / 0.6, 0, 1))
    return f0, p_male


def check_gender_and_silent_mouth(ad: AudioData, vd: VideoData) -> list[Defect]:
    out: list[Defect] = []
    if not vd.tracks:
        return out
    n = vd.times.size
    fps = vd.fps
    # pieces of speech of at most 4 s
    pieces = []
    for s, e in ad.speech:
        while e - s > 4.0:
            pieces.append((s, s + 4.0))
            s += 4.0
        if e - s >= 0.8:
            pieces.append((s, e))
    sigs = {tr.tid: _track_signal(tr, n) for tr in vd.tracks}

    if vd.gender_ok:
        per_track: dict[int, list] = {}
        raw: list[tuple] = []  # (start, end, face, voice, f0, conf, clear)
        for s, e in pieces:
            fr = (vd.times >= s) & (vd.times <= e)
            if not fr.any():
                continue
            cands = []
            for tr in vd.tracks:
                sig, present = sigs[tr.tid]
                if present[fr].mean() < 0.5:
                    continue
                cands.append((float(np.nanmean(sig[fr])), tr))
            if not cands:
                continue
            cands.sort(key=lambda c: c[0], reverse=True)
            speaker_mv, tr = cands[0]
            clear = len(cands) == 1 or speaker_mv > 1.5 * cands[1][0]
            if len(tr.genders) < 3:
                continue
            face_pm = float(np.mean([g[1] for g in tr.genders]))
            vp = _voice_p_male(ad, s, e)
            if vp is None:
                continue
            f0, voice_pm = vp
            per_track.setdefault(tr.tid, []).append((s, e, voice_pm, f0))
            face_conf = abs(face_pm - 0.5) * 2
            voice_conf = abs(voice_pm - 0.5) * 2
            if face_conf > 0.5 and voice_conf > 0.5 and (face_pm > 0.5) != (voice_pm > 0.5):
                face = "male" if face_pm > 0.5 else "female"
                voice = "male" if voice_pm > 0.5 else "female"
                conf = min(face_conf, voice_conf) * (0.85 if clear else 0.6)
                raw.append((s, e, face, voice, f0, conf, clear))
        # join back-to-back pieces with the same mismatch into one entry
        groups: list[list] = []
        for r in raw:
            if groups and groups[-1][-1][2:4] == r[2:4] and r[0] - groups[-1][-1][1] < 1.5:
                groups[-1].append(r)
            else:
                groups.append([r])
        for g in groups:
            f0 = float(np.median([r[4] for r in g]))
            clear = all(r[6] for r in g)
            out.append(Defect(GENDER_MISMATCH, g[0][0], g[-1][1],
                              f"Speaking face looks {g[0][2]}, but the voice sounds {g[0][3]} (voice pitch ~{f0:.0f} Hz)"
                              + ("" if clear else "; several faces on screen"),
                              clamp01(float(np.mean([r[5] for r in g]))), review=True))
        # same face, voice flips between male and female
        for tid, lst in per_track.items():
            for (s1, e1, p1, f1), (s2, e2, p2, f2) in zip(lst, lst[1:]):
                if abs(p1 - p2) > 0.8 and s2 - e1 < 5:
                    out.append(Defect(GENDER_MISMATCH, s1, e2,
                                      f"Voice of the same face changes from {'male' if p1 > 0.5 else 'female'} "
                                      f"(~{f1:.0f} Hz) to {'male' if p2 > 0.5 else 'female'} (~{f2:.0f} Hz)", 0.6, review=True))
        out.sort(key=lambda d: d.start)

    # mouth clearly moving (talking) but there is no voice
    speech = np.zeros(n, bool)
    for s, e in ad.speech:
        speech[(vd.times >= s - 0.2) & (vd.times <= e + 0.2)] = True
    for tr in vd.tracks:
        sig, present = sigs[tr.tid]
        talk_level = np.nanpercentile(sig, 60) if np.isfinite(sig).sum() > 10 else np.nan
        if not np.isfinite(talk_level) or np.nanmedian(np.array(tr.sizes)) < 40:
            continue
        s_ = np.where(np.isnan(sig), 0.0, sig)
        sm = uniform_filter1d(s_, max(3, int(0.5 * fps)))
        spk_level = np.nanmedian(sig[speech & present]) if (speech & present).sum() > fps else np.nan
        if not np.isfinite(spk_level) or spk_level < 1.0:
            continue
        talking = present & (sm > 0.8 * spk_level) & ~speech
        for a, b in runs(talking):
            if (b - a) / fps >= 1.5:
                # confirmed as real 'voice missing' defects on a real lesson video (2026-11-18)
                out.append(Defect(NO_AUDIO, vd.times[a], vd.times[b - 1],
                                  "Voice missing: the person is talking (mouth moving) but no voice is heard",
                                  0.75, review=True))
    return out
