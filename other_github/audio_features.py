"""
audio_features.py

Hand-engineered audio quality features for a single bow stroke (or a sliding
window cut from one), plus the log-mel spectrogram used as the CNN's image
input. Everything here operates on a mono float32 array at a fixed sample
rate -- callers are responsible for loading/slicing/resampling audio before
calling in.

Used by:
    dataset.py            -- offline feature extraction for training
    classifier.py          -- same extraction, called per real-time window
    inspect_features.py   -- standalone diagnostic / correlation tool
"""

import numpy as np
import librosa

SR           = 44100
N_FFT        = 1024
HOP_LENGTH   = 256
N_MELS       = 64
N_MFCC       = 13
F0_MIN_HZ    = 70.0    # cello A string open ~220Hz, but allow for low/buzzy notes
F0_MAX_HZ    = 1500.0
RELATIVE_SILENCE_DB = -30.0   # frames this far below the window's own peak frame RMS = silence/no-contact


def _silence_threshold(rms, relative_db=RELATIVE_SILENCE_DB):
    """Per-window noise floor: a fixed dB offset below this window's own peak frame RMS,
    instead of one fixed absolute level -- robust to gain/mic-noise differences across
    recordings."""
    peak = float(np.max(rms)) if len(rms) else 0.0
    return peak * (10.0 ** (relative_db / 20.0))

SCALAR_FEATURE_NAMES = [
    'hnr_db_mean', 'hnr_db_std',
    'flatness_mean', 'flatness_std',
    'f0_mean_hz', 'f0_stability_cents', 'voiced_fraction',
    'rms_mean', 'rms_std', 'envelope_cv', 'envelope_trend', 'envelope_outlier_rate',
    'attack_time_s', 'attack_overshoot',
] + [f'mfcc{i}_mean' for i in range(N_MFCC)] + [f'mfcc{i}_std' for i in range(N_MFCC)]

N_SCALAR_FEATURES = len(SCALAR_FEATURE_NAMES)


# ── I/O ──────────────────────────────────────────────────────────────

def load_audio(path, sr=SR):
    """Mono float32 array at `sr`, resampling if the file differs."""
    audio, file_sr = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32)


def normalize_peak(audio, target_peak=1.0, eps=1e-6):
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak < eps:
        return audio
    return (audio * (target_peak / peak)).astype(np.float32)


# ── Mel spectrogram (CNN input) ───────────────────────────────────────

def compute_log_mel_spectrogram(audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
                                 hop_length=HOP_LENGTH, target_frames=None):
    """
    Log-power mel spectrogram, shape (n_mels, n_frames).
    If `target_frames` is given, pad with the spectrogram's own floor or
    crop (center) so every window produces an identically-shaped tensor.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if target_frames is not None:
        log_mel = _pad_or_crop_frames(log_mel, target_frames)
    return log_mel.astype(np.float32)


def _pad_or_crop_frames(spec, target_frames):
    n = spec.shape[1]
    if n == target_frames:
        return spec
    if n > target_frames:
        start = (n - target_frames) // 2
        return spec[:, start:start + target_frames]
    pad_total = target_frames - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    floor = spec.min()
    return np.pad(spec, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=floor)


def mel_spectrogram_frames_for_duration(duration_s, sr=SR, hop_length=HOP_LENGTH):
    """Frame count librosa produces for `duration_s` seconds of audio at this hop."""
    n_samples = int(round(duration_s * sr))
    return 1 + n_samples // hop_length


# ── Harmonic-to-noise ratio ───────────────────────────────────────────

def compute_hnr(audio, sr=SR, frame_length=2048, hop_length=HOP_LENGTH,
                 f0_min=F0_MIN_HZ, f0_max=F0_MAX_HZ):
    """
    Frame-wise HNR (dB) via normalized autocorrelation: for each frame, find
    the autocorrelation peak within the plausible pitch-period lag range and
    convert the peak's normalized correlation coefficient to a harmonic/noise
    energy ratio in dB (de Krom-style estimator). Low HNR = noisy/scratchy
    stroke; high HNR = clean tone.
    Returns (mean_db, std_db) over non-silent frames; (0.0, 0.0) if no
    voiced frames are found.
    """
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length).T
    window = np.hanning(frame_length)

    min_lag = max(1, int(sr / f0_max))
    max_lag = min(frame_length - 1, int(sr / f0_min))

    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    threshold = _silence_threshold(frame_rms)

    hnr_vals = []
    for frame, energy in zip(frames, frame_rms):
        if energy < threshold:
            continue
        w = frame * window
        ac = np.correlate(w, w, mode='full')
        r0 = ac[frame_length - 1]
        if r0 <= 1e-12:
            continue
        ac = ac[frame_length - 1:] / r0   # normalized, lag 0 first

        lag_slice = ac[min_lag:max_lag + 1]
        if lag_slice.size == 0:
            continue
        best = float(np.max(lag_slice))
        best = float(np.clip(best, 1e-6, 1 - 1e-6))
        hnr_db = 10.0 * np.log10(best / (1.0 - best))
        hnr_vals.append(hnr_db)

    if not hnr_vals:
        return 0.0, 0.0
    return float(np.mean(hnr_vals)), float(np.std(hnr_vals))


# ── Spectral flatness ──────────────────────────────────────────────────

def compute_spectral_flatness(audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Mean/std of frame-wise spectral flatness in [0,1]. ~0 = tonal, ~1 = noisy."""
    flat = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)[0]
    return float(np.mean(flat)), float(np.std(flat))


# ── F0 stability ────────────────────────────────────────────────────────

def compute_f0_stability(audio, sr=SR, frame_length=2048, hop_length=HOP_LENGTH,
                          f0_min=F0_MIN_HZ, f0_max=F0_MAX_HZ):
    """
    Tracks F0 with the (fast, non-probabilistic) YIN algorithm and masks out
    unvoiced/silent frames by RMS energy rather than YIN's own confidence
    (YIN always returns an estimate, even on silence). Stability is the
    spread of the voiced F0 contour in cents around its median -- a clean,
    held tone is flat; a wobbling/breaking tone has wide spread.
    Returns (f0_mean_hz, f0_stability_cents, voiced_fraction).
    """
    f0 = librosa.yin(audio, fmin=f0_min, fmax=f0_max, sr=sr,
                      frame_length=frame_length, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]

    voiced = rms >= _silence_threshold(rms)
    voiced_fraction = float(np.mean(voiced)) if n else 0.0
    f0_voiced = f0[voiced]
    if f0_voiced.size < 2:
        return 0.0, 0.0, voiced_fraction

    median_f0 = float(np.median(f0_voiced))
    if median_f0 <= 1e-6:
        return 0.0, 0.0, voiced_fraction

    cents = 1200.0 * np.log2(np.clip(f0_voiced / median_f0, 1e-3, None))
    return float(np.mean(f0_voiced)), float(np.std(cents)), voiced_fraction


# ── RMS envelope ────────────────────────────────────────────────────────

def compute_rms_envelope_features(audio, sr=SR, frame_length=2048, hop_length=HOP_LENGTH):
    """
    Envelope shape/stability over the stroke:
        envelope_cv          -- std/mean, overall instability
        envelope_trend       -- linear-fit slope of envelope vs time, normalized
                                 by mean level (relative rise/fall per second)
        envelope_outlier_rate-- fraction of frames that deviate from a local
                                 median by >3 MAD (spikes/dips/dropouts)
    """
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    mean = float(np.mean(rms))
    std = float(np.std(rms))
    if mean < 1e-9:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    cv = std / mean

    t = np.arange(len(rms)) * (hop_length / sr)
    if len(t) >= 2:
        slope = float(np.polyfit(t, rms, 1)[0])
        trend = slope / mean
    else:
        trend = 0.0

    med = np.median(rms)
    mad = np.median(np.abs(rms - med)) + 1e-9
    outlier_rate = float(np.mean(np.abs(rms - med) > 3 * mad))

    return mean, std, cv, trend, outlier_rate


# ── Attack quality ──────────────────────────────────────────────────────

def compute_attack_quality(audio, sr=SR, frame_length=2048, hop_length=HOP_LENGTH):
    """
    Characterizes the onset of the stroke:
        attack_time_s   -- time from 10% to 80% of steady-state envelope level
        attack_overshoot-- peak envelope during the attack relative to the
                            steady-state level (1.0 = no overshoot; well above
                            1 means a scratchy/noisy transient before the tone settles)
    A clean bow stroke has a short, low-overshoot attack. Returns (0.0, 1.0)
    when no clear onset is found (e.g. near-silent recording).
    """
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) < 4:
        return 0.0, 1.0

    steady_region = rms[len(rms) // 3:]
    steady_level = float(np.median(steady_region)) if steady_region.size else float(np.median(rms))
    if steady_level < max(_silence_threshold(rms), 1e-9):
        return 0.0, 1.0

    onset_idx = next((i for i, v in enumerate(rms) if v >= 0.1 * steady_level), None)
    if onset_idx is None:
        return 0.0, 1.0
    rise_idx = next((i for i in range(onset_idx, len(rms)) if rms[i] >= 0.8 * steady_level), None)
    if rise_idx is None:
        rise_idx = len(rms) - 1

    attack_time_s = (rise_idx - onset_idx) * hop_length / sr
    attack_peak = float(np.max(rms[onset_idx:rise_idx + 1]))
    overshoot = attack_peak / steady_level

    return float(attack_time_s), float(overshoot)


# ── MFCC (timbre) ────────────────────────────────────────────────────────

def compute_mfcc_stats(audio, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Per-coefficient mean and std over the stroke. Returns (mean[n_mfcc], std[n_mfcc])."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc.mean(axis=1).astype(np.float32), mfcc.std(axis=1).astype(np.float32)


# ── Combined scalar feature vector ───────────────────────────────────────

def extract_scalar_features(audio, sr=SR) -> np.ndarray:
    """
    Fixed-length vector of all hand-engineered features, in the order of
    SCALAR_FEATURE_NAMES. This is the vector fused into the CNN's head
    alongside the mel spectrogram, and what inspect_features.py reports on.
    """
    hnr_mean, hnr_std = compute_hnr(audio, sr)
    flat_mean, flat_std = compute_spectral_flatness(audio, sr)
    f0_mean, f0_stab, voiced_frac = compute_f0_stability(audio, sr)
    rms_mean, rms_std, env_cv, env_trend, env_outlier = compute_rms_envelope_features(audio, sr)
    attack_time, attack_overshoot = compute_attack_quality(audio, sr)
    mfcc_mean, mfcc_std = compute_mfcc_stats(audio, sr)

    return np.concatenate([
        [hnr_mean, hnr_std, flat_mean, flat_std, f0_mean, f0_stab, voiced_frac,
         rms_mean, rms_std, env_cv, env_trend, env_outlier,
         attack_time, attack_overshoot],
        mfcc_mean, mfcc_std,
    ]).astype(np.float32)
