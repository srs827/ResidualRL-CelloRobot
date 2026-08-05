"""
feature_extractor.py

Recreates the 6 Essentia audio features used by the original DeepMLP
classifier, using librosa + pyloudnorm so the pipeline runs on Windows.

Features (must match selected_features.json):
    lowlevel.loudness_ebu128.short_term.mean
    lowlevel.melbands_crest.mean
    lowlevel.spectral_centroid.stdev
    lowlevel.spectral_entropy.mean
    lowlevel.spectral_flux.stdev
    lowlevel.spectral_rolloff.stdev
    pitch (MIDI number)

NOTE: librosa's implementations differ slightly from Essentia's, so the
absolute values won't perfectly match the values the scaler was trained on.
That's an acceptable trade-off for the demo; long term we'll retrain the
classifier on librosa features.
"""

from __future__ import annotations

import numpy as np
import librosa
import pyloudnorm as pyln


# ---------------------------------------------------------------------------
# Configuration (kept close to Essentia's MusicExtractor defaults)
# ---------------------------------------------------------------------------

DEFAULT_SR    = 44100
FRAME_LENGTH  = 2048
HOP_LENGTH    = 1024
N_MEL_BANDS   = 40


# ---------------------------------------------------------------------------
# Individual feature computations
# ---------------------------------------------------------------------------

def _loudness_ebu128_short_term_mean(y: np.ndarray, sr: int) -> float:
    """
    EBU R128 short-term loudness, averaged across the file.

    Essentia: lowlevel.loudness_ebu128.short_term.mean
    """
    meter = pyln.Meter(sr, block_size=0.4)  # 400ms short-term window
    # pyloudnorm.integrated_loudness gives a single number across the file,
    # which roughly approximates the mean of short-term loudness windows.
    try:
        loudness = meter.integrated_loudness(y)
    except ValueError:
        # File too short for the meter, fall back to -inf-like value
        loudness = -70.0
    return float(loudness)


def _melbands_crest_mean(y: np.ndarray, sr: int) -> float:
    """
    Mean of mel-band crest factors over time.
    Crest factor per frame = max(band_energy) / mean(band_energy).

    Essentia: lowlevel.melbands_crest.mean
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH,
        n_mels=N_MEL_BANDS,
        power=2.0,
    )
    # crest factor per frame, then mean over time
    eps = 1e-12
    crest_per_frame = mel.max(axis=0) / (mel.mean(axis=0) + eps)
    return float(np.mean(crest_per_frame))


def _spectral_centroid_stdev(y: np.ndarray, sr: int) -> float:
    """
    Standard deviation of spectral centroid over time.

    Essentia: lowlevel.spectral_centroid.stdev
    """
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr,
        n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH,
    )
    return float(np.std(centroid))


def _spectral_entropy_mean(y: np.ndarray, sr: int) -> float:
    """
    Mean of spectral entropy over time, computed as Shannon entropy (in bits)
    of the normalised magnitude spectrum.

    Essentia: lowlevel.spectral_entropy.mean
    NOTE: Essentia returns raw entropy (not normalised by log2(n_bins)), so
    we match that — values typically fall in [4, 10] for n_fft=2048.
    """
    S = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
    eps = 1e-12
    p = S / (S.sum(axis=0, keepdims=True) + eps)
    entropy = -np.sum(p * np.log2(p + eps), axis=0)  # raw entropy, no normalisation
    return float(np.mean(entropy))


def _spectral_flux_stdev(y: np.ndarray, sr: int) -> float:
    """
    Standard deviation of spectral flux over time.
    Flux per frame = L2 norm of positive differences in the *normalised*
    magnitude spectrum.

    Essentia: lowlevel.spectral_flux.stdev
    NOTE: Essentia normalises each magnitude spectrum to unit sum before
    diffing, which keeps flux values in [0, ~0.5]. Skipping that step
    (raw L2 of diffs) produces values 100× larger and breaks the scaler.
    """
    S = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
    # Normalise each frame to unit sum so flux is unitless and bounded
    S = S / (S.sum(axis=0, keepdims=True) + 1e-12)
    diff = np.diff(S, axis=1)
    diff = np.maximum(diff, 0.0)  # half-wave rectification
    flux = np.sqrt(np.sum(diff ** 2, axis=0))
    return float(np.std(flux))


def _spectral_rolloff_stdev(y: np.ndarray, sr: int) -> float:
    """
    Standard deviation of spectral rolloff over time.

    Essentia: lowlevel.spectral_rolloff.stdev
    """
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr,
        n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH,
        roll_percent=0.85,
    )
    return float(np.std(rolloff))


# ---------------------------------------------------------------------------
# Top-level extractor
# ---------------------------------------------------------------------------

# Map Essentia feature names → callables. Keep names exactly matching
# selected_features.json so the scaler can pick the right columns.
FEATURE_FUNCS = {
    "lowlevel.loudness_ebu128.short_term.mean": _loudness_ebu128_short_term_mean,
    "lowlevel.melbands_crest.mean":              _melbands_crest_mean,
    "lowlevel.spectral_centroid.stdev":          _spectral_centroid_stdev,
    "lowlevel.spectral_entropy.mean":            _spectral_entropy_mean,
    "lowlevel.spectral_flux.stdev":              _spectral_flux_stdev,
    "lowlevel.spectral_rolloff.stdev":           _spectral_rolloff_stdev,
}


def extract_features(
    y: np.ndarray,
    sr: int,
    pitch_midi: float,
) -> dict:
    """
    Extract the 7 features the DeepMLP expects.

    Args:
        y          : mono float32 audio (any length)
        sr         : sample rate
        pitch_midi : MIDI note number (e.g. A3 = 57.0). Caller is responsible
                     for figuring this out — either from filename, or by
                     calling librosa.yin / pyin.

    Returns:
        dict[str, float] keyed by Essentia feature name + "pitch"
    """
    feats = {name: fn(y, sr) for name, fn in FEATURE_FUNCS.items()}
    feats["pitch"] = float(pitch_midi)
    return feats


def estimate_pitch_midi(y: np.ndarray, sr: int) -> float:
    """
    Estimate fundamental pitch using librosa.pyin, return as MIDI note number.
    Useful when we don't have the note name from a filename.
    """
    f0, _, _ = librosa.pyin(
        y, sr=sr,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
    )
    # average the voiced frames
    f0 = f0[~np.isnan(f0)]
    if len(f0) == 0:
        return 57.0  # fallback to A3
    mean_hz = float(np.mean(f0))
    return float(librosa.hz_to_midi(mean_hz))
