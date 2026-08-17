"""
rl/harmonicity.py

Score-informed harmonic energy ratio: a short-note tone metric that needs no
human annotation and no FFT resolution.

Why this exists. The two metrics the reward had for "is this a note or a
scrape" both fail on exactly the notes that scrape:

  * hnr_db_mean / voiced_fraction come from librosa STFT at n_fft=1024
    (23.2 ms at 44.1 kHz). yunpiece's fast notes are 8-23 ms, so they were
    zero-padded to one frame and the features measured the padding.
  * The CNN judge scored two takes a listener called "much much better"
    apart at 0.474 vs 0.473 (measured 2026-08-17, .mid vs .mxl on the same
    rig minutes apart). It cannot hear the difference we care about.

The physics gives a way out. A bowed string in Helmholtz motion radiates
energy at a harmonic series of f0; a bow that is rubbing rather than gripping
radiates broadband noise. So "what fraction of this note's energy sits on the
harmonic series?" is a direct, physical read on whether the string is
speaking.

The trick that makes it work on an 8 ms note: we already KNOW f0. This robot
plays the open A string and nothing else (piece_env carries no pitch field at
all; safety_check reports `All strings: {'A'}`), measured at 220.5 Hz on both
takes. Because the frequencies are known in advance we project the signal onto
them directly -- correlate against sin/cos at each harmonic -- instead of
computing a spectrum and reading peaks off it. Direct projection has no
resolution/window tradeoff: it answers "how much energy is at THIS frequency"
for any segment long enough to hold a couple of cycles. At 220.5 Hz one period
is 4.5 ms, so a 20 ms note carries ~4.4 cycles, which is enough.

What comes out, all in [0, 1] and all higher-is-better:

    harmonic_ratio    energy on the harmonic series / total energy
    period_corr       cycle-to-cycle waveform correlation (time domain)
    onset_periods     periods elapsed before the waveform settles, mapped so
                      that a fast, clean attack scores high

Written 2026-08-17 to replace hnr_db_mean on short notes.
"""

from __future__ import annotations

import numpy as np

# Open A, measured from both 2026-08-17 takes by autocorrelation (peak 0.988).
# Not a nominal 220.0: the instrument is tuned where it is tuned, and a 0.2%
# error costs real energy at the 8th harmonic.
F0_A_STRING = 220.5

# Harmonics to include. Cello A has strong partials well past the 10th; the cap
# is the point where 44.1 kHz and the mic response stop being trustworthy.
N_HARMONICS = 12

# Half-width of the band credited to each harmonic, as a fraction of f0. Wide
# enough to tolerate the tuning drift and vibrato-free pitch wobble of a bowed
# open string, narrow enough that broadband noise does not leak in.
BAND_FRAC = 0.10


def _project_energy(x: np.ndarray, sr: float, freqs: np.ndarray) -> np.ndarray:
    """Energy at each of `freqs`, by direct sin/cos projection.

    This is the whole point of the module: no STFT, so no n_fft floor. For a
    segment of n samples the projection is exact for any frequency, and the
    only limit is how many cycles fit -- which is what BAND_FRAC absorbs.
    """
    n = len(x)
    t = np.arange(n) / sr
    # Hann window: without it, a segment holding a non-integer number of
    # cycles leaks energy into neighbouring bins and every note looks noisy.
    w = np.hanning(n) if n > 2 else np.ones(n)
    xw = x * w
    norm = np.sum(w ** 2) + 1e-20
    out = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        c = np.cos(2 * np.pi * f * t)
        s = np.sin(2 * np.pi * f * t)
        out[i] = ((xw @ c) ** 2 + (xw @ s) ** 2) / norm
    return out


def harmonic_ratio(x: np.ndarray, sr: float = 44100.0,
                   f0: float = F0_A_STRING,
                   n_harmonics: int = N_HARMONICS) -> float:
    """Fraction of the segment's energy sitting on the harmonic series of f0.

    1.0 = all energy on the harmonics (Helmholtz). 0.0 = none (rubbing).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 8:
        return 0.0
    x = x - x.mean()
    total = float(np.sum((x * (np.hanning(len(x)) if len(x) > 2 else 1)) ** 2))
    if total <= 1e-20:
        return 0.0

    ks = np.arange(1, n_harmonics + 1)
    freqs = f0 * ks
    freqs = freqs[freqs < 0.45 * sr]
    if not len(freqs):
        return 0.0

    # Credit a narrow band around each harmonic rather than the exact bin: the
    # string is not a perfect oscillator and the projection of a slightly
    # detuned partial splits across nearby frequencies.
    band = max(1, int(round(BAND_FRAC * f0 / max(sr / len(x), 1e-9))))
    offs = np.linspace(-BAND_FRAC * f0, BAND_FRAC * f0, max(3, min(band, 7)))
    e = 0.0
    for f in freqs:
        e += float(np.max(_project_energy(x, sr, f + offs)))
    return float(np.clip(e / total, 0.0, 1.0))


def period_correlation(x: np.ndarray, sr: float = 44100.0,
                       f0: float = F0_A_STRING) -> float:
    """Cycle-to-cycle waveform similarity, in the time domain.

    A speaking note repeats itself every 1/f0; a scrape does not. Needs only
    two periods (9 ms at 220.5 Hz), so it survives where an STFT cannot.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = int(round(sr / f0))
    if len(x) < 2 * p:
        return 0.0
    x = x - x.mean()
    a, b = x[:-p], x[p:]
    d = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if d <= 1e-20:
        return 0.0
    return float(np.clip((a @ b) / d, 0.0, 1.0))


def onset_periods(x: np.ndarray, sr: float = 44100.0,
                  f0: float = F0_A_STRING, max_periods: int = 12) -> float:
    """How quickly the note settles into periodicity, mapped to [0, 1].

    Walks forward one period at a time and finds the first cycle that
    correlates with its successor above 0.7. Returns 1.0 for an immediate
    Helmholtz attack, falling to 0.0 for a note that never settles -- which is
    what a rubbed short note does. This is the attack metric the CNN's
    attack_quality head is supposed to provide but cannot at 8 ms.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = int(round(sr / f0))
    if len(x) < 2 * p:
        return 0.0
    x = x - x.mean()
    n = min(max_periods, len(x) // p - 1)
    for k in range(n):
        a = x[k * p:(k + 1) * p]
        b = x[(k + 1) * p:(k + 2) * p]
        d = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
        if d > 1e-20 and (a @ b) / d > 0.7:
            return float(1.0 - k / max(max_periods, 1))
    return 0.0


def features(x: np.ndarray, sr: float = 44100.0,
             f0: float = F0_A_STRING) -> dict:
    """All three, as a dict ready to merge into a scorer's detail output."""
    return {
        "harmonic_ratio": harmonic_ratio(x, sr, f0),
        "period_corr": period_correlation(x, sr, f0),
        "onset_periods": onset_periods(x, sr, f0),
    }
