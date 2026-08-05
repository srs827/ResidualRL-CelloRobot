"""
mic_test.py

Standalone microphone test for Audio-Technica + Focusrite setup.
No robot required — validates the full audio chain before a recording session.

Tests:
    1. Device discovery       — lists all devices, finds Focusrite
    2. Noise floor            — 3s silence recording, checks RMS / peak
    3. Live level monitor     — real-time dBFS meter in terminal (Ctrl+C to stop)
    4. Signal recording       — records N seconds, saves to WAV for inspection
    5. Frequency response     — records a tone/clap, plots spectrum (optional)

Usage:
    python mic_test.py                  # run all tests interactively
    python mic_test.py --device 2       # force a specific device index
    python mic_test.py --list           # just list devices and exit
    python mic_test.py --no-plot        # skip the spectrum plot

Requirements:
    pip install sounddevice soundfile numpy matplotlib
"""

import argparse
import time
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_RATE     = 44100     # Must match Focusrite clock (check Audio MIDI Setup)
CHANNELS        = 1         # Mono — mic on input 1
DTYPE           = 'float32'

NOISE_FLOOR_DUR = 3.0       # seconds of silence to record for noise floor test
SIGNAL_REC_DUR  = 5.0       # seconds for the signal recording test
METER_BAR_WIDTH = 40        # terminal VU meter width

OUTPUT_DIR      = Path("mic_test_output")

# dBFS thresholds
CLIP_THRESHOLD_DBFS  = -1.0   # warn if peak gets this close to 0 dBFS
NOISE_MAX_DBFS       = -50.0  # noise floor should be below this
SIGNAL_MIN_DBFS      = -30.0  # a healthy signal should exceed this during playing


# ── Device helpers ────────────────────────────────────────────────────────────

def list_devices():
    """Print a formatted table of all audio input devices."""
    devices = sd.query_devices()
    print("\n── Audio Devices ─────────────────────────────────────────────")
    print(f"  {'IDX':>3}  {'Name':<40}  {'In':>3}  {'Out':>3}  {'Default SR':>10}")
    print(f"  {'───':>3}  {'────':<40}  {'──':>3}  {'───':>3}  {'──────────':>10}")
    for i, dev in enumerate(devices):
        marker = "►" if dev['max_input_channels'] > 0 else " "
        print(f"  {marker}{i:>3}  {dev['name']:<40}  "
              f"{dev['max_input_channels']:>3}  "
              f"{dev['max_output_channels']:>3}  "
              f"{int(dev['default_samplerate']):>10}")
    print()


def find_focusrite(verbose: bool = True) -> int | None:
    """
    Auto-detect Focusrite Scarlett by name.
    Returns device index or None if not found.
    """
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev['name']
        if ('Scarlett' in name or 'Focusrite' in name) and dev['max_input_channels'] >= 1:
            if verbose:
                print(f"  Found Focusrite at index {i}: {name}")
                print(f"    Input channels : {dev['max_input_channels']}")
                print(f"    Default SR     : {int(dev['default_samplerate'])} Hz")
            return i
    return None


def resolve_device(forced_index: int | None) -> int:
    """
    Return device index to use.
    Priority: --device flag > auto-detect > system default.
    """
    if forced_index is not None:
        dev = sd.query_devices(forced_index)
        print(f"  Using forced device [{forced_index}]: {dev['name']}")
        return forced_index

    idx = find_focusrite()
    if idx is not None:
        return idx

    # Fall back to system default input
    default = sd.default.device[0]  # (input, output)
    dev = sd.query_devices(default)
    print(f"  Focusrite not found — using system default [{default}]: {dev['name']}")
    return default


# ── Test 1: Noise floor ───────────────────────────────────────────────────────

def test_noise_floor(device: int) -> dict:
    """
    Record silence and measure noise floor.
    You should have no signal going into the mic during this test.
    """
    print(f"\n── Test 1: Noise Floor ({NOISE_FLOOR_DUR:.0f}s) ───────────────────────────────")
    print("  Keep the room quiet — do not play anything.")
    input("  Press ENTER to start...")

    audio = sd.rec(
        int(NOISE_FLOOR_DUR * SAMPLE_RATE),
        samplerate = SAMPLE_RATE,
        channels   = CHANNELS,
        dtype      = DTYPE,
        device     = device,
    )
    sd.wait()
    audio = audio.flatten()

    peak    = float(np.max(np.abs(audio)))
    rms     = float(np.sqrt(np.mean(audio ** 2)))
    db_peak = 20 * np.log10(peak + 1e-10)
    db_rms  = 20 * np.log10(rms  + 1e-10)

    print(f"\n  Peak : {peak:.5f}  ({db_peak:+.1f} dBFS)")
    print(f"  RMS  : {rms:.5f}  ({db_rms:+.1f} dBFS)")

    if peak < 1e-6:
        print("  ✗  No signal at all — check mic is selected and phantom power is on")
        status = 'no_signal'
    elif db_peak > NOISE_MAX_DBFS:
        print(f"  ⚠  Noise floor too high (peak {db_peak:.1f} dBFS > {NOISE_MAX_DBFS} dBFS)")
        print("     Check for: USB interference, preamp gain too high, cable hum")
        status = 'noisy'
    else:
        print(f"  ✓  Noise floor OK")
        status = 'ok'

    return {'peak': peak, 'rms': rms, 'db_peak': db_peak, 'db_rms': db_rms, 'status': status}


# ── Test 2: Live level meter ──────────────────────────────────────────────────

def test_live_meter(device: int):
    """
    Stream audio and print a real-time dBFS bar meter.
    Press Ctrl+C to stop.
    """
    print(f"\n── Test 2: Live Level Meter ──────────────────────────────────")
    print("  Play/bow the cello. Watch the levels. Ctrl+C to stop.\n")

    BLOCK = 1024

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [stream status: {status}]")
        rms    = float(np.sqrt(np.mean(indata ** 2)))
        peak   = float(np.max(np.abs(indata)))
        db_rms = 20 * np.log10(rms  + 1e-10)
        db_pk  = 20 * np.log10(peak + 1e-10)

        # Normalise -60 dBFS → 0 dBFS to bar width
        norm  = max(0.0, min(1.0, (db_rms + 60) / 60))
        n_bar = int(norm * METER_BAR_WIDTH)
        bar   = '█' * n_bar + '░' * (METER_BAR_WIDTH - n_bar)

        clip_flag = ' CLIP!' if db_pk > CLIP_THRESHOLD_DBFS else '      '
        print(f"\r  [{bar}] {db_rms:+5.1f} dBFS  peak {db_pk:+5.1f}{clip_flag}",
              end='', flush=True)

    try:
        with sd.InputStream(
            samplerate = SAMPLE_RATE,
            channels   = CHANNELS,
            dtype      = DTYPE,
            device     = device,
            blocksize  = BLOCK,
            callback   = callback,
        ):
            while True:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n  Meter stopped.")


# ── Test 3: Signal recording + WAV save ──────────────────────────────────────

def test_signal_recording(device: int) -> Path | None:
    """
    Record SIGNAL_REC_DUR seconds with signal present and save to WAV.
    """
    print(f"\n── Test 3: Signal Recording ({SIGNAL_REC_DUR:.0f}s) ─────────────────────────")
    print("  Make sound during recording (bow a note, clap, etc.).")
    input("  Press ENTER to start...")

    audio = sd.rec(
        int(SIGNAL_REC_DUR * SAMPLE_RATE),
        samplerate = SAMPLE_RATE,
        channels   = CHANNELS,
        dtype      = DTYPE,
        device     = device,
    )

    # Live countdown
    for i in range(int(SIGNAL_REC_DUR), 0, -1):
        print(f"\r  Recording... {i}s remaining  ", end='', flush=True)
        time.sleep(1)
    sd.wait()
    print("\r  Done.                           ")

    audio = audio.flatten()
    peak    = float(np.max(np.abs(audio)))
    rms     = float(np.sqrt(np.mean(audio ** 2)))
    db_peak = 20 * np.log10(peak + 1e-10)
    db_rms  = 20 * np.log10(rms  + 1e-10)

    print(f"\n  Peak : {peak:.5f}  ({db_peak:+.1f} dBFS)")
    print(f"  RMS  : {rms:.5f}  ({db_rms:+.1f} dBFS)")

    if db_peak > CLIP_THRESHOLD_DBFS:
        print("  ⚠  Clipping detected — reduce Focusrite preamp gain")
    elif db_peak < SIGNAL_MIN_DBFS:
        print("  ⚠  Signal weak — increase Focusrite gain, or check mic position")
    else:
        print("  ✓  Signal level looks good")

    # Save WAV
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"mic_test_{ts}.wav"
    sf.write(str(path), audio, SAMPLE_RATE)
    print(f"\n  Saved: {path}")

    return path, audio


# ── Test 4: Spectrum plot ─────────────────────────────────────────────────────

def plot_spectrum(audio: np.ndarray, title: str = "Frequency Spectrum"):
    """
    Plot magnitude spectrum of recorded audio.
    Useful for checking frequency response and spotting hum (50/60 Hz).
    """
    if not HAS_MATPLOTLIB:
        print("  (matplotlib not installed — skipping spectrum plot)")
        return

    n     = len(audio)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    mag   = np.abs(np.fft.rfft(audio)) / n
    db    = 20 * np.log10(mag + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(freqs[1:], db[1:], linewidth=0.7, color='steelblue')
    ax.set_xlim(20, SAMPLE_RATE // 2)
    ax.set_ylim(-100, 0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dBFS)")
    ax.set_title(title)
    ax.axvline(60,  color='red',    alpha=0.5, linestyle='--', label='60 Hz hum')
    ax.axvline(50,  color='orange', alpha=0.5, linestyle='--', label='50 Hz hum')
    ax.axvline(220, color='green',  alpha=0.3, linestyle=':',  label='A3 (220 Hz)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"spectrum_{ts}.png"
    plt.savefig(str(path), dpi=150)
    print(f"  Spectrum saved: {path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Microphone test for AT mic + Focusrite")
    parser.add_argument('--device',  type=int, default=None,
                        help='Force a specific sounddevice device index')
    parser.add_argument('--list',    action='store_true',
                        help='List all audio devices and exit')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip the frequency spectrum plot')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Microphone Test — AT mic + Focusrite")
    print("="*60)

    list_devices()

    if args.list:
        return

    device = resolve_device(args.device)

    # Sanity check: can we even open the device at this sample rate?
    print(f"\n  Checking {SAMPLE_RATE} Hz is supported by device {device}...")
    try:
        sd.check_input_settings(device=device, channels=CHANNELS,
                                dtype=DTYPE, samplerate=SAMPLE_RATE)
        print("  ✓  Sample rate OK")
    except Exception as e:
        print(f"  ✗  {e}")
        print("     Try setting the Focusrite to 44100 Hz in macOS Audio MIDI Setup")
        print("     (Applications → Utilities → Audio MIDI Setup)")
        return

    # ── Run tests ──

    # Test 1: noise floor
    noise_result = test_noise_floor(device)

    # Test 2: live meter
    print("\n  (You can skip the live meter test by pressing Ctrl+C immediately)")
    try:
        test_live_meter(device)
    except KeyboardInterrupt:
        pass

    # Test 3: signal recording
    result = test_signal_recording(device)
    if result is None:
        return
    wav_path, audio = result

    # Test 4: spectrum
    if not args.no_plot:
        print(f"\n── Test 4: Frequency Spectrum ────────────────────────────────")
        plot_spectrum(audio, title=f"Spectrum — {wav_path.name}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  Device       : {sd.query_devices(device)['name']}")
    print(f"  Sample rate  : {SAMPLE_RATE} Hz")
    print(f"  Noise floor  : {noise_result['db_peak']:+.1f} dBFS peak  "
          f"({'OK' if noise_result['status'] == 'ok' else noise_result['status']})")
    print(f"  WAV saved to : {wav_path}")
    print()
    print("  Next step: open the WAV in Audacity or a DAW to verify it")
    print("  sounds clean before starting a robot recording session.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()