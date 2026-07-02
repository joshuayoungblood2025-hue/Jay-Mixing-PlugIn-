"""DSP correctness: FFT, biquads, loudness calibration, compressor/limiter."""

from __future__ import annotations

import math
from array import array

from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp import biquad
from mixassist.dsp.compressor import CompressorSettings, compress, limit
from mixassist.dsp.fft import power_spectrum
from mixassist.dsp.loudness import integrated_lufs, peak_dbfs


def _rms(x) -> float:
    return math.sqrt(sum(v * v for v in x) / len(x))


def test_fft_peak_bin():
    sr = 8192
    sig = array("d", [math.sin(2 * math.pi * 512 * i / sr) for i in range(sr)])
    freqs, power = power_spectrum(sig, sr, fft_size=1024)
    peak_bin = max(range(len(power)), key=lambda k: power[k])
    assert abs(freqs[peak_bin] - 512.0) < 10.0


def test_lowpass_attenuates_highs():
    sr = 44100
    low = array("d", [math.sin(2 * math.pi * 100 * i / sr) for i in range(sr)])
    high = array("d", [math.sin(2 * math.pi * 2000 * i / sr) for i in range(sr)])
    biquad.low_pass(sr, 300).process_inplace(low)
    biquad.low_pass(sr, 300).process_inplace(high)
    assert _rms(low) > 0.6  # 100 Hz mostly passes
    assert _rms(high) < 0.1  # 2 kHz strongly attenuated


def test_peaking_gain_is_accurate():
    sr = 44100
    t = array("d", [math.sin(2 * math.pi * 1000 * i / sr) for i in range(sr)])
    before = _rms(t)
    biquad.peaking(sr, 1000.0, 6.0, 1.0).process_inplace(t)
    gain_db = 20 * math.log10(_rms(t) / before)
    assert abs(gain_db - 6.0) < 0.3


def test_lufs_calibration():
    sr = 48000
    # 1 kHz sine at -20 dBFS should read ~ -23 LUFS
    sig = array("d", [0.1 * math.sin(2 * math.pi * 1000 * i / sr) for i in range(sr * 2)])
    lufs = integrated_lufs(AudioBuffer([sig], sr))
    assert abs(lufs - (-23.0)) < 1.0


def test_lufs_scales_with_level():
    sr = 48000
    quiet = AudioBuffer(
        [array("d", [0.1 * math.sin(2 * math.pi * 1000 * i / sr) for i in range(sr)])], sr
    )
    loud = AudioBuffer(
        [array("d", [0.5 * math.sin(2 * math.pi * 1000 * i / sr) for i in range(sr)])], sr
    )
    # 0.5/0.1 = ~14 dB louder
    assert abs((integrated_lufs(loud) - integrated_lufs(quiet)) - 13.98) < 0.5


def test_compressor_reduces_dynamics():
    sr = 44100
    buf = AudioBuffer(
        [array("d", [0.8 * math.sin(2 * math.pi * 200 * i / sr) for i in range(sr)])], sr
    )
    gr = compress(buf, CompressorSettings(threshold_db=-12, ratio=4, makeup_db=0))
    assert gr > 3.0


def test_limiter_caps_peak():
    sr = 44100
    buf = AudioBuffer(
        [
            array("d", [1.5 * math.sin(2 * math.pi * 100 * i / sr) for i in range(sr)])
            for _ in range(2)
        ],
        sr,
    )
    limit(buf, ceiling_db=-1.0)
    assert peak_dbfs(buf) <= -1.0 + 0.1
