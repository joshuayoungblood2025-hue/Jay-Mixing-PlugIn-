"""Per-stem feature extraction used for classification and mixing decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from mixassist.analysis.spectrum import TonalBalance, tonal_balance
from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp.fft import power_spectrum
from mixassist.dsp.loudness import integrated_lufs, peak_dbfs, rms_dbfs


@dataclass
class Features:
    name: str
    duration_s: float
    num_channels: int
    lufs: float
    peak_dbfs: float
    rms_dbfs: float
    crest_db: float  # peak - rms, transient/dynamic indicator
    centroid_hz: float
    zero_cross_rate: float  # crossings per second
    low_ratio: float  # fraction of power below 250 Hz
    mid_ratio: float  # fraction 250 Hz .. 4 kHz
    high_ratio: float  # fraction above 4 kHz
    stereo_width: float  # 0 (mono) .. 1+ (wide)
    silence: bool
    tonal: TonalBalance

    def summary(self) -> dict:
        return {
            "lufs": round(self.lufs, 2) if math.isfinite(self.lufs) else None,
            "peak_dbfs": round(self.peak_dbfs, 2) if math.isfinite(self.peak_dbfs) else None,
            "crest_db": round(self.crest_db, 2) if math.isfinite(self.crest_db) else None,
            "centroid_hz": round(self.centroid_hz, 1),
            "zero_cross_rate": round(self.zero_cross_rate, 1),
            "low_ratio": round(self.low_ratio, 3),
            "mid_ratio": round(self.mid_ratio, 3),
            "high_ratio": round(self.high_ratio, 3),
            "stereo_width": round(self.stereo_width, 3),
        }


def _zero_cross_rate(samples, sample_rate: int) -> float:
    n = len(samples)
    if n < 2:
        return 0.0
    crossings = 0
    prev = samples[0]
    for i in range(1, n):
        cur = samples[i]
        if (prev >= 0.0) != (cur >= 0.0):
            crossings += 1
        prev = cur
    return crossings * sample_rate / n


def _stereo_width(buf: AudioBuffer) -> float:
    if buf.num_channels < 2:
        return 0.0
    left, right = buf.channels[0], buf.channels[1]
    n = min(len(left), len(right))
    if n == 0:
        return 0.0
    mid_e = 0.0
    side_e = 0.0
    for i in range(n):
        m = 0.5 * (left[i] + right[i])
        s = 0.5 * (left[i] - right[i])
        mid_e += m * m
        side_e += s * s
    if mid_e <= 1e-12:
        return 1.0 if side_e > 1e-12 else 0.0
    return math.sqrt(side_e / mid_e)


def extract_features(name: str, buf: AudioBuffer) -> Features:
    mono = buf.mono()
    sr = buf.sample_rate
    peak = peak_dbfs(buf)
    rms = rms_dbfs(mono)
    crest = (peak - rms) if (math.isfinite(peak) and math.isfinite(rms)) else 0.0
    silence = not math.isfinite(rms) or rms < -80.0

    freqs, power = power_spectrum(mono, sr, fft_size=4096)
    total = sum(power) or 1e-12
    centroid = sum(f * p for f, p in zip(freqs, power, strict=False)) / total
    low = sum(p for f, p in zip(freqs, power, strict=False) if f < 250.0) / total
    mid = sum(p for f, p in zip(freqs, power, strict=False) if 250.0 <= f < 4000.0) / total
    high = sum(p for f, p in zip(freqs, power, strict=False) if f >= 4000.0) / total

    return Features(
        name=name,
        duration_s=buf.duration_seconds,
        num_channels=buf.num_channels,
        lufs=integrated_lufs(buf),
        peak_dbfs=peak,
        rms_dbfs=rms,
        crest_db=crest,
        centroid_hz=centroid,
        zero_cross_rate=_zero_cross_rate(mono, sr),
        low_ratio=low,
        mid_ratio=mid,
        high_ratio=high,
        stereo_width=_stereo_width(buf),
        silence=silence,
        tonal=tonal_balance(buf),
    )
