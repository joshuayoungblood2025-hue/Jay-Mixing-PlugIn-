"""Master-bus metrics for feedback and visualization.

Collects the numbers a mixing dashboard needs: integrated / momentary / short-term
loudness, sample peak, crest factor, peak-to-loudness ratio, stereo correlation, and
per-band stereo width. All derived from the finished stereo bus.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field

from mixassist.analysis.spectrum import TonalBalance, tonal_balance
from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp import biquad
from mixassist.dsp.loudness import (
    integrated_lufs,
    loudness_time_series,
    peak_dbfs,
    rms_dbfs,
)


def _mid_side(buf: AudioBuffer) -> tuple[array, array]:
    st = buf.to_stereo()
    left, right = st.channels[0], st.channels[1]
    n = min(len(left), len(right))
    mid = array("d", bytes(8 * n))
    side = array("d", bytes(8 * n))
    for i in range(n):
        mid[i] = 0.5 * (left[i] + right[i])
        side[i] = 0.5 * (left[i] - right[i])
    return mid, side


def _energy(samples: array) -> float:
    return sum(v * v for v in samples)


def _band_copy(samples: array, fs: int, kind: str) -> array:
    out = array("d", samples)
    if kind == "low":
        biquad.low_pass(fs, 250.0).process_inplace(out)
    elif kind == "high":
        biquad.high_pass(fs, 4000.0).process_inplace(out)
    else:  # mid band 250..4000
        biquad.high_pass(fs, 250.0).process_inplace(out)
        biquad.low_pass(fs, 4000.0).process_inplace(out)
    return out


def _width(mid: array, side: array, fs: int, kind: str) -> float:
    m = _energy(_band_copy(mid, fs, kind))
    s = _energy(_band_copy(side, fs, kind))
    if m <= 1e-12:
        return 1.0 if s > 1e-12 else 0.0
    return math.sqrt(s / m)


def _correlation(buf: AudioBuffer) -> float:
    if buf.num_channels < 2:
        return 1.0
    left, right = buf.channels[0], buf.channels[1]
    n = min(len(left), len(right))
    if n == 0:
        return 1.0
    sll = srr = slr = 0.0
    for i in range(n):
        a = left[i]
        b = right[i]
        sll += a * a
        srr += b * b
        slr += a * b
    denom = math.sqrt(sll * srr)
    if denom <= 1e-12:
        return 1.0
    return max(-1.0, min(1.0, slr / denom))


@dataclass
class MasterMetrics:
    duration_s: float
    integrated_lufs: float
    momentary_max_lufs: float
    short_term_max_lufs: float
    peak_dbfs: float
    rms_dbfs: float
    crest_db: float
    plr_db: float  # peak-to-loudness ratio (headroom vs perceived loudness)
    stereo_correlation: float
    width_overall: float
    width_low: float
    width_mid: float
    width_high: float
    tonal: TonalBalance
    short_term_series: list[tuple[float, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        def r(x: float) -> float | None:
            return round(x, 2) if math.isfinite(x) else None

        return {
            "duration_s": round(self.duration_s, 2),
            "integrated_lufs": r(self.integrated_lufs),
            "momentary_max_lufs": r(self.momentary_max_lufs),
            "short_term_max_lufs": r(self.short_term_max_lufs),
            "peak_dbfs": r(self.peak_dbfs),
            "rms_dbfs": r(self.rms_dbfs),
            "crest_db": r(self.crest_db),
            "plr_db": r(self.plr_db),
            "stereo_correlation": round(self.stereo_correlation, 3),
            "width": {
                "overall": round(self.width_overall, 3),
                "low": round(self.width_low, 3),
                "mid": round(self.width_mid, 3),
                "high": round(self.width_high, 3),
            },
            "tonal_balance": self.tonal.as_dict(),
        }


def compute_master_metrics(buf: AudioBuffer) -> MasterMetrics:
    mono = buf.mono()
    integ = integrated_lufs(buf)
    peak = peak_dbfs(buf)
    rms = rms_dbfs(mono)
    crest = (peak - rms) if (math.isfinite(peak) and math.isfinite(rms)) else 0.0
    plr = (peak - integ) if (math.isfinite(peak) and math.isfinite(integ)) else 0.0

    momentary = loudness_time_series(buf, window_s=0.4, hop_s=0.1)
    short_term = loudness_time_series(buf, window_s=3.0, hop_s=0.5)
    mom_max = max((v for _, v in momentary if math.isfinite(v)), default=float("-inf"))
    st_max = max((v for _, v in short_term if math.isfinite(v)), default=float("-inf"))

    mid, side = _mid_side(buf)
    fs = buf.sample_rate
    mid_e = _energy(mid)
    width_overall = math.sqrt(_energy(side) / mid_e) if mid_e > 1e-12 else 0.0

    return MasterMetrics(
        duration_s=buf.duration_seconds,
        integrated_lufs=integ,
        momentary_max_lufs=mom_max,
        short_term_max_lufs=st_max,
        peak_dbfs=peak,
        rms_dbfs=rms,
        crest_db=crest,
        plr_db=plr,
        stereo_correlation=_correlation(buf),
        width_overall=width_overall,
        width_low=_width(mid, side, fs, "low"),
        width_mid=_width(mid, side, fs, "mid"),
        width_high=_width(mid, side, fs, "high"),
        tonal=tonal_balance(buf),
        short_term_series=short_term,
    )
