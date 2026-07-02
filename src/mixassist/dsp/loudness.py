"""Loudness measurement per ITU-R BS.1770-4 (integrated LUFS) plus peak helpers.

The K-weighting pre-filter + RLB high-pass coefficients are derived for the actual sample
rate (same formulation used by common open-source loudness meters). Integrated loudness
applies the -70 LUFS absolute gate and the -10 LU relative gate over 400 ms blocks with
75% overlap.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Sequence

from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp.biquad import Biquad

_ABS_GATE_LUFS = -70.0
_REL_GATE_LU = -10.0


def _k_weighting(fs: float) -> tuple[Biquad, Biquad]:
    """Return the two K-weighting stages (high-shelf pre-filter, RLB high-pass)."""
    # Stage 1: high-frequency shelving "pre-filter".
    f0 = 1681.974450955533
    g = 3.999843853973347
    q = 0.7071752369554196
    k = math.tan(math.pi * f0 / fs)
    vh = 10.0 ** (g / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    stage1 = Biquad(
        b0=(vh + vb * k / q + k * k) / a0,
        b1=2.0 * (k * k - vh) / a0,
        b2=(vh - vb * k / q + k * k) / a0,
        a1=2.0 * (k * k - 1.0) / a0,
        a2=(1.0 - k / q + k * k) / a0,
    )

    # Stage 2: RLB high-pass.
    f0 = 38.13547087602444
    q = 0.5003270373238773
    k = math.tan(math.pi * f0 / fs)
    denom = 1.0 + k / q + k * k
    stage2 = Biquad(
        b0=1.0,
        b1=-2.0,
        b2=1.0,
        a1=2.0 * (k * k - 1.0) / denom,
        a2=(1.0 - k / q + k * k) / denom,
    )
    return stage1, stage2


# Per-channel loudness weights G_i (mono/stereo = 1.0; surround channels weigh more).
_CHANNEL_GAINS = (1.0, 1.0, 1.0, 1.41, 1.41)


def _block_mean_squares(weighted: list[array], fs: int) -> tuple[list[float], list[float]]:
    """Compute per-block summed mean-square (over channels) and block loudness."""
    block = int(round(0.4 * fs))
    step = int(round(0.1 * fs))
    if block <= 0:
        return [], []
    n = len(weighted[0])
    means: list[float] = []
    loud: list[float] = []
    start = 0
    while start + block <= n:
        z_sum = 0.0
        for c, ch in enumerate(weighted):
            g = _CHANNEL_GAINS[c] if c < len(_CHANNEL_GAINS) else 1.0
            s = 0.0
            for i in range(start, start + block):
                v = ch[i]
                s += v * v
            z_sum += g * (s / block)
        means.append(z_sum)
        loud.append(-0.691 + 10.0 * math.log10(z_sum) if z_sum > 0 else -math.inf)
        start += step
    return means, loud


def _k_weighted_channels(buf: AudioBuffer) -> list[array]:
    """Return each channel passed through the two K-weighting stages."""
    fs = buf.sample_rate
    out: list[array] = []
    for ch in buf.channels:
        s1, s2 = _k_weighting(fs)
        y = s1.process(ch)
        s2.process_inplace(y)
        out.append(y)
    return out


def integrated_lufs(buf: AudioBuffer) -> float:
    """Integrated (gated) loudness in LUFS. Returns -inf for silence."""
    fs = buf.sample_rate
    weighted = _k_weighted_channels(buf)

    means, loud = _block_mean_squares(weighted, fs)
    if not means:
        return -math.inf

    # Absolute gate at -70 LUFS.
    kept = [m for m, lb in zip(means, loud, strict=False) if lb > _ABS_GATE_LUFS]
    if not kept:
        return -math.inf
    ungated = sum(kept) / len(kept)
    rel_thresh = -0.691 + 10.0 * math.log10(ungated) + _REL_GATE_LU

    # Relative gate.
    kept2 = [
        m for m, lb in zip(means, loud, strict=False) if lb > _ABS_GATE_LUFS and lb > rel_thresh
    ]
    if not kept2:
        return -math.inf
    gated = sum(kept2) / len(kept2)
    return -0.691 + 10.0 * math.log10(gated) if gated > 0 else -math.inf


def rms_dbfs(samples: Sequence[float]) -> float:
    n = len(samples)
    if not n:
        return -math.inf
    s = 0.0
    for v in samples:
        s += v * v
    ms = s / n
    return 10.0 * math.log10(ms) if ms > 0 else -math.inf


def peak_dbfs(buf: AudioBuffer) -> float:
    p = buf.peak()
    return 20.0 * math.log10(p) if p > 0 else -math.inf


def lin_to_db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 0 else -math.inf


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def loudness_time_series(
    buf: AudioBuffer, window_s: float = 3.0, hop_s: float = 0.5
) -> list[tuple[float, float]]:
    """Return ``[(time_s, lufs), ...]`` ungated loudness over a sliding window.

    ``window_s`` of 0.4 gives momentary loudness; 3.0 gives short-term (BS.1770). Uses a
    prefix-sum over K-weighted power so cost stays O(n) regardless of hop size.
    """
    fs = buf.sample_rate
    weighted = _k_weighted_channels(buf)
    n = len(weighted[0])
    if n == 0:
        return []

    # Prefix sum of channel-weighted instantaneous power.
    psum = array("d", bytes(8 * (n + 1)))
    gains = [_CHANNEL_GAINS[c] if c < len(_CHANNEL_GAINS) else 1.0 for c in range(len(weighted))]
    acc = 0.0
    for i in range(n):
        p = 0.0
        for c, ch in enumerate(weighted):
            v = ch[i]
            p += gains[c] * v * v
        acc += p
        psum[i + 1] = acc

    win = max(1, int(round(window_s * fs)))
    hop = max(1, int(round(hop_s * fs)))
    series: list[tuple[float, float]] = []

    if win >= n:  # signal shorter than the window: one measurement over all of it
        ms = psum[n] / n
        lufs = -0.691 + 10.0 * math.log10(ms) if ms > 0 else -math.inf
        return [(n / (2 * fs), lufs)]

    start = 0
    while start + win <= n:
        ms = (psum[start + win] - psum[start]) / win
        lufs = -0.691 + 10.0 * math.log10(ms) if ms > 0 else -math.inf
        series.append(((start + win / 2) / fs, lufs))
        start += hop
    return series
