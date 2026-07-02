"""A small, dependency-free FFT and spectrum helpers.

Iterative radix-2 Cooley-Tukey FFT plus an averaged (Welch-style) magnitude spectrum used
for tonal-balance and feature analysis. Not tuned for speed on huge files; analysis works
on windowed frames rather than whole signals, which keeps it tractable.
"""

from __future__ import annotations

import cmath
import math
from array import array
from collections.abc import Sequence


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def fft(a: list[complex]) -> list[complex]:
    """In-place iterative FFT. ``len(a)`` must be a power of two."""
    n = len(a)
    if n & (n - 1):
        raise ValueError("FFT length must be a power of two")
    # bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        ang = -2.0j * math.pi / length
        wlen = cmath.exp(ang)
        half = length >> 1
        for i in range(0, n, length):
            w = 1.0 + 0.0j
            for k in range(half):
                u = a[i + k]
                v = a[i + k + half] * w
                a[i + k] = u + v
                a[i + k + half] = u - v
                w *= wlen
        length <<= 1
    return a


def _hann(n: int) -> array:
    if n == 1:
        return array("d", [1.0])
    w = array("d")
    for i in range(n):
        w.append(0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1)))
    return w


def power_spectrum(
    samples: Sequence[float],
    sample_rate: int,
    fft_size: int = 4096,
    hop: int | None = None,
    max_frames: int = 240,
) -> tuple[array, array]:
    """Return ``(freqs, power)`` — the averaged one-sided power spectrum.

    ``power`` is linear mean power per bin (windowed, averaged across frames). At most
    ``max_frames`` evenly spaced frames are used to bound cost on long signals.
    """
    n = len(samples)
    fft_size = next_pow2(fft_size)
    if n < fft_size:
        # zero-pad a single frame
        frame_starts = [0]
    else:
        hop = hop or fft_size // 2
        total = 1 + (n - fft_size) // hop
        if total <= max_frames:
            frame_starts = [i * hop for i in range(total)]
        else:
            step = total / max_frames
            frame_starts = [int(k * step) * hop for k in range(max_frames)]

    win = _hann(fft_size)
    win_power = sum(w * w for w in win) or 1.0
    half = fft_size // 2
    acc = array("d", bytes(8 * (half + 1)))
    frames_used = 0

    for start in frame_starts:
        buf: list[complex] = [0j] * fft_size
        for i in range(fft_size):
            idx = start + i
            s = samples[idx] if idx < n else 0.0
            buf[i] = complex(s * win[i], 0.0)
        fft(buf)
        for k in range(half + 1):
            re = buf[k].real
            im = buf[k].imag
            acc[k] += re * re + im * im
        frames_used += 1

    norm = 1.0 / (frames_used * win_power) if frames_used else 0.0
    power = array("d", (v * norm for v in acc))
    freqs = array("d", (k * sample_rate / fft_size for k in range(half + 1)))
    return freqs, power


def band_powers(
    freqs: Sequence[float], power: Sequence[float], edges: Sequence[float]
) -> list[float]:
    """Sum power within each ``[edges[i], edges[i+1])`` band.

    Returns ``len(edges) - 1`` linear power sums.
    """
    n_bands = len(edges) - 1
    out = [0.0] * n_bands
    for f, p in zip(freqs, power, strict=False):
        for b in range(n_bands):
            if edges[b] <= f < edges[b + 1]:
                out[b] += p
                break
    return out
