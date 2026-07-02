"""Kick-hit detection and side-chain trigger generation.

Rather than relying on (artifact-prone) source separation to isolate a kick, we detect
*when* the kick hits by following low-frequency energy, then synthesize a clean
exponential-decay trigger at those moments. That trigger is an ideal, artifact-free
side-chain key for ducking the bass — even from a full, un-separated drum loop.
"""

from __future__ import annotations

import math
from array import array

from mixassist.dsp.biquad import low_pass


def detect_kick_onsets(
    mono,
    sample_rate: int,
    low_hz: float = 150.0,
    min_interval_ms: float = 60.0,
    threshold_ratio: float = 0.22,
) -> list[int]:
    """Return sample indices where kick hits occur.

    Works on a low-passed copy (so hats/snares don't trigger it), builds a fast-attack /
    slow-release energy envelope, and marks threshold crossings with hysteresis and a
    minimum spacing so a single hit isn't counted twice.
    """
    n = len(mono)
    if n == 0:
        return []
    low = array("d", mono)
    low_pass(sample_rate, low_hz).process_inplace(low)

    rel = math.exp(-1.0 / (sample_rate * 0.03))  # 30 ms release
    env = 0.0
    env_buf = array("d", bytes(8 * n))
    peak = 0.0
    for i in range(n):
        r = low[i]
        if r < 0.0:
            r = -r
        env = r if r > env else rel * env + (1.0 - rel) * r
        env_buf[i] = env
        if env > peak:
            peak = env
    if peak < 1e-6:
        return []

    thresh = peak * threshold_ratio
    release_level = thresh * 0.5
    min_gap = int(min_interval_ms * 0.001 * sample_rate)
    hits: list[int] = []
    armed = True
    last = -min_gap
    for i in range(n):
        e = env_buf[i]
        if armed and e > thresh and (i - last) >= min_gap:
            hits.append(i)
            last = i
            armed = False
        elif e < release_level:
            armed = True
    return hits


def build_trigger(hits: list[int], n: int, sample_rate: int, decay_ms: float = 180.0) -> array:
    """Build a clean side-chain key: an exponential-decay pulse at each detected hit."""
    key = array("d", bytes(8 * n))
    if not hits or n == 0:
        return key
    decay = max(1.0, sample_rate * decay_ms * 0.001)
    span = min(n, int(decay * 5))
    for h in hits:
        if h >= n:
            continue
        limit = min(span, n - h)
        for j in range(limit):
            v = math.exp(-j / decay)
            if v > key[h + j]:
                key[h + j] = v
    return key
