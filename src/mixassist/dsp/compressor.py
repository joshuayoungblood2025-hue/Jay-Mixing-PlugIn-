"""Feed-forward dynamics processing: a peak/RMS compressor and a lookahead-free limiter.

Both operate on linear samples. The compressor uses a decoupled peak detector with
attack/release ballistics and a soft knee, computing gain reduction in the log domain. A
linked stereo detector keeps the image stable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mixassist.audio.buffer import AudioBuffer


@dataclass
class CompressorSettings:
    threshold_db: float = -18.0
    ratio: float = 3.0
    attack_ms: float = 15.0
    release_ms: float = 120.0
    knee_db: float = 6.0
    makeup_db: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "threshold_db": round(self.threshold_db, 2),
            "ratio": round(self.ratio, 2),
            "attack_ms": round(self.attack_ms, 1),
            "release_ms": round(self.release_ms, 1),
            "knee_db": round(self.knee_db, 1),
            "makeup_db": round(self.makeup_db, 2),
            "reason": self.reason,
        }


def _coef(time_ms: float, fs: float) -> float:
    if time_ms <= 0:
        return 0.0
    return math.exp(-1.0 / (fs * time_ms / 1000.0))


def _gain_curve(level_db: float, thr: float, ratio: float, knee: float) -> float:
    """Return output level (dB) for an input level using a soft-knee curve."""
    if knee > 0 and (2.0 * (level_db - thr)) < -knee:
        return level_db
    if knee > 0 and abs(2.0 * (level_db - thr)) <= knee:
        # quadratic interpolation across the knee
        x = level_db - thr + knee / 2.0
        return level_db + (1.0 / ratio - 1.0) * (x * x) / (2.0 * knee)
    return thr + (level_db - thr) / ratio


def compress(buf: AudioBuffer, s: CompressorSettings) -> float:
    """Apply compression in place. Returns the maximum gain reduction (dB) observed."""
    fs = buf.sample_rate
    atk = _coef(s.attack_ms, fs)
    rel = _coef(s.release_ms, fs)
    makeup = 10.0 ** (s.makeup_db / 20.0)
    channels = buf.channels
    nch = len(channels)
    n = buf.num_frames

    env = 0.0  # linked peak envelope (linear)
    max_gr_db = 0.0
    for i in range(n):
        # linked detector: max abs across channels
        peak = 0.0
        for c in range(nch):
            a = channels[c][i]
            if a < 0.0:
                a = -a
            if a > peak:
                peak = a
        # decoupled peak follower
        if peak > env:
            env = atk * env + (1.0 - atk) * peak
        else:
            env = rel * env + (1.0 - rel) * peak

        level_db = 20.0 * math.log10(env) if env > 1e-12 else -120.0
        out_db = _gain_curve(level_db, s.threshold_db, s.ratio, s.knee_db)
        gr_db = out_db - level_db  # <= 0
        if -gr_db > max_gr_db:
            max_gr_db = -gr_db
        g = (10.0 ** (gr_db / 20.0)) * makeup
        for c in range(nch):
            channels[c][i] *= g
    return max_gr_db


def limit(buf: AudioBuffer, ceiling_db: float = -1.0, release_ms: float = 50.0) -> float:
    """Simple feedback brickwall-ish limiter (no lookahead). Returns max GR in dB.

    Adequate as a safety/pre-master stage; a true-peak lookahead limiter is a later
    upgrade. Pairs well with leaving headroom upstream.
    """
    fs = buf.sample_rate
    ceiling = 10.0 ** (ceiling_db / 20.0)
    rel = _coef(release_ms, fs)
    channels = buf.channels
    nch = len(channels)
    n = buf.num_frames
    env = 0.0
    max_gr_db = 0.0
    for i in range(n):
        peak = 0.0
        for c in range(nch):
            a = channels[c][i]
            if a < 0.0:
                a = -a
            if a > peak:
                peak = a
        target_gain = ceiling / peak if peak > ceiling else 1.0
        if target_gain < env or env == 0.0:
            env = target_gain  # instant attack
        else:
            env = rel * env + (1.0 - rel) * target_gain
            if env > 1.0:
                env = 1.0
        if env < 1.0:
            gr = -20.0 * math.log10(env)
            if gr > max_gr_db:
                max_gr_db = gr
        for c in range(nch):
            channels[c][i] *= env
    return max_gr_db
