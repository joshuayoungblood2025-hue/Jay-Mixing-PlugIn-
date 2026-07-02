"""Analog-style saturation via tanh waveshaping.

Adds harmonic content and gentle compression-like "glue"/warmth. ``drive`` sets how hard
the signal is pushed into the nonlinearity; the output is normalized so level stays roughly
constant, and a dry/wet ``mix`` keeps it musical. A small even-harmonic ``bias`` option
gives a warmer, more "tube-like" character.
"""

from __future__ import annotations

import math

from mixassist.audio.buffer import AudioBuffer


def saturate(buf: AudioBuffer, drive: float = 0.3, mix: float = 1.0, bias: float = 0.0) -> None:
    """Apply tanh saturation to every channel in place.

    ``drive`` 0..1 maps to a push of roughly 1x..5x into the curve. ``bias`` (0..~0.3) adds
    even harmonics for warmth. ``mix`` blends dry/processed.
    """
    if drive <= 0.0 or mix <= 0.0:
        return
    d = 1.0 + drive * 4.0
    norm = math.tanh(d)  # keep unity-ish output level
    inv_norm = 1.0 / norm if norm != 0 else 1.0
    b = bias * 0.5
    dry = 1.0 - mix
    for ch in buf.channels:
        for i in range(len(ch)):
            x = ch[i]
            wet = (math.tanh(x * d + b) - math.tanh(b)) * inv_norm
            ch[i] = x * dry + wet * mix


def saturate_samples(samples, drive: float = 0.3, mix: float = 1.0) -> None:
    """Same curve, applied to a raw sample array in place."""
    if drive <= 0.0 or mix <= 0.0:
        return
    d = 1.0 + drive * 4.0
    inv_norm = 1.0 / math.tanh(d)
    dry = 1.0 - mix
    for i in range(len(samples)):
        x = samples[i]
        samples[i] = x * dry + (math.tanh(x * d) * inv_norm) * mix
