"""Gain and stereo-placement utilities."""

from __future__ import annotations

import math
from array import array

from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp.backend import get_backend


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin_to_db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 0 else -math.inf


def pan_to_stereo(buf: AudioBuffer, pan: float, width: float = 1.0) -> AudioBuffer:
    """Place a source in the stereo field.

    ``pan`` in [-1, 1] (equal-power). For a stereo source, ``width`` in [0, 1] collapses
    (0 = mono) or preserves (1 = full) the stereo image before panning the mid.
    """
    pan = max(-1.0, min(1.0, pan))
    angle = (pan + 1.0) * 0.25 * math.pi  # 0..pi/2
    gl = math.cos(angle)
    gr = math.sin(angle)

    if buf.num_channels == 1:
        mono = buf.channels[0]
        left = array("d", (s * gl for s in mono))
        right = array("d", (s * gr for s in mono))
        return AudioBuffer([left, right], buf.sample_rate)

    src = buf.to_stereo()
    l_in, r_in = src.channels[0], src.channels[1]
    n = len(l_in)
    left = array("d", bytes(8 * n))
    right = array("d", bytes(8 * n))
    # Normalize equal-power gains so a centered source stays at unity.
    bl = gl * 1.4142135623730951
    br = gr * 1.4142135623730951
    for i in range(n):
        mid = 0.5 * (l_in[i] + r_in[i])
        side = 0.5 * (l_in[i] - r_in[i]) * width
        left[i] = (mid + side) * bl
        right[i] = (mid - side) * br
    return AudioBuffer([left, right], buf.sample_rate)


def mix_into(dest: AudioBuffer, src: AudioBuffer, gain: float = 1.0) -> None:
    """Sum ``src`` (stereo) into ``dest`` (stereo) in place with ``gain``."""
    backend = get_backend()
    for c in range(2):
        backend.mix_add(dest.channels[c], src.channels[c], gain)
