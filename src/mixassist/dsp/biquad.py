"""Biquad IIR filters (RBJ audio-EQ cookbook) with Direct Form I processing.

Each :class:`Biquad` carries its own state so it can be applied to a stream in chunks.
Design helpers cover the filter types the EQ and K-weighting stages need.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class Biquad:
    """Normalized biquad coefficients (a0 == 1) plus per-channel state."""

    b0: float
    b1: float
    b2: float
    a1: float
    a2: float
    # Direct Form I state
    _x1: float = 0.0
    _x2: float = 0.0
    _y1: float = 0.0
    _y2: float = 0.0

    def reset(self) -> None:
        self._x1 = self._x2 = self._y1 = self._y2 = 0.0

    def copy(self) -> Biquad:
        return Biquad(self.b0, self.b1, self.b2, self.a1, self.a2)

    def process(self, samples: Iterable[float]) -> array:
        """Filter a sequence, returning a new float64 array. Updates internal state."""
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        x1, x2, y1, y2 = self._x1, self._x2, self._y1, self._y2
        out = array("d")
        ap = out.append
        for x in samples:
            y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2 = x1
            x1 = x
            y2 = y1
            y1 = y
            ap(y)
        self._x1, self._x2, self._y1, self._y2 = x1, x2, y1, y2
        return out

    def process_inplace(self, samples: array) -> None:
        """Filter a mutable array in place."""
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        x1, x2, y1, y2 = self._x1, self._x2, self._y1, self._y2
        for i in range(len(samples)):
            x = samples[i]
            y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2 = x1
            x1 = x
            y2 = y1
            y1 = y
            samples[i] = y
        self._x1, self._x2, self._y1, self._y2 = x1, x2, y1, y2


def _normalize(b0, b1, b2, a0, a1, a2) -> Biquad:
    return Biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def low_pass(fs: float, freq: float, q: float = 0.7071) -> Biquad:
    w0 = 2.0 * math.pi * freq / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    b1 = 1.0 - cw
    b0 = b2 = b1 / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cw
    a2 = 1.0 - alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


def high_pass(fs: float, freq: float, q: float = 0.7071) -> Biquad:
    w0 = 2.0 * math.pi * freq / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    b0 = (1.0 + cw) / 2.0
    b1 = -(1.0 + cw)
    b2 = b0
    a0 = 1.0 + alpha
    a1 = -2.0 * cw
    a2 = 1.0 - alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


def peaking(fs: float, freq: float, gain_db: float, q: float = 1.0) -> Biquad:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / a
    return _normalize(b0, b1, b2, a0, a1, a2)


def low_shelf(fs: float, freq: float, gain_db: float, slope: float = 1.0) -> Biquad:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / 2.0 * math.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)
    two_sqrt_a_alpha = 2.0 * math.sqrt(a) * alpha
    b0 = a * ((a + 1.0) - (a - 1.0) * cw + two_sqrt_a_alpha)
    b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cw)
    b2 = a * ((a + 1.0) - (a - 1.0) * cw - two_sqrt_a_alpha)
    a0 = (a + 1.0) + (a - 1.0) * cw + two_sqrt_a_alpha
    a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cw)
    a2 = (a + 1.0) + (a - 1.0) * cw - two_sqrt_a_alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


def high_shelf(fs: float, freq: float, gain_db: float, slope: float = 1.0) -> Biquad:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / 2.0 * math.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)
    two_sqrt_a_alpha = 2.0 * math.sqrt(a) * alpha
    b0 = a * ((a + 1.0) + (a - 1.0) * cw + two_sqrt_a_alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cw)
    b2 = a * ((a + 1.0) + (a - 1.0) * cw - two_sqrt_a_alpha)
    a0 = (a + 1.0) - (a - 1.0) * cw + two_sqrt_a_alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cw)
    a2 = (a + 1.0) - (a - 1.0) * cw - two_sqrt_a_alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


class BiquadChain:
    """A cascade of biquads applied in series (shares no state between channels)."""

    def __init__(self, stages: list[Biquad] | None = None) -> None:
        self.stages: list[Biquad] = stages or []

    def add(self, stage: Biquad) -> BiquadChain:
        self.stages.append(stage)
        return self

    def reset(self) -> None:
        for s in self.stages:
            s.reset()

    def copy(self) -> BiquadChain:
        return BiquadChain([s.copy() for s in self.stages])

    def process_inplace(self, samples: array) -> None:
        for s in self.stages:
            s.process_inplace(samples)
