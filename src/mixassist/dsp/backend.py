"""Pluggable numeric backend for hot-path array math.

The whole DSP layer is written in pure Python so it runs anywhere (this sandbox has no
numpy and no way to install it). But the per-sample loops are the obvious bottleneck, so
the vectorizable operations are funneled through a small backend interface. If numpy is
importable, :func:`get_backend` returns a vectorized backend; otherwise it falls back to
the pure-Python one. Nothing else in the codebase needs to change to gain the speed-up.

This is the honest "numpy-ready" seam promised for Phase 4: the abstraction and fallback
are real and tested; the numpy path activates automatically wherever numpy exists.
"""

from __future__ import annotations

from array import array
from typing import Protocol


class Backend(Protocol):
    name: str

    def apply_gain(self, samples: array, gain: float) -> None:
        """Multiply ``samples`` in place by a scalar gain."""
        ...

    def mix_add(self, dest: array, src: array, gain: float) -> None:
        """Accumulate ``dest[i] += src[i] * gain`` in place (over the shorter length)."""
        ...

    def peak(self, samples: array) -> float:
        """Return the maximum absolute value."""
        ...


class PurePythonBackend:
    name = "pure-python"

    def apply_gain(self, samples: array, gain: float) -> None:
        if gain == 1.0:
            return
        for i in range(len(samples)):
            samples[i] *= gain

    def mix_add(self, dest: array, src: array, gain: float) -> None:
        n = min(len(dest), len(src))
        for i in range(n):
            dest[i] += src[i] * gain

    def peak(self, samples: array) -> float:
        p = 0.0
        for s in samples:
            a = s if s >= 0.0 else -s
            if a > p:
                p = a
        return p


class NumpyBackend:
    """Vectorized backend used automatically when numpy is available.

    Operates on the same :class:`array.array` objects by wrapping them in numpy views, so
    callers are unchanged. Never imported unless numpy is present.
    """

    name = "numpy"

    def __init__(self, np) -> None:  # np injected by get_backend
        self._np = np

    def apply_gain(self, samples: array, gain: float) -> None:
        if gain == 1.0:
            return
        view = self._np.frombuffer(samples, dtype="float64")
        view *= gain

    def mix_add(self, dest: array, src: array, gain: float) -> None:
        n = min(len(dest), len(src))
        d = self._np.frombuffer(dest, dtype="float64")
        s = self._np.frombuffer(src, dtype="float64")
        d[:n] += s[:n] * gain

    def peak(self, samples: array) -> float:
        if len(samples) == 0:
            return 0.0
        view = self._np.frombuffer(samples, dtype="float64")
        return float(self._np.max(self._np.abs(view)))


_backend: Backend | None = None


def get_backend() -> Backend:
    """Return the active backend, selecting numpy if importable, else pure Python."""
    global _backend
    if _backend is None:
        _backend = _select_backend()
    return _backend


def _select_backend() -> Backend:
    try:
        import numpy as np  # noqa: PLC0415 (deliberate optional import)

        return NumpyBackend(np)
    except ImportError:
        return PurePythonBackend()


def set_backend(backend: Backend | None) -> None:
    """Override the active backend (mainly for tests). ``None`` re-enables auto-select."""
    global _backend
    _backend = backend
