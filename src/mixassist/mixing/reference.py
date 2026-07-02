"""Reference-track tonal matching and corrective bus EQ.

Compares the mix bus tonal balance against either a reference track's balance or the genre
target curve, and produces a small set of gentle peaking filters that nudge the mix toward
the target. Moves are capped and scaled so matching stays musical rather than surgical.
"""

from __future__ import annotations

import math

from mixassist.analysis.spectrum import BANDS, TonalBalance
from mixassist.dsp.eq import EQBand


def _band_center(lo: float, hi: float) -> float:
    return math.sqrt(lo * hi)


def _band_q(lo: float, hi: float) -> float:
    center = _band_center(lo, hi)
    bw = hi - lo
    return max(0.4, min(2.0, center / bw)) if bw > 0 else 1.0


def build_corrective_eq(
    current: TonalBalance,
    target_shape: list[float],
    max_db: float = 3.5,
    strength: float = 0.6,
) -> list[EQBand]:
    """Return peaking bands nudging ``current`` toward ``target_shape`` (relative dB)."""
    cur = current.normalized_db()
    bands: list[EQBand] = []
    for i, (name, lo, hi) in enumerate(BANDS):
        if i >= len(target_shape) or i >= len(cur):
            break
        delta = (target_shape[i] - cur[i]) * strength
        delta = max(-max_db, min(max_db, delta))
        if abs(delta) < 0.4:
            continue  # ignore trivial moves
        center = _band_center(lo, hi)
        bands.append(
            EQBand(
                kind="peak",
                freq=center,
                gain_db=delta,
                q=_band_q(lo, hi),
                reason=f"match '{name}' band toward target ({delta:+.1f} dB)",
            )
        )
    return bands


def reference_shape(ref: TonalBalance) -> list[float]:
    """Normalized (average-relative) band levels of a reference track."""
    return ref.normalized_db()
