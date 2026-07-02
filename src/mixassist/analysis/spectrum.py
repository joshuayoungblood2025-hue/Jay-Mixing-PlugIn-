"""Tonal-balance analysis over a canonical set of frequency bands."""

from __future__ import annotations

import math
from dataclasses import dataclass

from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp.fft import band_powers, power_spectrum

# Canonical analysis bands: (name, low_hz, high_hz)
BANDS: list[tuple[str, float, float]] = [
    ("sub", 20.0, 60.0),
    ("low_bass", 60.0, 120.0),
    ("bass", 120.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("high_mid", 2000.0, 4000.0),
    ("presence", 4000.0, 8000.0),
    ("brilliance", 8000.0, 12000.0),
    ("air", 12000.0, 20000.0),
]

BAND_NAMES = [b[0] for b in BANDS]
_EDGES = [BANDS[0][1]] + [b[2] for b in BANDS]


@dataclass
class TonalBalance:
    """Per-band energy expressed in dB relative to total (sums approx via power)."""

    band_names: list[str]
    band_db: list[float]  # dB level per band (10*log10 of band power)
    total_power: float

    def as_dict(self) -> dict:
        return {
            "bands": {n: round(v, 2) for n, v in zip(self.band_names, self.band_db, strict=False)},
            "total_power_db": round(
                10.0 * math.log10(self.total_power) if self.total_power > 0 else -120.0, 2
            ),
        }

    def normalized_db(self) -> list[float]:
        """Band levels relative to the average band level (shape, not absolute)."""
        finite = [v for v in self.band_db if v > -120.0]
        avg = sum(finite) / len(finite) if finite else 0.0
        return [v - avg for v in self.band_db]


def tonal_balance(buf: AudioBuffer, fft_size: int = 4096) -> TonalBalance:
    mono = buf.mono()
    freqs, power = power_spectrum(mono, buf.sample_rate, fft_size=fft_size)
    bp = band_powers(freqs, power, _EDGES)
    total = sum(bp)
    band_db = [10.0 * math.log10(p) if p > 0 else -120.0 for p in bp]
    return TonalBalance(BAND_NAMES, band_db, total)
