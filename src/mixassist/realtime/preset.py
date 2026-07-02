"""Bus-chain presets: the bridge between the offline engine and the real-time chain/plugin.

A :class:`BusChainPreset` is a portable, sample-rate-independent description of the fixed
bus processing path (EQ + glue compression + output gain + limiter). It can be:

* snapshotted from a finished mix (:meth:`from_bus_plan`) so the exact chain the assistant
  designed can be re-applied in real time or in the plugin, or
* built generically from a genre/learned target (:meth:`from_target`).

The same JSON is read by the Python :class:`BusChain` and by the JUCE plugin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mixassist.analysis.spectrum import BANDS
from mixassist.dsp.compressor import CompressorSettings
from mixassist.dsp.eq import EQBand
from mixassist.realtime.blocks import BusChain

PRESET_VERSION = 1
PRESET_SUFFIX = ".chain.json"


def _comp_from_dict(d: dict) -> CompressorSettings:
    return CompressorSettings(
        threshold_db=float(d.get("threshold_db", -18.0)),
        ratio=float(d.get("ratio", 2.0)),
        attack_ms=float(d.get("attack_ms", 30.0)),
        release_ms=float(d.get("release_ms", 200.0)),
        knee_db=float(d.get("knee_db", 6.0)),
        makeup_db=float(d.get("makeup_db", 0.0)),
        reason=d.get("reason", ""),
    )


@dataclass
class BusChainPreset:
    name: str
    eq: list[EQBand] = field(default_factory=list)
    comp: CompressorSettings | None = None
    limiter_ceiling_db: float = -1.0
    output_gain_db: float = 0.0
    target_lufs: float = -14.0
    notes: str = ""

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_bus_plan(cls, result, name: str | None = None) -> BusChainPreset:
        """Snapshot the exact bus chain designed for a finished mix (see engine.MixResult)."""
        bp = result.bus_plan
        return cls(
            name=name or f"{result.target.name}_master",
            eq=list(bp.tonal_eq),
            comp=bp.glue,
            limiter_ceiling_db=bp.peak_ceiling_db,
            output_gain_db=bp.normalize_gain_db,
            target_lufs=bp.target_lufs,
            notes=f"Snapshot of the '{result.target.name}' master bus chain.",
        )

    @classmethod
    def from_target(
        cls,
        target,
        intensity: float = 0.5,
        output_gain_db: float = 0.0,
    ) -> BusChainPreset:
        """Build a generic static bus chain from a genre/learned target's tonal curve."""
        bands: list[EQBand] = []
        for i, (band_name, lo, hi) in enumerate(BANDS):
            curve = target.curve_list()
            if i >= len(curve):
                break
            gain = curve[i] * (0.5 + 0.5 * intensity)
            if abs(gain) < 0.3:
                continue
            center = (lo * hi) ** 0.5
            q = max(0.4, min(2.0, center / (hi - lo))) if hi > lo else 1.0
            bands.append(EQBand("peak", center, gain_db=gain, q=q, reason=f"tone: {band_name}"))
        comp = CompressorSettings(
            threshold_db=-18.0,
            ratio=1.5 + 0.8 * intensity,
            attack_ms=30.0,
            release_ms=200.0,
            knee_db=6.0,
            reason="bus glue",
        )
        return cls(
            name=getattr(target, "name", "custom"),
            eq=bands,
            comp=comp,
            limiter_ceiling_db=getattr(target, "peak_ceiling_db", -1.0),
            output_gain_db=output_gain_db,
            target_lufs=getattr(target, "bus_lufs", -14.0),
            notes=f"Generic static chain from target '{getattr(target, 'name', 'custom')}'.",
        )

    # -- (de)serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": PRESET_VERSION,
            "name": self.name,
            "eq": [b.as_dict() for b in self.eq],
            "comp": self.comp.as_dict() if self.comp else None,
            "limiter_ceiling_db": round(self.limiter_ceiling_db, 3),
            "output_gain_db": round(self.output_gain_db, 3),
            "target_lufs": round(self.target_lufs, 2),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BusChainPreset:
        eq = [
            EQBand(
                kind=b["kind"],
                freq=float(b["freq"]),
                gain_db=float(b.get("gain_db", 0.0)),
                q=float(b.get("q", 0.7071)),
                reason=b.get("reason", ""),
            )
            for b in d.get("eq", [])
        ]
        comp = _comp_from_dict(d["comp"]) if d.get("comp") else None
        return cls(
            name=d.get("name", "custom"),
            eq=eq,
            comp=comp,
            limiter_ceiling_db=float(d.get("limiter_ceiling_db", -1.0)),
            output_gain_db=float(d.get("output_gain_db", 0.0)),
            target_lufs=float(d.get("target_lufs", -14.0)),
            notes=d.get("notes", ""),
        )

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return str(p)

    @classmethod
    def load(cls, path: str) -> BusChainPreset:
        return cls.from_dict(json.loads(Path(path).read_text()))

    # -- realization ---------------------------------------------------------

    def build_chain(self, sample_rate: int, num_channels: int = 2) -> BusChain:
        """Instantiate a stateful :class:`BusChain` for the given sample rate."""
        return BusChain(
            fs=sample_rate,
            eq_bands=self.eq,
            comp=self.comp,
            limiter_ceiling_db=self.limiter_ceiling_db,
            output_gain_db=self.output_gain_db,
            num_channels=num_channels,
        )
