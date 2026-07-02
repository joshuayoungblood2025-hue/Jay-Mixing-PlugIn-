"""Learn a mixing target from a corpus of finished reference mixes.

For each reference WAV we measure its master metrics (tonal balance, loudness, dynamics,
stereo width) and aggregate them with a median (robust to outliers) into a target profile.
Because references are finished stereo files (no stems), the per-role level balance and
panning strategy are inherited from a chosen base genre; the *tonal curve*, *loudness*,
*dynamics* and *width* are what get learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mixassist.analysis.metrics import MasterMetrics, compute_master_metrics
from mixassist.analysis.spectrum import BAND_NAMES
from mixassist.audio.io import load_wav
from mixassist.mixing.targets import GenreTarget, get_target


def _median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


@dataclass
class CorpusSummary:
    name: str
    num_references: int
    per_reference: list[MasterMetrics]
    learned_bus_lufs: float
    learned_plr_db: float
    learned_width: float
    learned_curve: dict[str, float]


def analyze_references(paths: list[str]) -> list[MasterMetrics]:
    metrics: list[MasterMetrics] = []
    for p in paths:
        buf = load_wav(p)
        m = compute_master_metrics(buf)
        if math.isfinite(m.integrated_lufs):
            metrics.append(m)
    return metrics


def learn_profile_from_references(
    name: str,
    paths: list[str],
    base_genre: str = "default",
) -> tuple[GenreTarget, CorpusSummary]:
    """Derive a :class:`GenreTarget` from a set of reference mixes."""
    metrics = analyze_references(paths)
    if not metrics:
        raise ValueError("no usable (non-silent) reference tracks were provided")

    base = get_target(base_genre)

    # Learned tonal curve: per-band median of each reference's normalized (average-relative)
    # band levels. This becomes the target "shape" the bus is nudged toward.
    curve: dict[str, float] = {}
    for i, band in enumerate(BAND_NAMES):
        band_vals = []
        for m in metrics:
            shape = m.tonal.normalized_db()
            if i < len(shape):
                band_vals.append(shape[i])
        curve[band] = round(_median(band_vals), 3)

    bus_lufs = round(_median([m.integrated_lufs for m in metrics]), 2)
    plr = round(_median([m.plr_db for m in metrics]), 2)
    width = round(_median([m.width_overall for m in metrics]), 3)

    target = GenreTarget(
        name=name,
        role_level_lu=dict(base.role_level_lu),
        tonal_curve_db=curve,
        bus_lufs=bus_lufs,
        peak_ceiling_db=base.peak_ceiling_db,
        spread=base.spread,
        notes=f"Learned from {len(metrics)} reference mix(es); level balance from '{base_genre}'.",
        target_plr_db=plr,
        target_width=width,
        learned_from=len(metrics),
    )
    summary = CorpusSummary(
        name=name,
        num_references=len(metrics),
        per_reference=metrics,
        learned_bus_lufs=bus_lufs,
        learned_plr_db=plr,
        learned_width=width,
        learned_curve=curve,
    )
    return target, summary
