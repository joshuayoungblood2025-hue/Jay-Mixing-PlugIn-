"""Build machine-readable (JSON) and human-readable mix reports from a MixResult."""

from __future__ import annotations

import math

from mixassist.analysis.metrics import MasterMetrics
from mixassist.analysis.spectrum import BAND_NAMES
from mixassist.mixing.engine import MixResult, MixSettings
from mixassist.mixing.suggestions import Suggestion


def _fmt_db(x: float) -> str:
    if not math.isfinite(x):
        return "-inf"
    return f"{x:+.1f} dB"


def _fmt_lufs(x: float) -> str:
    return f"{x:.1f} LUFS" if math.isfinite(x) else "silent"


def _bar(value: float, lo: float, hi: float, width: int = 32) -> str:
    if not math.isfinite(value):
        return "─" * width
    frac = (max(lo, min(hi, value)) - lo) / (hi - lo) if hi > lo else 0.0
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def build_report(
    result: MixResult,
    settings: MixSettings,
    metrics: MasterMetrics | None = None,
    suggestions: list[Suggestion] | None = None,
) -> dict:
    s = settings.clamped()
    tracks = []
    for p in result.track_plans:
        tracks.append(
            {
                "name": p.name,
                "classification": p.classification.as_dict(),
                "features": p.features.summary(),
                "locked": p.locked,
                "muted": p.muted,
                "input_stage_db": round(p.input_stage_db, 2),
                "gain_db": round(p.gain_db, 2),
                "gain_trim_db": round(p.gain_trim_db, 2),
                "pan": round(p.pan, 2),
                "width": round(p.width, 2),
                "in_lufs": round(p.in_lufs, 2) if math.isfinite(p.in_lufs) else None,
                "out_lufs": round(p.out_lufs, 2) if math.isfinite(p.out_lufs) else None,
                "eq": [b.as_dict() for b in p.eq_bands],
                "compression": p.comp.as_dict() if p.comp else None,
                "compression_gr_db": round(p.comp_gr_db, 2),
                "sidechain_gr_db": round(p.sidechain_gr_db, 2),
            }
        )

    bp = result.bus_plan
    master = {
        "target_lufs": bp.target_lufs,
        "final_lufs": round(bp.final_lufs, 2) if math.isfinite(bp.final_lufs) else None,
        "final_peak_dbfs": round(bp.final_peak_dbfs, 2)
        if math.isfinite(bp.final_peak_dbfs)
        else None,
        "peak_ceiling_db": bp.peak_ceiling_db,
        "normalize_gain_db": round(bp.normalize_gain_db, 2),
        "glue_gr_db": round(bp.glue_gr_db, 2),
        "limiter_gr_db": round(bp.limiter_gr_db, 2),
        "creative_fx": {
            "reverb": round(bp.reverb_amount, 2),
            "delay": round(bp.delay_amount, 2),
            "drive": round(bp.drive_amount, 2),
        },
        "tonal_correction_eq": [b.as_dict() for b in bp.tonal_eq],
    }

    return {
        "settings": {
            "genre": s.genre,
            "intensity": s.intensity,
            "vocal_prominence": s.vocal_prominence,
            "tone": s.tone,
            "reference_used": s.reference is not None,
            "locked_tracks": sorted(s.locked),
        },
        "target_notes": result.target.notes,
        "tracks": tracks,
        "master": master,
        "metrics": metrics.as_dict() if metrics is not None else None,
        "suggestions": [x.as_dict() for x in suggestions] if suggestions is not None else None,
    }


def render_text(
    result: MixResult,
    settings: MixSettings,
    metrics: MasterMetrics | None = None,
    suggestions: list[Suggestion] | None = None,
) -> str:
    s = settings.clamped()
    bp = result.bus_plan
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add("  AI MIXING ASSISTANT — MIX REPORT")
    add("=" * 72)
    add(f"Genre target : {result.target.name}  — {result.target.notes}")
    tone_word = "neutral"
    if s.tone > 0.02:
        tone_word = f"clarity ({s.tone:+.2f})"
    elif s.tone < -0.02:
        tone_word = f"warmth ({s.tone:+.2f})"
    add(
        f"Controls     : intensity {s.intensity:.2f} | "
        f"vocal prominence {s.vocal_prominence:.2f} | tone {tone_word}"
    )
    if s.reference is not None:
        add("Reference    : tonal balance matched to supplied reference track")
    if s.locked:
        add(f"Locked       : {', '.join(sorted(s.locked))}")
    add("")

    add("-" * 72)
    add("  TRACKS")
    add("-" * 72)
    for p in result.track_plans:
        c = p.classification
        add(
            f"* {p.name}  ->  {c.role.upper()} / {c.subtype}  "
            f"(confidence {c.confidence:.2f}, {c.method})"
        )
        if p.locked:
            add("    [locked] passed through unprocessed")
            continue
        if p.input_stage_db != 0.0:
            add(f"    staged: {_fmt_db(p.input_stage_db)} to ~-18 dBFS RMS (before any plugins)")
        add(
            f"    level : in {_fmt_lufs(p.in_lufs)} -> gain {_fmt_db(p.gain_db)} "
            f"(pre-gain {_fmt_lufs(p.out_lufs)}) | pan {p.pan:+.2f} | width {p.width:.2f}"
        )
        if p.eq_bands:
            add("    EQ    :")
            for b in p.eq_bands:
                reason = f"  — {b.reason}" if b.reason else ""
                add(f"        - {b.describe()}{reason}")
        if p.comp:
            add(
                f"    comp  : thr {p.comp.threshold_db:.0f} dB, ratio {p.comp.ratio:.1f}:1, "
                f"atk {p.comp.attack_ms:.0f}ms, rel {p.comp.release_ms:.0f}ms "
                f"(GR {p.comp_gr_db:.1f} dB) — {p.comp.reason}"
            )
        if p.sidechain_gr_db > 0.0:
            add(f"    sidechain: ducked up to {p.sidechain_gr_db:.1f} dB by the kick")
        add("")

    add("-" * 72)
    add("  MASTER BUS" if s.master else "  MIX BUS  (no mastering)")
    add("-" * 72)
    if bp.tonal_eq:
        label = "reference match" if s.reference is not None else "genre tonal target"
        add(f"tonal correction ({label}):")
        for b in bp.tonal_eq:
            add(f"    - {b.describe()}")
    if bp.reverb_amount or bp.delay_amount or bp.drive_amount:
        add(
            f"creative fx   : reverb {bp.reverb_amount:.2f} | delay {bp.delay_amount:.2f} | "
            f"drive {bp.drive_amount:.2f}"
        )
    if s.master:
        add(f"glue comp     : GR {bp.glue_gr_db:.1f} dB")
        add(f"normalize     : {_fmt_db(bp.normalize_gain_db)} to hit {bp.target_lufs:.1f} LUFS")
        add(f"limiter       : ceiling {bp.peak_ceiling_db:.1f} dBFS (GR {bp.limiter_gr_db:.1f} dB)")
        add("")
        add(
            f"FINAL         : {_fmt_lufs(bp.final_lufs)} integrated, "
            f"peak {_fmt_db(bp.final_peak_dbfs)}"
        )
        headroom = (
            bp.peak_ceiling_db - bp.final_peak_dbfs if math.isfinite(bp.final_peak_dbfs) else 0.0
        )
        add(f"                {abs(headroom):.1f} dB from ceiling — ready for mastering")
    else:
        add("mastering     : skipped (balance / gain-stage / EQ / compress / pan only)")
        add(f"headroom trim : {_fmt_db(bp.normalize_gain_db)} (peak set to ~ -6 dBFS)")
        add("")
        add(
            f"FINAL         : peak {_fmt_db(bp.final_peak_dbfs)} — balanced mix, "
            "master & add FX in your DAW"
        )
    add("=" * 72)

    if metrics is not None:
        add("")
        add("-" * 72)
        add("  METERS")
        add("-" * 72)
        meters = [
            ("Integrated", metrics.integrated_lufs, -30.0, 0.0, "LUFS"),
            ("Short-term ", metrics.short_term_max_lufs, -30.0, 0.0, "LUFS"),
            ("Momentary  ", metrics.momentary_max_lufs, -30.0, 0.0, "LUFS"),
            ("Sample peak", metrics.peak_dbfs, -30.0, 0.0, "dBFS"),
        ]
        for label, val, lo, hi, unit in meters:
            shown = _fmt_lufs(val) if unit == "LUFS" else _fmt_db(val)
            add(f"  {label}  [{_bar(val, lo, hi)}] {shown}")
        add(f"  PLR          {metrics.plr_db:.1f} dB   |   crest {metrics.crest_db:.1f} dB")
        add(
            f"  Correlation  {metrics.stereo_correlation:+.2f}   |   "
            f"width L/M/H {metrics.width_low:.2f}/{metrics.width_mid:.2f}/{metrics.width_high:.2f}"
        )
        add("")
        add("  Frequency balance (dB vs. average):")
        shape = metrics.tonal.normalized_db()
        for i, name in enumerate(BAND_NAMES):
            v = shape[i] if i < len(shape) else 0.0
            # centered bar: 16 chars each side, 0 at center
            half = 14
            n = int(round(max(-9.0, min(9.0, v)) / 9.0 * half))
            if n >= 0:
                barstr = " " * half + "|" + "█" * n + " " * (half - n)
            else:
                barstr = " " * (half + n) + "█" * (-n) + "|" + " " * half
            add(f"    {name:11s} {v:+5.1f} {barstr}")

    if suggestions:
        add("")
        add("-" * 72)
        add("  SUGGESTIONS")
        add("-" * 72)
        for x in suggestions:
            add(f"  [{x.severity.upper():4s}] {x.area}: {x.message}")

    return "\n".join(lines)
