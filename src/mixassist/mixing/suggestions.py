"""Rule-based mix suggestions and explanations.

Turns the finished mix (per-track plans + master metrics + genre target) into a list of
actionable, human-readable notes: tonal-balance deviations, loudness vs. target, dynamics,
stereo/phase issues, over-limiting, and potential midrange masking of the vocal. This is
the "assistant explains itself" layer; Phase 3 can augment it with learned heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass

from mixassist.analysis.classify import VOCAL
from mixassist.analysis.metrics import MasterMetrics
from mixassist.analysis.spectrum import BAND_NAMES
from mixassist.mixing.engine import MixResult

# severity levels
OK = "ok"
INFO = "info"
WARN = "warn"

_ORDER = {WARN: 0, INFO: 1, OK: 2}


@dataclass
class Suggestion:
    severity: str
    area: str
    message: str

    def as_dict(self) -> dict:
        return {"severity": self.severity, "area": self.area, "message": self.message}


def _tonal_notes(metrics: MasterMetrics, target_curve: list[float]) -> list[Suggestion]:
    out: list[Suggestion] = []
    shape = metrics.tonal.normalized_db()
    deviations: list[tuple[float, str, float]] = []
    for i, name in enumerate(BAND_NAMES):
        if i >= len(shape) or i >= len(target_curve):
            break
        delta = shape[i] - target_curve[i]  # +ve = mix has more than target
        deviations.append((abs(delta), name, delta))
    deviations.sort(reverse=True)
    for mag, name, delta in deviations[:2]:
        if mag < 2.5:
            continue
        if delta > 0:
            msg = f"'{name}' band is ~{mag:.1f} dB hotter than target — consider easing it back"
        else:
            msg = f"'{name}' band is ~{mag:.1f} dB below target — consider a gentle lift"
        out.append(Suggestion(WARN if mag > 4.0 else INFO, "tonal balance", msg))
    return out


def _loudness_notes(result: MixResult, metrics: MasterMetrics) -> list[Suggestion]:
    out: list[Suggestion] = []
    target = result.bus_plan.target_lufs
    diff = metrics.integrated_lufs - target
    if abs(diff) >= 1.5:
        direction = "louder than" if diff > 0 else "quieter than"
        out.append(
            Suggestion(
                INFO,
                "loudness",
                f"Integrated loudness is {abs(diff):.1f} LU {direction} the "
                f"{target:.1f} LUFS target (now {metrics.integrated_lufs:.1f} LUFS).",
            )
        )
    else:
        out.append(Suggestion(OK, "loudness", f"On target at {metrics.integrated_lufs:.1f} LUFS."))

    plr = metrics.plr_db
    if plr < 8.0:
        out.append(
            Suggestion(
                WARN,
                "dynamics",
                f"Low peak-to-loudness ratio ({plr:.1f} dB): the mix is heavily "
                "compressed/limited. Lower intensity or the loudness target for more punch.",
            )
        )
    elif plr > 18.0:
        out.append(
            Suggestion(
                INFO,
                "dynamics",
                f"Very dynamic ({plr:.1f} dB PLR): great for detail, but it may feel "
                "quiet next to loud streaming masters.",
            )
        )

    gr = result.bus_plan.limiter_gr_db
    if gr > 6.0:
        out.append(
            Suggestion(
                WARN,
                "limiter",
                f"Safety limiter is pulling {gr:.1f} dB — reduce upstream level or "
                "intensity to avoid pumping/distortion.",
            )
        )
    return out


def _stereo_notes(metrics: MasterMetrics) -> list[Suggestion]:
    out: list[Suggestion] = []
    corr = metrics.stereo_correlation
    if corr < 0.0:
        out.append(
            Suggestion(
                WARN,
                "phase",
                f"Negative stereo correlation ({corr:.2f}): likely phase problems that will "
                "weaken or cancel in mono. Check polarity/timing of stereo/duplicated tracks.",
            )
        )
    elif corr < 0.2:
        out.append(
            Suggestion(
                INFO,
                "phase",
                f"Low stereo correlation ({corr:.2f}); verify mono compatibility.",
            )
        )
    elif corr > 0.99:
        out.append(
            Suggestion(
                INFO,
                "stereo",
                "Mix is essentially mono — there is room to widen instruments/FX if desired.",
            )
        )

    if metrics.width_low > 0.6:
        out.append(
            Suggestion(
                WARN,
                "low end",
                f"Wide low end (width {metrics.width_low:.2f}): keep sub/bass mono below "
                "~120 Hz for a tighter, more translatable bottom.",
            )
        )
    return out


def _masking_notes(result: MixResult) -> list[Suggestion]:
    out: list[Suggestion] = []
    plans = result.track_plans
    vocals = [p for p in plans if p.classification.role == VOCAL and not p.features.silence]
    if not vocals:
        return out
    vocal = max(vocals, key=lambda p: p.out_lufs if p.out_lufs != float("-inf") else -120)
    v_level = vocal.out_lufs if vocal.out_lufs != float("-inf") else -120.0

    for p in plans:
        if p is vocal or p.features.silence:
            continue
        if p.classification.role == VOCAL:
            continue
        # midrange-heavy source at a similar level competes with the vocal
        if (
            p.features.mid_ratio > 0.45
            and p.out_lufs != float("-inf")
            and p.out_lufs >= v_level - 3.0
        ):
            out.append(
                Suggestion(
                    INFO,
                    "masking",
                    f"'{p.name}' has strong midrange near the vocal's level — a small dip "
                    "around 2-4 kHz on it (or a lift on the vocal) can improve clarity.",
                )
            )
    return out[:3]


def generate_suggestions(result: MixResult, metrics: MasterMetrics) -> list[Suggestion]:
    """Produce a prioritized list of suggestions (warnings first)."""
    notes: list[Suggestion] = []
    notes += _loudness_notes(result, metrics)
    notes += _tonal_notes(metrics, result.target.curve_list())
    notes += _stereo_notes(metrics)
    notes += _masking_notes(result)
    notes.sort(key=lambda s: _ORDER.get(s.severity, 3))
    return notes
