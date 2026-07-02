"""Suggestions engine behavior."""

from __future__ import annotations

from mixassist.analysis.metrics import compute_master_metrics
from mixassist.mixing.engine import MixSettings, mix
from mixassist.mixing.suggestions import WARN, generate_suggestions


def _mix_and_suggest(stems, settings):
    res = mix(stems, settings)
    metrics = compute_master_metrics(res.master)
    return generate_suggestions(res, metrics)


def test_returns_loudness_note(synth_stems):
    sug = _mix_and_suggest(synth_stems, MixSettings(genre="pop"))
    assert any(s.area == "loudness" for s in sug)


def test_heavy_processing_triggers_limiter_warning(synth_stems):
    # Force hard limiting by targeting an unrealistically hot loudness.
    sug = _mix_and_suggest(synth_stems, MixSettings(genre="edm", intensity=1.0, target_lufs=-6.0))
    assert any(s.area == "limiter" and s.severity == WARN for s in sug)


def test_warnings_sorted_first(synth_stems):
    sug = _mix_and_suggest(synth_stems, MixSettings(genre="edm", intensity=1.0))
    severities = [s.severity for s in sug]
    # once we hit a non-warn, no warn should follow
    seen_non_warn = False
    for sev in severities:
        if sev != WARN:
            seen_non_warn = True
        elif seen_non_warn:
            raise AssertionError("warnings are not grouped first")


def test_all_suggestions_serializable(synth_stems):
    sug = _mix_and_suggest(synth_stems, MixSettings(genre="rnb"))
    for s in sug:
        d = s.as_dict()
        assert set(d) == {"severity", "area", "message"}
