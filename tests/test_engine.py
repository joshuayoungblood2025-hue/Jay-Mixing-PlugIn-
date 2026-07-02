"""End-to-end engine tests: loudness/peak targets, structure, controls, reference."""

from __future__ import annotations

import math

from mixassist.analysis.spectrum import tonal_balance
from mixassist.mixing.engine import MixSettings, mix
from mixassist.mixing.report import build_report, render_text


def test_mix_hits_peak_ceiling_and_near_target(synth_stems):
    res = mix(synth_stems, MixSettings(genre="pop", intensity=0.6, vocal_prominence=0.6))
    bp = res.bus_plan
    assert res.master.num_channels == 2
    # peak must not exceed the ceiling
    assert bp.final_peak_dbfs <= bp.peak_ceiling_db + 0.1
    # loudness lands within a couple LU of the target (limiter pulls it down a touch)
    assert abs(bp.final_lufs - bp.target_lufs) < 2.5


def test_all_stems_processed_and_planned(synth_stems):
    res = mix(synth_stems, MixSettings(genre="rock"))
    assert set(res.stems.keys()) == set(synth_stems.keys())
    assert len(res.track_plans) == len(synth_stems)
    for p in res.track_plans:
        assert p.classification.role
        assert res.stems[p.name].num_channels == 2


def test_locked_track_is_untouched(synth_stems):
    res = mix(synth_stems, MixSettings(genre="pop", locked=frozenset({"Bass"})))
    bass_plan = next(p for p in res.track_plans if p.name == "Bass")
    assert bass_plan.locked
    assert bass_plan.gain_db == 0.0
    assert bass_plan.eq_bands == []
    assert bass_plan.comp is None


def test_vocal_prominence_raises_vocal_gain(synth_stems):
    low = mix(synth_stems, MixSettings(genre="pop", vocal_prominence=0.2))
    high = mix(synth_stems, MixSettings(genre="pop", vocal_prominence=0.9))
    g_low = next(p.gain_db for p in low.track_plans if p.name == "Lead Vocal")
    g_high = next(p.gain_db for p in high.track_plans if p.name == "Lead Vocal")
    assert g_high > g_low


def test_reference_matching_produces_correction(synth_stems):
    # Use the vocal stem as a (bright, midrange) reference.
    ref = tonal_balance(synth_stems["Lead Vocal"])
    res = mix(synth_stems, MixSettings(genre="pop", reference=ref))
    assert len(res.bus_plan.tonal_eq) > 0


def test_report_is_serializable_and_textual(synth_stems):
    settings = MixSettings(genre="hiphop", intensity=0.7, tone=-0.3)
    res = mix(synth_stems, settings)
    report = build_report(res, settings)
    assert report["settings"]["genre"] == "hiphop"
    assert report["master"]["final_lufs"] is not None
    assert len(report["tracks"]) == len(synth_stems)
    text = render_text(res, settings)
    assert "MIX REPORT" in text
    assert "MASTER BUS" in text


def test_invalid_genre_falls_back_to_default(synth_stems):
    res = mix(synth_stems, MixSettings(genre="polka"))
    # default target LUFS is -15
    assert math.isclose(res.bus_plan.target_lufs, -15.0)
