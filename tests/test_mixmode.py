"""Mix-prep mode: master=False produces a balanced, un-mastered mix with headroom."""

from __future__ import annotations

from mixassist.mixing.engine import MixSettings, mix
from mixassist.mixing.targets import auto_controls


def test_no_master_leaves_headroom_and_skips_mastering(synth_stems):
    res = mix(synth_stems, MixSettings(genre="pop", master=False))
    bp = res.bus_plan
    # no mastering chain ran
    assert bp.glue is None
    assert bp.glue_gr_db == 0.0
    assert bp.limiter_gr_db == 0.0
    assert bp.tonal_eq == []
    # ~6 dB of headroom left for mastering in a DAW
    assert abs(bp.final_peak_dbfs - (-6.0)) < 0.2


def test_master_default_still_masters(synth_stems):
    res = mix(synth_stems, MixSettings(genre="pop"))
    assert res.bus_plan.glue is not None
    assert res.bus_plan.final_peak_dbfs <= res.bus_plan.peak_ceiling_db + 0.1


def test_no_master_still_balances_and_processes(synth_stems):
    # tracks are still classified, gain-staged, EQ'd, compressed, panned
    res = mix(synth_stems, MixSettings(genre="pop", master=False))
    for p in res.track_plans:
        assert p.classification.role
        if not p.locked and not p.features.silence:
            assert p.eq_bands  # per-track EQ applied
    # a panned instrument should not be dead-center
    pad = next((p for p in res.track_plans if p.name == "Pad"), None)
    assert pad is not None


def test_auto_controls_cover_all_genres():
    for g in ("pop", "rock", "hiphop", "rnb", "edm", "film", "acoustic", "default", "unknown"):
        c = auto_controls(g)
        assert {"intensity", "vocal", "tone", "reverb", "delay", "drive", "sidechain"} <= set(c)


def test_auto_settings_builder():
    s = MixSettings.auto("hiphop")
    assert s.genre == "hiphop"
    assert s.sidechain > 0.0  # hiphop auto enables ducking
    s2 = MixSettings.auto("hiphop", master=False, reverb=0.0)
    assert s2.master is False
    assert s2.reverb == 0.0
