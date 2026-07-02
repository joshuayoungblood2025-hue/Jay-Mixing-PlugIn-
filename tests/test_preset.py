"""Bus-chain preset export/import and realization."""

from __future__ import annotations

from mixassist.mixing.engine import MixSettings, mix
from mixassist.mixing.targets import get_target
from mixassist.realtime.preset import BusChainPreset


def test_from_target_and_roundtrip(tmp_path):
    preset = BusChainPreset.from_target(get_target("pop"), intensity=0.6, output_gain_db=-1.0)
    assert preset.comp is not None
    assert preset.output_gain_db == -1.0
    d = preset.to_dict()
    assert BusChainPreset.from_dict(d).to_dict() == d
    path = str(tmp_path / "x.chain.json")
    preset.save(path)
    assert BusChainPreset.load(path).to_dict() == d


def test_build_chain_processes(tmp_path):
    preset = BusChainPreset.from_target(get_target("edm"), intensity=0.7)
    chain = preset.build_chain(44100, num_channels=2)
    assert chain is not None
    assert len(chain.eq_chains) == 2


def test_from_bus_plan_snapshots_chain(synth_stems):
    res = mix(synth_stems, MixSettings(genre="pop", intensity=0.6))
    preset = BusChainPreset.from_bus_plan(res)
    assert len(preset.eq) > 0
    assert preset.comp is not None
    assert preset.limiter_ceiling_db == res.bus_plan.peak_ceiling_db
    assert abs(preset.output_gain_db - res.bus_plan.normalize_gain_db) < 1e-9


def test_preset_json_has_plugin_keys(tmp_path):
    # keys the JUCE plugin loader reads must be present
    preset = BusChainPreset.from_target(get_target("rnb"), intensity=0.5)
    d = preset.to_dict()
    assert {"name", "eq", "comp", "limiter_ceiling_db", "output_gain_db"} <= set(d)
    for band in d["eq"]:
        assert {"kind", "freq", "gain_db", "q"} <= set(band)
