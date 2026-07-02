"""Corpus learning and engine target override."""

from __future__ import annotations

import math

import pytest

from mixassist.audio.io import save_wav
from mixassist.learn.corpus import analyze_references, learn_profile_from_references
from mixassist.mixing.engine import MixSettings, mix


def _render_refs(synth_stems, tmp_path, n=2):
    paths = []
    for i in range(n):
        res = mix(synth_stems, MixSettings(genre="edm", intensity=0.8))
        p = str(tmp_path / f"ref{i}.wav")
        save_wav(p, res.master)
        paths.append(p)
    return paths


def test_learn_profile_from_references(synth_stems, tmp_path):
    paths = _render_refs(synth_stems, tmp_path)
    target, summary = learn_profile_from_references("mystyle", paths, base_genre="edm")
    assert target.learned_from == 2
    assert target.is_learned
    assert len(target.tonal_curve_db) == 9
    assert math.isfinite(target.bus_lufs)
    assert target.target_plr_db is not None
    assert summary.num_references == 2
    # level balance inherited from base genre
    assert target.role_level_lu  # non-empty


def test_learn_requires_usable_references(tmp_path):
    with pytest.raises(ValueError):
        learn_profile_from_references("x", [], base_genre="pop")


def test_analyze_skips_silent(tmp_path, synth_stems):
    # a silent file should be dropped
    from array import array

    from mixassist.audio.buffer import AudioBuffer

    silent = AudioBuffer([array("d", [0.0] * 44100)], 44100)
    sp = str(tmp_path / "silent.wav")
    save_wav(sp, silent)
    good = _render_refs(synth_stems, tmp_path, n=1)[0]
    metrics = analyze_references([sp, good])
    assert len(metrics) == 1


def test_target_override_drives_bus_target(synth_stems, tmp_path):
    paths = _render_refs(synth_stems, tmp_path)
    target, _ = learn_profile_from_references("mystyle", paths, base_genre="edm")
    res = mix(synth_stems, MixSettings(genre="pop", target_override=target))
    assert res.target.name == "mystyle"
    assert math.isclose(res.bus_plan.target_lufs, target.bus_lufs, abs_tol=0.01)
