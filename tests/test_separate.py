"""Drum separation wrapper: label mapping + output collection (no ML needed to test)."""

from __future__ import annotations

from mixassist.separate import collect_outputs, friendly_name, separate


def test_friendly_name_maps_drumsep_and_common_labels():
    assert friendly_name("bombo") == "Kick"
    assert friendly_name("redoblante") == "Snare"
    assert friendly_name("platillos") == "Cymbals"
    assert friendly_name("toms") == "Toms"
    assert friendly_name("snare") == "Snare"
    assert friendly_name("HiHat") == "HiHat"
    assert friendly_name("Vocals") == "Vocals"
    assert friendly_name("mystery") == "Mystery"  # unknown -> capitalized passthrough


def test_collect_outputs_flattens_and_renames(tmp_path):
    track = tmp_path / "raw" / "modelo_final" / "loop"
    track.mkdir(parents=True)
    for src in ("bombo", "redoblante", "platillos", "toms"):
        (track / f"{src}.wav").write_bytes(b"RIFF____WAVE")
    out = tmp_path / "out" / "stems"
    created = collect_outputs(tmp_path / "raw", out)
    assert set(created) == {"Kick", "Snare", "Cymbals", "Toms"}
    for name in ("Kick", "Snare", "Cymbals", "Toms"):
        assert (out / f"{name}.wav").exists()


def test_collect_outputs_dedupes_same_name(tmp_path):
    root = tmp_path / "raw"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "a" / "hihat.wav").write_bytes(b"x")
    (root / "b" / "hh.wav").write_bytes(b"y")  # also maps to HiHat
    out = tmp_path / "stems"
    created = collect_outputs(root, out)
    assert len(created) == 2
    assert (out / "HiHat.wav").exists()
    assert (out / "HiHat 2.wav").exists()


def test_separate_reports_missing_input(tmp_path):
    res = separate(str(tmp_path / "does_not_exist.wav"), str(tmp_path / "out"))
    assert res["ok"] is False
    assert "not found" in res["error"]
