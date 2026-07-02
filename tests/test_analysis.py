"""Classification tests: spectral heuristics and filename hints."""

from __future__ import annotations

from mixassist.analysis.classify import BASS, DRUMS, VOCAL, classify_track
from mixassist.analysis.features import extract_features


def test_spectral_classifies_bass(synth_stems):
    feat = extract_features("trk_a", synth_stems["Bass"])
    cls = classify_track(feat)
    assert cls.role == BASS
    assert cls.method == "spectral"
    assert cls.confidence > 0.5


def test_spectral_classifies_vocal(synth_stems):
    feat = extract_features("trk_b", synth_stems["Lead Vocal"])
    cls = classify_track(feat)
    assert cls.role == VOCAL


def test_spectral_classifies_drums(synth_stems):
    feat = extract_features("trk_c", synth_stems["Hats"])
    cls = classify_track(feat)
    assert cls.role == DRUMS


def test_filename_hint_overrides(synth_stems):
    # spectrally this is bass, but the name says vocal -> filename wins
    feat = extract_features("Lead Vocal 1", synth_stems["Bass"])
    cls = classify_track(feat)
    assert cls.role == VOCAL
    assert cls.method in ("filename", "hybrid")


def test_filename_hint_agrees_is_hybrid(synth_stems):
    feat = extract_features("Bass DI", synth_stems["Bass"])
    cls = classify_track(feat)
    assert cls.role == BASS
    assert cls.method == "hybrid"
    assert cls.confidence >= 0.9
