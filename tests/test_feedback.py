"""Preference/feedback model behavior."""

from __future__ import annotations

from mixassist.learn.feedback import PreferenceModel, personalize_settings, record_feedback
from mixassist.mixing.engine import MixSettings


def test_no_history_is_noop():
    adj, notes = personalize_settings(MixSettings(genre="pop", intensity=0.5), PreferenceModel([]))
    assert adj.intensity == 0.5
    assert "No highly-rated" in notes[0]


def test_personalize_moves_toward_liked(tmp_path):
    p = str(tmp_path / "prefs.json")
    record_feedback(p, MixSettings(genre="pop", intensity=0.9, vocal_prominence=0.9, tone=0.5), 5)
    record_feedback(p, MixSettings(genre="pop", intensity=0.85, vocal_prominence=0.8, tone=0.4), 5)
    model = PreferenceModel.load(p)
    base = MixSettings(genre="pop", intensity=0.5, vocal_prominence=0.5, tone=0.0)
    adj, _ = personalize_settings(base, model)
    assert adj.intensity > 0.5
    assert adj.vocal_prominence > 0.5
    assert adj.tone > 0.0


def test_low_ratings_are_not_liked(tmp_path):
    p = str(tmp_path / "prefs.json")
    record_feedback(p, MixSettings(genre="pop", intensity=0.9), 2)
    model = PreferenceModel.load(p)
    adj, notes = personalize_settings(MixSettings(genre="pop", intensity=0.5), model)
    assert adj.intensity == 0.5
    assert "No highly-rated" in notes[0]


def test_feedback_persists(tmp_path):
    p = str(tmp_path / "prefs.json")
    record_feedback(p, MixSettings(genre="rnb", intensity=0.6), 4)
    model = PreferenceModel.load(p)
    assert len(model.examples) == 1
    assert model.examples[0].genre == "rnb"
    assert model.examples[0].rating == 4.0


def test_personalize_preserves_other_settings(tmp_path):
    p = str(tmp_path / "prefs.json")
    record_feedback(p, MixSettings(genre="pop", intensity=0.9), 5)
    model = PreferenceModel.load(p)
    base = MixSettings(genre="pop", intensity=0.5, locked=frozenset({"Bass"}), target_lufs=-16.0)
    adj, _ = personalize_settings(base, model)
    assert adj.locked == frozenset({"Bass"})
    assert adj.target_lufs == -16.0
