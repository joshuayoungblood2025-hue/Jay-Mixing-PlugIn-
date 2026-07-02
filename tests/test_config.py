"""Control-surface JSON config round-trip."""

from __future__ import annotations

from mixassist.dsp.eq import EQBand
from mixassist.mixing import config as config_mod
from mixassist.mixing.engine import MixSettings, TrackOverride


def test_config_roundtrip(tmp_path):
    settings = MixSettings(
        genre="rnb",
        intensity=0.7,
        vocal_prominence=0.65,
        tone=-0.2,
        track_overrides={
            "Lead Vocal": TrackOverride(
                gain_trim_db=1.5,
                pan=0.2,
                extra_eq=[EQBand("peak", 5000.0, 2.0, 1.0, "user air")],
            ),
            "Kick": TrackOverride(mute=True),
            "Gtr": TrackOverride(solo=True, width=0.5),
        },
    )
    path = tmp_path / "mix.json"
    config_mod.save_config(str(path), settings, reference_path="refs/song.wav")

    loaded, ref = config_mod.load_config(str(path))
    assert ref == "refs/song.wav"
    assert loaded.genre == "rnb"
    assert abs(loaded.intensity - 0.7) < 1e-9
    assert abs(loaded.tone - (-0.2)) < 1e-9

    vov = loaded.override_for("Lead Vocal")
    assert abs(vov.gain_trim_db - 1.5) < 1e-9
    assert vov.pan == 0.2
    assert len(vov.extra_eq) == 1
    assert vov.extra_eq[0].freq == 5000.0

    assert loaded.override_for("Kick").mute is True
    assert loaded.override_for("Gtr").solo is True
    assert loaded.override_for("Gtr").width == 0.5


def test_lock_from_config_populates_locked_set(tmp_path):
    data = {"genre": "pop", "tracks": {"Bass": {"lock": True}}}
    settings, _ = config_mod.settings_from_dict(data)
    assert "Bass" in settings.locked


def test_empty_overrides_omitted(tmp_path):
    settings = MixSettings(genre="pop")
    d = config_mod.settings_to_dict(settings)
    assert "tracks" not in d
