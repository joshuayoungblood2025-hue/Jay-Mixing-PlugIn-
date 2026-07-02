"""Profile serialization and storage."""

from __future__ import annotations

from mixassist.mixing import profiles
from mixassist.mixing.targets import get_target


def test_target_roundtrip():
    t = get_target("rnb")
    t2 = profiles.target_from_dict(profiles.target_to_dict(t))
    assert t2.name == t.name
    assert t2.bus_lufs == t.bus_lufs
    assert t2.curve_list() == t.curve_list()
    assert t2.role_level_lu == t.role_level_lu


def test_save_load_list(tmp_path):
    t = get_target("pop")
    path = profiles.save_profile(str(tmp_path), t)
    assert profiles.load_profile(path).name == "pop"
    names = [x.name for x in profiles.list_profiles(str(tmp_path))]
    assert "pop" in names


def test_list_missing_dir_is_empty(tmp_path):
    assert profiles.list_profiles(str(tmp_path / "nope")) == []


def test_learned_metadata_roundtrips():
    t = get_target("pop")
    t.target_plr_db = 12.5
    t.target_width = 0.4
    t.learned_from = 7
    t2 = profiles.target_from_dict(profiles.target_to_dict(t))
    assert t2.target_plr_db == 12.5
    assert t2.target_width == 0.4
    assert t2.learned_from == 7
    assert t2.is_learned
