"""Serialize, store, and load mixing target profiles.

A "profile" is just a :class:`GenreTarget` — whether hand-authored or learned from a
reference corpus (Phase 3). Profiles live as JSON files in a directory so learned styles
and custom presets can be saved, shared, and re-used across sessions.
"""

from __future__ import annotations

import json
from pathlib import Path

from mixassist.analysis.spectrum import BAND_NAMES
from mixassist.mixing.targets import GenreTarget

PROFILE_SUFFIX = ".profile.json"


def target_to_dict(t: GenreTarget) -> dict:
    return {
        "name": t.name,
        "role_level_lu": dict(t.role_level_lu),
        "tonal_curve_db": {n: round(t.tonal_curve_db.get(n, 0.0), 3) for n in BAND_NAMES},
        "bus_lufs": round(t.bus_lufs, 2),
        "peak_ceiling_db": round(t.peak_ceiling_db, 2),
        "spread": round(t.spread, 3),
        "notes": t.notes,
        "target_plr_db": (round(t.target_plr_db, 2) if t.target_plr_db is not None else None),
        "target_width": (round(t.target_width, 3) if t.target_width is not None else None),
        "learned_from": t.learned_from,
    }


def target_from_dict(d: dict) -> GenreTarget:
    return GenreTarget(
        name=d["name"],
        role_level_lu={k: float(v) for k, v in d.get("role_level_lu", {}).items()},
        tonal_curve_db={n: float(d.get("tonal_curve_db", {}).get(n, 0.0)) for n in BAND_NAMES},
        bus_lufs=float(d["bus_lufs"]),
        peak_ceiling_db=float(d.get("peak_ceiling_db", -1.0)),
        spread=float(d.get("spread", 0.7)),
        notes=d.get("notes", ""),
        target_plr_db=(None if d.get("target_plr_db") is None else float(d["target_plr_db"])),
        target_width=(None if d.get("target_width") is None else float(d["target_width"])),
        learned_from=int(d.get("learned_from", 0)),
    )


def save_profile(directory: str, target: GenreTarget) -> str:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{target.name}{PROFILE_SUFFIX}"
    path.write_text(json.dumps(target_to_dict(target), indent=2))
    return str(path)


def load_profile(path: str) -> GenreTarget:
    return target_from_dict(json.loads(Path(path).read_text()))


def list_profiles(directory: str) -> list[GenreTarget]:
    d = Path(directory)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob(f"*{PROFILE_SUFFIX}")):
        try:
            out.append(load_profile(str(p)))
        except (ValueError, KeyError):
            continue
    return out
