"""Load/save the full control-surface state as JSON.

A config captures the global controls plus per-track overrides so a mix is fully
reproducible and tweakable. The reference track is stored as a path (resolved by the
caller) rather than embedded audio.

Example::

    {
      "genre": "pop",
      "intensity": 0.6,
      "vocal_prominence": 0.7,
      "tone": 0.1,
      "target_lufs": null,
      "peak_ceiling_db": null,
      "reference": "refs/song.wav",
      "tracks": {
        "Lead Vocal": {"gain_trim_db": 1.0, "pan": 0.0,
                        "eq": [{"kind": "peak", "freq": 5000, "gain_db": 1.5, "q": 1.0}]},
        "Snare":      {"solo": false, "mute": false, "pan": 0.2}
      }
    }
"""

from __future__ import annotations

import json
from pathlib import Path

from mixassist.dsp.eq import EQBand
from mixassist.mixing.engine import MixSettings, TrackOverride


def _override_from_dict(d: dict) -> TrackOverride:
    eq = [
        EQBand(
            kind=b["kind"],
            freq=float(b["freq"]),
            gain_db=float(b.get("gain_db", 0.0)),
            q=float(b.get("q", 0.7071)),
            reason=b.get("reason", "user EQ"),
        )
        for b in d.get("eq", [])
    ]
    return TrackOverride(
        mute=bool(d.get("mute", False)),
        solo=bool(d.get("solo", False)),
        lock=bool(d.get("lock", False)),
        gain_trim_db=float(d.get("gain_trim_db", 0.0)),
        pan=(None if d.get("pan") is None else float(d["pan"])),
        width=(None if d.get("width") is None else float(d["width"])),
        extra_eq=eq,
    )


def _override_to_dict(o: TrackOverride) -> dict:
    d: dict = {}
    if o.mute:
        d["mute"] = True
    if o.solo:
        d["solo"] = True
    if o.lock:
        d["lock"] = True
    if o.gain_trim_db:
        d["gain_trim_db"] = round(o.gain_trim_db, 2)
    if o.pan is not None:
        d["pan"] = round(o.pan, 3)
    if o.width is not None:
        d["width"] = round(o.width, 3)
    if o.extra_eq:
        d["eq"] = [b.as_dict() for b in o.extra_eq]
    return d


def settings_from_dict(data: dict) -> tuple[MixSettings, str | None]:
    """Build (MixSettings, reference_path) from a parsed config dict."""
    overrides = {
        name: _override_from_dict(od or {}) for name, od in (data.get("tracks") or {}).items()
    }
    locked = frozenset(name for name, ov in overrides.items() if ov.lock) | frozenset(
        data.get("locked", [])
    )
    settings = MixSettings(
        genre=data.get("genre", "default"),
        intensity=float(data.get("intensity", 0.5)),
        vocal_prominence=float(data.get("vocal_prominence", 0.5)),
        tone=float(data.get("tone", 0.0)),
        target_lufs=data.get("target_lufs"),
        peak_ceiling_db=data.get("peak_ceiling_db"),
        locked=locked,
        track_overrides=overrides,
        reverb=float(data.get("reverb", 0.0)),
        delay=float(data.get("delay", 0.0)),
        drive=float(data.get("drive", 0.0)),
        sidechain=float(data.get("sidechain", 0.0)),
        master=bool(data.get("master", True)),
    )
    return settings, data.get("reference")


def load_config(path: str) -> tuple[MixSettings, str | None]:
    data = json.loads(Path(path).read_text())
    return settings_from_dict(data)


def settings_to_dict(settings: MixSettings, reference_path: str | None = None) -> dict:
    data: dict = {
        "genre": settings.genre,
        "intensity": settings.intensity,
        "vocal_prominence": settings.vocal_prominence,
        "tone": settings.tone,
        "target_lufs": settings.target_lufs,
        "peak_ceiling_db": settings.peak_ceiling_db,
        "reverb": settings.reverb,
        "delay": settings.delay,
        "drive": settings.drive,
        "sidechain": settings.sidechain,
        "master": settings.master,
    }
    if reference_path:
        data["reference"] = reference_path
    tracks = {name: _override_to_dict(ov) for name, ov in settings.track_overrides.items()}
    tracks = {k: v for k, v in tracks.items() if v}
    if tracks:
        data["tracks"] = tracks
    return data


def save_config(path: str, settings: MixSettings, reference_path: str | None = None) -> None:
    Path(path).write_text(json.dumps(settings_to_dict(settings, reference_path), indent=2))
