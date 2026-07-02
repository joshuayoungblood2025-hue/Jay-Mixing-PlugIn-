"""Genre-aware mixing targets.

These encode an opinionated but transparent starting point for each genre: relative level
balance between roles (in LU, vocal-anchored), a gentle tonal-balance target curve for the
bus, per-role dynamics defaults, panning strategy, and a bus loudness/peak target that
leaves headroom for mastering. Phase 3 can learn these from reference corpora.
"""

from __future__ import annotations

from dataclasses import dataclass

from mixassist.analysis.classify import BASS, DRUMS, FX, INSTRUMENT, VOCAL
from mixassist.analysis.spectrum import BAND_NAMES


@dataclass
class GenreTarget:
    name: str
    # Relative loudness balance between roles, in LU. Vocal is the anchor (0.0).
    role_level_lu: dict[str, float]
    # Desired bus tonal shape: dB offset per canonical band relative to flat.
    tonal_curve_db: dict[str, float]
    # Integrated loudness target for the finished mix bus (pre/light-master), LUFS.
    bus_lufs: float
    # Peak ceiling for the safety limiter, dBFS.
    peak_ceiling_db: float
    # Stereo spread amount for non-centered roles, 0..1.
    spread: float = 0.7
    notes: str = ""
    # --- Optional learned-profile metadata (Phase 3) ---
    # Target peak-to-loudness ratio (dB) learned from a corpus; None = not specified.
    target_plr_db: float | None = None
    # Target overall stereo width learned from a corpus; None = not specified.
    target_width: float | None = None
    # Number of reference tracks this profile was learned from (0 = hand-authored).
    learned_from: int = 0

    def curve_list(self) -> list[float]:
        return [self.tonal_curve_db.get(n, 0.0) for n in BAND_NAMES]

    @property
    def is_learned(self) -> bool:
        return self.learned_from > 0


def _curve(**kw: float) -> dict[str, float]:
    return {n: kw.get(n, 0.0) for n in BAND_NAMES}


_GENRES: dict[str, GenreTarget] = {
    "pop": GenreTarget(
        name="pop",
        role_level_lu={VOCAL: 0.0, DRUMS: -1.5, BASS: -2.5, INSTRUMENT: -4.5, FX: -7.0},
        tonal_curve_db=_curve(low_bass=1.0, bass=0.5, low_mid=-1.0, presence=1.5, air=2.0),
        bus_lufs=-14.0,
        peak_ceiling_db=-1.0,
        spread=0.7,
        notes="Vocal-forward, bright and polished.",
    ),
    "rock": GenreTarget(
        name="rock",
        role_level_lu={VOCAL: 0.0, DRUMS: -1.0, BASS: -3.0, INSTRUMENT: -2.0, FX: -8.0},
        tonal_curve_db=_curve(bass=1.0, low_mid=0.5, high_mid=1.0, presence=1.0),
        bus_lufs=-13.0,
        peak_ceiling_db=-1.0,
        spread=0.8,
        notes="Guitars up front, punchy midrange.",
    ),
    "hiphop": GenreTarget(
        name="hiphop",
        role_level_lu={VOCAL: 0.0, DRUMS: -1.0, BASS: -0.5, INSTRUMENT: -5.0, FX: -8.0},
        tonal_curve_db=_curve(sub=2.0, low_bass=2.0, bass=1.0, presence=1.5, air=1.5),
        bus_lufs=-13.0,
        peak_ceiling_db=-1.0,
        spread=0.6,
        notes="Big low end (808/sub), crisp vocals.",
    ),
    "rnb": GenreTarget(
        name="rnb",
        role_level_lu={VOCAL: 0.0, DRUMS: -3.0, BASS: -2.0, INSTRUMENT: -4.0, FX: -7.0},
        tonal_curve_db=_curve(low_bass=1.5, low_mid=-1.5, presence=1.0, air=2.5),
        bus_lufs=-15.0,
        peak_ceiling_db=-1.0,
        spread=0.7,
        notes="Smooth, silky top end, intimate vocals.",
    ),
    "edm": GenreTarget(
        name="edm",
        role_level_lu={VOCAL: -1.0, DRUMS: 0.0, BASS: -1.0, INSTRUMENT: -3.0, FX: -5.0},
        tonal_curve_db=_curve(sub=2.0, low_bass=1.5, presence=1.5, brilliance=2.0, air=2.5),
        bus_lufs=-12.0,
        peak_ceiling_db=-1.0,
        spread=0.85,
        notes="Loud, wide, energetic; strong kick/bass and bright synths.",
    ),
    "film": GenreTarget(
        name="film",
        role_level_lu={VOCAL: 0.0, DRUMS: -4.0, BASS: -3.0, INSTRUMENT: -2.0, FX: -3.0},
        tonal_curve_db=_curve(sub=1.0, low_mid=0.5, presence=0.5),
        bus_lufs=-18.0,
        peak_ceiling_db=-1.0,
        spread=0.9,
        notes="Dynamic and wide; preserves transients and space.",
    ),
    "acoustic": GenreTarget(
        name="acoustic",
        role_level_lu={VOCAL: 0.0, DRUMS: -4.0, BASS: -4.0, INSTRUMENT: -2.0, FX: -10.0},
        tonal_curve_db=_curve(low_mid=-0.5, high_mid=0.5, presence=1.0, air=1.5),
        bus_lufs=-16.0,
        peak_ceiling_db=-1.0,
        spread=0.6,
        notes="Natural, dynamic, minimal processing.",
    ),
    "default": GenreTarget(
        name="default",
        role_level_lu={VOCAL: 0.0, DRUMS: -2.0, BASS: -3.0, INSTRUMENT: -4.0, FX: -8.0},
        tonal_curve_db=_curve(presence=1.0, air=1.0),
        bus_lufs=-15.0,
        peak_ceiling_db=-1.0,
        spread=0.7,
        notes="Balanced general-purpose target.",
    ),
}

# Base per-stem loudness anchor (LUFS). Only relative differences matter because the bus
# is normalized to the genre target afterward, but a sane anchor keeps gains reasonable.
STEM_ANCHOR_LUFS = -20.0


def available_genres() -> list[str]:
    return sorted(_GENRES.keys())


def get_target(genre: str) -> GenreTarget:
    return _GENRES.get(genre.lower(), _GENRES["default"])
