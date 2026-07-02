"""A persisted preference model that personalizes control settings from user ratings.

Each time a user rates a mix (1-5 stars), we store the control settings that produced it.
:func:`personalize_settings` then blends the base controls toward the settings of the
user's highly-rated mixes, weighted by rating and recency. It is a transparent, explainable
"learn my taste over time" loop — not a black box — and it degrades gracefully to a no-op
when there is little or no feedback yet.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mixassist.mixing.engine import MixSettings

# Controls we personalize (all continuous, easy to blend).
_LIKED_THRESHOLD = 4.0  # ratings >= this count as "liked"
_RECENCY_DECAY = 0.95  # per-step weight decay from newest to oldest
_MAX_BLEND = 0.6  # never move more than 60% toward learned preference
_BLEND_PER_LIKE = 0.15  # confidence gained per liked example


@dataclass
class PreferenceExample:
    genre: str
    intensity: float
    vocal_prominence: float
    tone: float
    rating: float
    integrated_lufs: float | None = None
    plr_db: float | None = None
    timestamp: float = field(default_factory=lambda: time.time())

    def as_dict(self) -> dict:
        return asdict(self)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class PreferenceModel:
    """A collection of rated examples plus the logic to personalize new settings."""

    def __init__(self, examples: list[PreferenceExample] | None = None) -> None:
        self.examples: list[PreferenceExample] = examples or []

    # -- persistence ---------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> PreferenceModel:
        p = Path(path)
        if not p.exists():
            return cls([])
        data = json.loads(p.read_text())
        return cls([PreferenceExample(**e) for e in data.get("examples", [])])

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"examples": [e.as_dict() for e in self.examples]}, indent=2)
        )

    # -- updates -------------------------------------------------------------

    def add(self, example: PreferenceExample) -> None:
        self.examples.append(example)

    # -- inference -----------------------------------------------------------

    def _relevant(self, genre: str) -> list[PreferenceExample]:
        same = [e for e in self.examples if e.genre == genre]
        return same if same else list(self.examples)

    def preferred_controls(self, genre: str) -> tuple[dict[str, float], float] | None:
        """Return (preferred control means, total weight) from liked examples, or None."""
        rel = self._relevant(genre)
        liked = [e for e in rel if e.rating >= _LIKED_THRESHOLD]
        if not liked:
            return None
        n = len(liked)
        acc = {"intensity": 0.0, "vocal_prominence": 0.0, "tone": 0.0}
        wsum = 0.0
        for i, e in enumerate(liked):  # i=0 oldest
            recency = _RECENCY_DECAY ** (n - 1 - i)
            w = max(0.0, e.rating - 3.0) * recency
            acc["intensity"] += e.intensity * w
            acc["vocal_prominence"] += e.vocal_prominence * w
            acc["tone"] += e.tone * w
            wsum += w
        if wsum <= 0:
            return None
        means = {k: v / wsum for k, v in acc.items()}
        return means, float(len(liked))


def record_feedback(
    path: str,
    settings: MixSettings,
    rating: float,
    integrated_lufs: float | None = None,
    plr_db: float | None = None,
) -> PreferenceModel:
    """Append a rating for the given settings and persist the model."""
    model = PreferenceModel.load(path)
    s = settings.clamped()
    model.add(
        PreferenceExample(
            genre=s.genre,
            intensity=s.intensity,
            vocal_prominence=s.vocal_prominence,
            tone=s.tone,
            rating=float(rating),
            integrated_lufs=integrated_lufs,
            plr_db=plr_db,
        )
    )
    model.save(path)
    return model


def personalize_settings(
    base: MixSettings, model: PreferenceModel
) -> tuple[MixSettings, list[str]]:
    """Blend ``base`` toward the user's learned preferences. Returns (settings, notes)."""
    pref = model.preferred_controls(base.genre)
    if pref is None:
        return base, ["No highly-rated history yet — using settings as provided."]
    means, num_liked = pref
    blend = min(_MAX_BLEND, _BLEND_PER_LIKE * num_liked)

    new_intensity = _clamp(base.intensity + blend * (means["intensity"] - base.intensity), 0.0, 1.0)
    new_vocal = _clamp(
        base.vocal_prominence + blend * (means["vocal_prominence"] - base.vocal_prominence),
        0.0,
        1.0,
    )
    new_tone = _clamp(base.tone + blend * (means["tone"] - base.tone), -1.0, 1.0)

    notes = [
        f"Personalized from {int(num_liked)} liked mix(es) (blend {blend:.0%}):",
        f"  intensity {base.intensity:.2f} -> {new_intensity:.2f}",
        f"  vocal     {base.vocal_prominence:.2f} -> {new_vocal:.2f}",
        f"  tone      {base.tone:+.2f} -> {new_tone:+.2f}",
    ]

    adjusted = MixSettings(
        genre=base.genre,
        intensity=new_intensity,
        vocal_prominence=new_vocal,
        tone=new_tone,
        target_lufs=base.target_lufs,
        peak_ceiling_db=base.peak_ceiling_db,
        locked=base.locked,
        reference=base.reference,
        track_overrides=dict(base.track_overrides),
        target_override=base.target_override,
    )
    return adjusted, notes
