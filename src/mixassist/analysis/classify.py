"""Rule-based track-role classification.

Hybrid approach: strong filename hints (engineers name their stems) combined with spectral
heuristics from :class:`Features`. Returns a role plus a confidence and human-readable
reasons so the report can explain itself. This is deliberately transparent and rule-based;
Phase 3 can replace the scorer with a learned model behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mixassist.analysis.features import Features

# Canonical mixing roles.
VOCAL = "vocal"
DRUMS = "drums"
BASS = "bass"
INSTRUMENT = "instrument"
FX = "fx"
ROLES = (VOCAL, DRUMS, BASS, INSTRUMENT, FX)

# Filename keyword hints -> (role, subtype). Checked as substrings, longest first.
_KEYWORDS: list[tuple[str, str, str]] = [
    ("lead vocal", VOCAL, "lead_vocal"),
    ("backing vocal", VOCAL, "backing_vocal"),
    ("bgv", VOCAL, "backing_vocal"),
    ("adlib", VOCAL, "adlib"),
    ("vocal", VOCAL, "vocal"),
    ("vox", VOCAL, "vocal"),
    ("lead vox", VOCAL, "lead_vocal"),
    ("voc", VOCAL, "vocal"),
    ("sing", VOCAL, "vocal"),
    ("rap", VOCAL, "rap"),
    ("verse", VOCAL, "vocal"),
    ("kick", DRUMS, "kick"),
    ("snare", DRUMS, "snare"),
    ("hat", DRUMS, "hihat"),
    ("hi-hat", DRUMS, "hihat"),
    ("hihat", DRUMS, "hihat"),
    ("cymbal", DRUMS, "cymbal"),
    ("crash", DRUMS, "cymbal"),
    ("ride", DRUMS, "cymbal"),
    ("tom", DRUMS, "tom"),
    ("perc", DRUMS, "percussion"),
    ("clap", DRUMS, "clap"),
    ("drum", DRUMS, "drums"),
    ("beat", DRUMS, "drums"),
    ("808", BASS, "808"),
    ("sub", BASS, "sub"),
    ("bass", BASS, "bass"),
    ("guitar", INSTRUMENT, "guitar"),
    ("gtr", INSTRUMENT, "guitar"),
    ("piano", INSTRUMENT, "piano"),
    ("keys", INSTRUMENT, "keys"),
    ("rhodes", INSTRUMENT, "keys"),
    ("organ", INSTRUMENT, "keys"),
    ("synth", INSTRUMENT, "synth"),
    ("pad", INSTRUMENT, "pad"),
    ("string", INSTRUMENT, "strings"),
    ("horn", INSTRUMENT, "horns"),
    ("brass", INSTRUMENT, "horns"),
    ("lead", INSTRUMENT, "lead"),
    ("arp", INSTRUMENT, "synth"),
    ("fx", FX, "fx"),
    ("riser", FX, "riser"),
    ("sweep", FX, "sweep"),
    ("impact", FX, "impact"),
    ("noise", FX, "noise"),
    ("ambient", FX, "ambience"),
    ("foley", FX, "foley"),
]


@dataclass
class Classification:
    role: str
    subtype: str
    confidence: float
    method: str  # "filename", "spectral", or "hybrid"
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "role": self.role,
            "subtype": self.subtype,
            "confidence": round(self.confidence, 2),
            "method": self.method,
            "reasons": self.reasons,
        }


def _filename_hint(name: str) -> tuple[str, str] | None:
    low = name.lower()
    for kw, role, subtype in _KEYWORDS:
        if kw in low:
            return role, subtype
    return None


def _spectral_scores(f: Features) -> dict[str, float]:
    """Heuristic role scores in roughly [0, 1] from spectral features."""
    scores = {r: 0.0 for r in ROLES}
    if f.silence:
        scores[FX] = 0.2
        return scores

    # BASS: energy concentrated low, dark centroid, not too transient.
    if f.low_ratio > 0.5:
        scores[BASS] += 0.5 + min(0.4, (f.low_ratio - 0.5))
    if f.centroid_hz < 300:
        scores[BASS] += 0.3
    if f.crest_db > 16:
        scores[BASS] -= 0.2  # very transient -> more likely percussive

    # DRUMS: transient (high crest), broadband, often high ZCR.
    if f.crest_db > 12:
        scores[DRUMS] += 0.4 + min(0.3, (f.crest_db - 12) / 20.0)
    if f.zero_cross_rate > 3000:
        scores[DRUMS] += 0.2
    if f.high_ratio > 0.4:
        scores[DRUMS] += 0.15  # cymbals / hats

    # VOCAL: midrange dominant, centroid in speech/vocal region, some dynamics.
    if f.mid_ratio > 0.45 and 350 <= f.centroid_hz <= 3000:
        scores[VOCAL] += 0.5
    if 8 <= f.crest_db <= 20:
        scores[VOCAL] += 0.2
    if f.stereo_width < 0.3:
        scores[VOCAL] += 0.1  # lead vocals usually centered/narrow

    # INSTRUMENT: balanced spectrum, moderate everything.
    if f.mid_ratio >= 0.2 and f.low_ratio < 0.6:
        scores[INSTRUMENT] += 0.35
    if 200 <= f.centroid_hz <= 5000:
        scores[INSTRUMENT] += 0.15

    # FX: very wide, or extreme centroid, or noisy sustained texture.
    if f.stereo_width > 0.8:
        scores[FX] += 0.3
    if f.centroid_hz > 8000:
        scores[FX] += 0.2

    return scores


def classify_track(features: Features) -> Classification:
    hint = _filename_hint(features.name)
    scores = _spectral_scores(features)
    best_role = max(scores, key=lambda r: scores[r])
    best_score = scores[best_role]
    # squash spectral score into a confidence
    spectral_conf = max(0.0, min(0.85, best_score))

    reasons: list[str] = []
    reasons.append(
        f"centroid {features.centroid_hz:.0f} Hz, "
        f"low/mid/high {features.low_ratio:.2f}/"
        f"{features.mid_ratio:.2f}/{features.high_ratio:.2f}, "
        f"crest {features.crest_db:.1f} dB"
    )

    if hint is not None:
        role, subtype = hint
        agrees = role == best_role
        confidence = 0.95 if agrees else 0.8
        method = "hybrid" if agrees else "filename"
        reasons.insert(0, f"filename matched role '{role}' (subtype {subtype})")
        if not agrees:
            reasons.append(
                f"spectral guess was '{best_role}' ({spectral_conf:.2f}); trusting filename"
            )
        return Classification(role, subtype, confidence, method, reasons)

    subtype = {
        VOCAL: "vocal",
        DRUMS: "drums",
        BASS: "bass",
        INSTRUMENT: "instrument",
        FX: "fx",
    }[best_role]
    reasons.insert(0, f"no filename hint; spectral classifier chose '{best_role}'")
    return Classification(best_role, subtype, spectral_conf, "spectral", reasons)
