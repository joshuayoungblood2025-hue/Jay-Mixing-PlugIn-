"""The mixing engine: turns raw stems into a balanced stereo mix.

Pipeline per run:

1. Analyze + classify every stem.
2. Build a per-track processing plan (EQ, compression) from role + genre + user controls.
3. Process each stem, then re-measure loudness and gain-stage it to its target balance.
4. Pan/place each stem in the stereo field.
5. Sum to a stereo bus; apply reference/genre tonal correction + glue compression.
6. Normalize the bus to the genre loudness target and apply a safety limiter.

Everything is deterministic and explainable; the returned plans/report capture every move.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field

from mixassist.analysis.classify import (
    BASS,
    DRUMS,
    INSTRUMENT,
    VOCAL,
    Classification,
    classify_track,
)
from mixassist.analysis.features import Features, extract_features
from mixassist.analysis.spectrum import tonal_balance
from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp.compressor import CompressorSettings, compress, limit, sidechain_duck
from mixassist.dsp.delay import StereoDelay
from mixassist.dsp.eq import EQBand, apply_eq
from mixassist.dsp.gain import db_to_lin, mix_into, pan_to_stereo
from mixassist.dsp.loudness import integrated_lufs, peak_dbfs
from mixassist.dsp.onset import build_trigger, detect_kick_onsets
from mixassist.dsp.reverb import Reverb
from mixassist.dsp.saturation import saturate
from mixassist.mixing import reference as refmod
from mixassist.mixing.targets import STEM_ANCHOR_LUFS, GenreTarget, get_target

_CENTERED_SUBTYPES = {"kick", "lead_vocal", "808", "sub", "bass"}

# Per-role/-subtype send amounts for the creative FX return buses (0..1). Looked up by
# subtype first, then role. These are tasteful engineering defaults, scaled by the user's
# Reverb/Delay amount.
_REVERB_SENDS = {
    "vocal": 0.9,
    "lead_vocal": 1.0,
    "backing_vocal": 0.9,
    "rap": 0.5,
    "snare": 0.5,
    "drums": 0.3,
    "hihat": 0.15,
    "cymbal": 0.2,
    "kick": 0.0,
    "808": 0.0,
    "sub": 0.0,
    "bass": 0.0,
    "instrument": 0.4,
    "synth": 0.45,
    "guitar": 0.4,
    "keys": 0.4,
    "piano": 0.35,
    "pad": 0.6,
    "strings": 0.55,
    "fx": 0.6,
}
_DELAY_SENDS = {
    "vocal": 0.45,
    "lead_vocal": 0.55,
    "rap": 0.4,
    "snare": 0.15,
    "drums": 0.0,
    "kick": 0.0,
    "808": 0.0,
    "sub": 0.0,
    "bass": 0.0,
    "instrument": 0.25,
    "synth": 0.3,
    "guitar": 0.3,
    "keys": 0.2,
    "fx": 0.4,
}


def _send_amount(table: dict[str, float], role: str, subtype: str) -> float:
    if subtype in table:
        return table[subtype]
    return table.get(role, 0.0)


@dataclass
class TrackOverride:
    """User overrides for a single track (the "control surface" per channel)."""

    mute: bool = False
    solo: bool = False
    lock: bool = False
    gain_trim_db: float = 0.0
    pan: float | None = None
    width: float | None = None
    extra_eq: list[EQBand] = field(default_factory=list)


@dataclass
class MixSettings:
    genre: str = "default"
    intensity: float = 0.5  # 0 (subtle) .. 1 (aggressive)
    vocal_prominence: float = 0.5  # 0 (buried) .. 1 (up front); 0.5 neutral
    tone: float = 0.0  # -1 (warm) .. +1 (clarity/bright)
    target_lufs: float | None = None
    peak_ceiling_db: float | None = None
    locked: frozenset[str] = frozenset()
    reference: object | None = None  # TonalBalance or None
    track_overrides: dict[str, TrackOverride] = field(default_factory=dict)
    target_override: GenreTarget | None = None  # a learned profile / custom target, or None
    # Creative FX amounts (0..1). Library default is 0 (off); the CLI/app set tasteful defaults.
    reverb: float = 0.0
    delay: float = 0.0
    drive: float = 0.0  # bus saturation / warmth
    sidechain: float = 0.0  # kick -> bass ducking depth (0..1)

    def clamped(self) -> MixSettings:
        return MixSettings(
            genre=self.genre,
            intensity=_clamp01(self.intensity),
            vocal_prominence=_clamp01(self.vocal_prominence),
            tone=max(-1.0, min(1.0, self.tone)),
            target_lufs=self.target_lufs,
            peak_ceiling_db=self.peak_ceiling_db,
            locked=frozenset(self.locked),
            reference=self.reference,
            track_overrides=dict(self.track_overrides),
            target_override=self.target_override,
            reverb=_clamp01(self.reverb),
            delay=_clamp01(self.delay),
            drive=_clamp01(self.drive),
            sidechain=_clamp01(self.sidechain),
        )

    def override_for(self, name: str) -> TrackOverride:
        return self.track_overrides.get(name, TrackOverride())

    @property
    def solo_active(self) -> bool:
        return any(o.solo for o in self.track_overrides.values())


@dataclass
class TrackPlan:
    name: str
    classification: Classification
    features: Features
    locked: bool
    muted: bool = False
    gain_db: float = 0.0
    gain_trim_db: float = 0.0
    pan: float = 0.0
    width: float = 1.0
    eq_bands: list[EQBand] = field(default_factory=list)
    comp: CompressorSettings | None = None
    comp_gr_db: float = 0.0
    sidechain_gr_db: float = 0.0
    in_lufs: float = float("-inf")
    out_lufs: float = float("-inf")


@dataclass
class BusPlan:
    tonal_eq: list[EQBand] = field(default_factory=list)
    glue: CompressorSettings | None = None
    glue_gr_db: float = 0.0
    normalize_gain_db: float = 0.0
    limiter_gr_db: float = 0.0
    target_lufs: float = -15.0
    peak_ceiling_db: float = -1.0
    final_lufs: float = float("-inf")
    final_peak_dbfs: float = float("-inf")
    reverb_amount: float = 0.0
    delay_amount: float = 0.0
    drive_amount: float = 0.0
    sidechain_kick_hits: int = 0


@dataclass
class MixResult:
    master: AudioBuffer
    stems: dict[str, AudioBuffer]
    track_plans: list[TrackPlan]
    bus_plan: BusPlan
    target: GenreTarget


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# --------------------------------------------------------------------------- EQ rules


def _tone_bands(tone: float, role: str) -> list[EQBand]:
    bands: list[EQBand] = []
    if role == BASS:
        return bands
    if tone > 0.02:  # clarity / brighter
        bands.append(EQBand("highshelf", 8000.0, gain_db=tone * 2.5, reason="clarity: lift air"))
        bands.append(
            EQBand("peak", 300.0, gain_db=-tone * 1.5, q=0.9, reason="clarity: reduce mud")
        )
    elif tone < -0.02:  # warmth
        w = -tone
        bands.append(EQBand("lowshelf", 200.0, gain_db=w * 2.5, reason="warmth: add body"))
        bands.append(EQBand("highshelf", 9000.0, gain_db=-w * 2.0, reason="warmth: soften top"))
    return bands


def _role_eq(role: str, subtype: str, feat: Features, s: MixSettings) -> list[EQBand]:
    i = s.intensity
    bands: list[EQBand] = []

    if role == VOCAL:
        bands.append(EQBand("highpass", 90.0, reason="remove rumble below voice"))
        bands.append(EQBand("peak", 300.0, gain_db=-2.0 * i, q=1.0, reason="reduce boxiness"))
        bands.append(
            EQBand("peak", 4000.0, gain_db=3.0 * i, q=0.8, reason="add presence/intelligibility")
        )
        bands.append(EQBand("highshelf", 11000.0, gain_db=2.0 * i, reason="add air"))
    elif role == BASS:
        bands.append(EQBand("highpass", 28.0, reason="remove subsonic energy"))
        bands.append(EQBand("lowshelf", 90.0, gain_db=1.5 * i, reason="reinforce low-end weight"))
        bands.append(EQBand("peak", 250.0, gain_db=-1.5 * i, q=1.0, reason="clean low-mid mud"))
        bands.append(EQBand("peak", 800.0, gain_db=1.0 * i, q=1.2, reason="definition/attack"))
    elif role == DRUMS:
        if subtype == "kick":
            bands.append(EQBand("highpass", 30.0, reason="tighten sub"))
            bands.append(EQBand("peak", 60.0, gain_db=2.0 * i, q=1.0, reason="kick thump"))
            bands.append(EQBand("peak", 3500.0, gain_db=2.0 * i, q=1.0, reason="beater attack"))
        elif subtype == "snare":
            bands.append(EQBand("highpass", 90.0, reason="remove low rumble"))
            bands.append(EQBand("peak", 200.0, gain_db=1.5 * i, q=1.0, reason="snare body"))
            bands.append(EQBand("peak", 4500.0, gain_db=2.0 * i, q=0.9, reason="crack/snap"))
        elif subtype in ("hihat", "cymbal"):
            bands.append(EQBand("highpass", 300.0, reason="remove spill/low bleed"))
            bands.append(EQBand("highshelf", 10000.0, gain_db=2.0 * i, reason="sparkle"))
        else:
            bands.append(EQBand("highpass", 60.0, reason="clean sub"))
            bands.append(EQBand("peak", 3000.0, gain_db=1.5 * i, q=0.9, reason="drum attack"))
    elif role == INSTRUMENT:
        bands.append(EQBand("highpass", 70.0, reason="clear low-end for bass/kick"))
        bands.append(EQBand("peak", 300.0, gain_db=-1.0 * i, q=1.0, reason="reduce mud buildup"))
        bands.append(EQBand("peak", 3000.0, gain_db=1.0 * i, q=0.9, reason="presence"))
    else:  # FX
        bands.append(EQBand("highpass", 120.0, reason="keep low-end clean for FX"))

    bands.extend(_tone_bands(s.tone, role))
    # drop no-op bands
    return [b for b in bands if b.kind in ("highpass", "lowpass") or abs(b.gain_db) >= 0.1]


# --------------------------------------------------------------------- compression rules


def _role_comp(role: str, subtype: str, s: MixSettings) -> CompressorSettings | None:
    i = s.intensity
    if role == VOCAL:
        return CompressorSettings(
            threshold_db=-20.0,
            ratio=2.5 + 1.5 * i,
            attack_ms=8.0,
            release_ms=120.0,
            knee_db=6.0,
            reason="controlled, consistent vocal level",
        )
    if role == BASS:
        return CompressorSettings(
            threshold_db=-18.0,
            ratio=3.0 + 1.5 * i,
            attack_ms=15.0,
            release_ms=120.0,
            knee_db=6.0,
            reason="even, sustained low end",
        )
    if role == DRUMS:
        if subtype in ("hihat", "cymbal"):
            return None
        return CompressorSettings(
            threshold_db=-18.0,
            ratio=3.0 + 2.0 * i,
            attack_ms=5.0,
            release_ms=90.0,
            knee_db=4.0,
            reason="punch and consistency",
        )
    if role == INSTRUMENT:
        if i < 0.25:
            return None
        return CompressorSettings(
            threshold_db=-20.0,
            ratio=2.0 + 1.0 * i,
            attack_ms=20.0,
            release_ms=150.0,
            knee_db=6.0,
            reason="gentle level control",
        )
    return None  # FX untouched by default


# ------------------------------------------------------------------------- gain staging


def _desired_lufs(role: str, target: GenreTarget, s: MixSettings) -> float:
    offset = target.role_level_lu.get(role, target.role_level_lu.get(INSTRUMENT, -4.0))
    vp = (s.vocal_prominence - 0.5) * 8.0  # +/-4 LU at extremes
    if role == VOCAL:
        offset += vp
    else:
        offset -= vp * 0.4
    return STEM_ANCHOR_LUFS + offset


# ----------------------------------------------------------------------------- panning


class _Panner:
    """Assigns deterministic pan positions, keeping key roles centered."""

    def __init__(self, spread: float) -> None:
        self.spread = spread
        self._n = 0

    def place(self, role: str, subtype: str, feat: Features) -> tuple[float, float]:
        centered = subtype in _CENTERED_SUBTYPES or role == BASS
        if role == VOCAL and subtype in ("vocal", "lead_vocal", "rap"):
            centered = True
        if centered:
            return 0.0, min(1.0, feat.stereo_width if feat.stereo_width else 0.0) or (
                1.0 if feat.num_channels > 1 else 0.0
            )
        # Spread remaining sources alternately outward.
        k = self._n
        self._n += 1
        sign = -1.0 if (k % 2 == 0) else 1.0
        step = (k // 2 + 1) / 3.0
        pos = sign * self.spread * min(1.0, step)
        width = 1.0 if feat.num_channels > 1 else 0.0
        return pos, width


# ----------------------------------------------------------------------------- engine


def _process_track(name: str, buf: AudioBuffer, plan: TrackPlan, s: MixSettings) -> AudioBuffer:
    work = buf.copy()
    if not plan.locked:
        apply_eq(work, plan.eq_bands)
        if plan.comp is not None:
            plan.comp_gr_db = compress(work, plan.comp)
        # Re-measure post-EQ/comp and gain-stage to the balance target.
        measured = integrated_lufs(work)
        plan.out_lufs = measured
        if plan.gain_db != 0.0:
            work.apply_gain(db_to_lin(plan.gain_db))
    else:
        plan.out_lufs = plan.in_lufs
    return pan_to_stereo(work, plan.pan, plan.width)


def _apply_creative_fx(bus, plans, processed, s, bus_plan, sample_rate: int, n: int) -> None:
    """Build reverb/delay send buses from role-appropriate amounts and add the wet returns."""
    if s.reverb <= 0.0 and s.delay <= 0.0:
        return
    rev_send = array("d", bytes(8 * n))
    dly_send = array("d", bytes(8 * n))
    have_rev = have_dly = False
    for plan in plans:
        if plan.muted or plan.features.silence:
            continue
        cls = plan.classification
        mono = processed[plan.name].mono()
        m = min(n, len(mono))
        if s.reverb > 0.0:
            amt = _send_amount(_REVERB_SENDS, cls.role, cls.subtype) * s.reverb
            if amt > 0.0:
                have_rev = True
                for i in range(m):
                    rev_send[i] += mono[i] * amt
        if s.delay > 0.0:
            amt = _send_amount(_DELAY_SENDS, cls.role, cls.subtype) * s.delay
            if amt > 0.0:
                have_dly = True
                for i in range(m):
                    dly_send[i] += mono[i] * amt

    if s.reverb > 0.0 and have_rev:
        wet = Reverb(
            sample_rate, room_size=0.55 + 0.35 * s.reverb, damping=0.5, width=1.0
        ).process_send(rev_send)
        mix_into(bus, wet, gain=1.4)
        bus_plan.reverb_amount = s.reverb

    if s.delay > 0.0 and have_dly:
        wet = StereoDelay(
            sample_rate,
            time_ms=350.0,
            feedback=0.3 + 0.25 * s.delay,
            damping=0.3,
            ping_pong=True,
        ).process_send(dly_send)
        mix_into(bus, wet, gain=0.6)
        bus_plan.delay_amount = s.delay


def _apply_sidechain(plans, processed, s) -> int:
    """Duck bass tracks using a clean trigger detected from the kick. Returns hit count.

    We don't rely on separation: we detect *when* the kick hits (low-frequency transients)
    and synthesize a clean exponential-decay trigger at those moments — an artifact-free
    side-chain key, even from a full drum loop.
    """
    if s.sidechain <= 0.0:
        return 0
    drums = [
        p
        for p in plans
        if p.classification.role == DRUMS and not p.muted and not p.features.silence
    ]
    basses = [
        p for p in plans if p.classification.role == BASS and not p.muted and not p.features.silence
    ]
    if not drums or not basses:
        return 0

    # Prefer an isolated kick as the detection source; else use the whole drum bus.
    kicks = [p for p in drums if p.classification.subtype == "kick"]
    key_src = kicks if kicks else drums
    fs = processed[key_src[0].name].sample_rate
    n = max(processed[p.name].num_frames for p in key_src)
    raw = array("d", bytes(8 * n))
    for p in key_src:
        mono = processed[p.name].mono()
        m = min(n, len(mono))
        for i in range(m):
            raw[i] += mono[i]

    hits = detect_kick_onsets(raw, fs)
    if not hits:
        return 0
    key = build_trigger(hits, n, fs)
    for b in basses:
        b.sidechain_gr_db = sidechain_duck(processed[b.name], key, amount=s.sidechain)
    return len(hits)


def mix(stems: dict[str, AudioBuffer], settings: MixSettings) -> MixResult:
    if not stems:
        raise ValueError("no stems to mix")
    s = settings.clamped()
    target = s.target_override if s.target_override is not None else get_target(s.genre)
    target_lufs = s.target_lufs if s.target_lufs is not None else target.bus_lufs
    ceiling = s.peak_ceiling_db if s.peak_ceiling_db is not None else target.peak_ceiling_db

    sample_rate = max(b.sample_rate for b in stems.values())
    max_frames = max(b.num_frames for b in stems.values())

    panner = _Panner(target.spread)
    plans: list[TrackPlan] = []
    processed: dict[str, AudioBuffer] = {}

    # Sort so centered/anchor roles are placed first, then spread the rest.
    def sort_key(item: tuple[str, AudioBuffer]) -> tuple[int, str]:
        return (0, item[0])

    for name, buf in sorted(stems.items(), key=sort_key):
        feat = extract_features(name, buf)
        cls = classify_track(feat)
        ov = s.override_for(name)
        locked = name in s.locked or ov.lock
        muted = ov.mute or (s.solo_active and not ov.solo)
        plan = TrackPlan(
            name=name,
            classification=cls,
            features=feat,
            locked=locked,
            muted=muted,
            gain_trim_db=ov.gain_trim_db,
            in_lufs=feat.lufs,
        )
        if not locked and not feat.silence:
            plan.eq_bands = _role_eq(cls.role, cls.subtype, feat, s)
            if ov.extra_eq:
                plan.eq_bands = plan.eq_bands + list(ov.extra_eq)
            plan.comp = _role_comp(cls.role, cls.subtype, s)
            desired = _desired_lufs(cls.role, target, s)
            # gain relative to *input* loudness; recomputed against post-processing below
            if feat.lufs != float("-inf"):
                raw_gain = desired - feat.lufs
                plan.gain_db = max(-24.0, min(24.0, raw_gain))
        # Panning: honor explicit overrides, else auto-place.
        auto_pan, auto_width = panner.place(cls.role, cls.subtype, feat)
        plan.pan = ov.pan if ov.pan is not None else auto_pan
        plan.width = ov.width if ov.width is not None else auto_width
        plans.append(plan)
        processed[name] = _process_track(name, buf, plan, s)

    # Recompute the applied gain against the post-processing loudness for accuracy,
    # so the balance target is met regardless of what EQ/compression did. A user gain
    # trim is added on top of the auto-balanced level.
    for plan in plans:
        if plan.locked or plan.features.silence:
            if plan.gain_trim_db and not plan.features.silence:
                processed[plan.name].apply_gain(db_to_lin(plan.gain_trim_db))
            continue
        if plan.out_lufs != float("-inf"):
            desired = _desired_lufs(plan.classification.role, target, s)
            corrected = max(-24.0, min(24.0, desired - plan.out_lufs)) + plan.gain_trim_db
            # adjust the already-panned stem by the delta
            delta = corrected - plan.gain_db
            if abs(delta) > 0.01:
                processed[plan.name].apply_gain(db_to_lin(delta))
            plan.gain_db = corrected

    # --- Kick -> bass side-chain ducking (before summing) ------------------
    sidechain_hits = _apply_sidechain(plans, processed, s)

    # --- Sum the bus (muted / solo-excluded tracks are rendered but not summed) ---
    bus = AudioBuffer.silence(max_frames, sample_rate, 2)
    for plan in plans:
        if plan.muted:
            continue
        mix_into(bus, processed[plan.name], gain=1.0)

    bus_plan = BusPlan(target_lufs=target_lufs, peak_ceiling_db=ceiling)
    bus_plan.sidechain_kick_hits = sidechain_hits

    # --- Creative FX send returns (reverb / delay) added into the bus ------
    _apply_creative_fx(bus, plans, processed, s, bus_plan, sample_rate, max_frames)

    # --- Tonal correction (reference or genre curve) -----------------------
    current_balance = tonal_balance(bus)
    if s.reference is not None:
        target_shape = refmod.reference_shape(s.reference)  # type: ignore[arg-type]
        bus_plan.tonal_eq = refmod.build_corrective_eq(
            current_balance, target_shape, max_db=4.0, strength=0.7
        )
    else:
        bus_plan.tonal_eq = refmod.build_corrective_eq(
            current_balance, target.curve_list(), max_db=2.5, strength=0.5
        )
    apply_eq(bus, bus_plan.tonal_eq)

    # --- Glue compression --------------------------------------------------
    glue = CompressorSettings(
        threshold_db=-18.0,
        ratio=1.5 + 0.8 * s.intensity,
        attack_ms=30.0,
        release_ms=200.0,
        knee_db=6.0,
        reason="bus glue: cohere the mix",
    )
    bus_plan.glue = glue
    bus_plan.glue_gr_db = compress(bus, glue)

    # --- Saturation / drive (warmth + harmonic glue) -----------------------
    if s.drive > 0.0:
        saturate(bus, drive=s.drive * 0.7, mix=0.85, bias=0.12)
        bus_plan.drive_amount = s.drive

    # --- Normalize to target loudness --------------------------------------
    pre_lufs = integrated_lufs(bus)
    if pre_lufs != float("-inf"):
        norm_gain = target_lufs - pre_lufs
        bus_plan.normalize_gain_db = norm_gain
        bus.apply_gain(db_to_lin(norm_gain))

    # --- Safety limiter ----------------------------------------------------
    bus_plan.limiter_gr_db = limit(bus, ceiling_db=ceiling, release_ms=50.0)

    bus_plan.final_lufs = integrated_lufs(bus)
    bus_plan.final_peak_dbfs = peak_dbfs(bus)

    return MixResult(
        master=bus,
        stems=processed,
        track_plans=plans,
        bus_plan=bus_plan,
        target=target,
    )
