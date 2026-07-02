"""Kick -> bass side-chain ducking."""

from __future__ import annotations

import math
import random
from array import array

from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp.compressor import sidechain_duck
from mixassist.dsp.onset import build_trigger, detect_kick_onsets
from mixassist.mixing.engine import MixSettings, mix


def test_sidechain_ducks_on_key_hits():
    sr = 44100
    n = sr
    # steady bass tone
    bass = AudioBuffer(
        [
            array("d", [0.5 * math.sin(2 * math.pi * 60 * i / sr) for i in range(n)])
            for _ in range(2)
        ],
        sr,
    )
    # key: a kick pulse at t=0 and t=0.5s
    key = array("d", bytes(8 * n))
    for start in (0, sr // 2):
        for j in range(2000):
            key[start + j] = math.exp(-j / 400.0)

    gr = sidechain_duck(bass, key, amount=0.8, attack_ms=2.0, release_ms=120.0)
    assert gr > 3.0  # meaningful ducking happened

    # level right after a kick should be lower than well between kicks
    def rms(a, s, e):
        return math.sqrt(sum(a[i] * a[i] for i in range(s, e)) / (e - s))

    ducked = rms(bass.channels[0], 200, 1200)  # just after the first kick
    recovered = rms(bass.channels[0], sr // 2 - 1200, sr // 2 - 200)  # before next kick
    assert ducked < recovered


def test_sidechain_noop_when_amount_zero():
    sr = 44100
    bass = AudioBuffer([array("d", [0.5] * sr), array("d", [0.5] * sr)], sr)
    key = array("d", [1.0] * sr)
    gr = sidechain_duck(bass, key, amount=0.0)
    assert gr == 0.0
    assert bass.channels[0][100] == 0.5  # untouched


def test_engine_sidechain_records_on_bass(synth_stems):
    res = mix(synth_stems, MixSettings(genre="hiphop", sidechain=0.6))
    bass_plan = next((p for p in res.track_plans if p.classification.role == "bass"), None)
    assert bass_plan is not None
    assert bass_plan.sidechain_gr_db > 0.0


def test_engine_no_sidechain_by_default(synth_stems):
    res = mix(synth_stems, MixSettings(genre="hiphop"))
    for p in res.track_plans:
        assert p.sidechain_gr_db == 0.0


def test_detect_kick_onsets_finds_kicks_ignores_hats():
    sr = 44100
    n = 2 * sr
    loop = array("d", bytes(8 * n))
    rng = random.Random(1)
    for start in (0, sr // 2, sr, 3 * sr // 2):  # kicks at 0/0.5/1.0/1.5 s
        for j in range(6000):
            if start + j < n:
                loop[start + j] += 0.9 * math.sin(2 * math.pi * 55 * j / sr) * math.exp(-j / 2500)
    for start in range(0, n, sr // 8):  # hats every 1/8 s (should be ignored)
        for j in range(800):
            if start + j < n:
                loop[start + j] += (rng.random() * 2 - 1) * 0.3 * math.exp(-j / 300)

    hits = detect_kick_onsets(loop, sr)
    assert len(hits) == 4
    times = [h / sr for h in hits]
    for expected in (0.0, 0.5, 1.0, 1.5):
        assert any(abs(t - expected) < 0.03 for t in times)


def test_build_trigger_pulses_at_hits():
    sr = 44100
    n = sr
    key = build_trigger([0, sr // 2], n, sr)
    assert key[0] == 1.0
    assert key[sr // 2] == 1.0
    assert max(key) <= 1.0


def test_detect_onsets_empty_is_safe():
    assert detect_kick_onsets(array("d", []), 44100) == []
