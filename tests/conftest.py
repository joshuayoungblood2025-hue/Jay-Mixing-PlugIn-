"""Shared test fixtures: synthetic signals and a small stem session."""

from __future__ import annotations

import math
import random
from array import array

import pytest

from mixassist.audio.buffer import AudioBuffer


def sine(freq: float, sr: int, seconds: float, amp: float = 0.5) -> array:
    n = int(sr * seconds)
    return array("d", [amp * math.sin(2 * math.pi * freq * i / sr) for i in range(n)])


@pytest.fixture
def sr() -> int:
    return 44100


@pytest.fixture
def tone_1k_minus20(sr: int) -> AudioBuffer:
    # 1 kHz sine at -20 dBFS (amplitude 0.1)
    return AudioBuffer([sine(1000.0, sr, 2.0, amp=0.1)], sr)


@pytest.fixture
def synth_stems(sr: int) -> dict[str, AudioBuffer]:
    rng = random.Random(42)
    seconds = 2.0
    n = int(sr * seconds)

    bass = sine(80.0, sr, seconds, amp=0.5)
    vox = array(
        "d",
        [
            0.3
            * (
                math.sin(2 * math.pi * 280 * i / sr)
                + 0.6 * math.sin(2 * math.pi * 950 * i / sr)
                + 0.4 * math.sin(2 * math.pi * 2100 * i / sr)
            )
            for i in range(n)
        ],
    )
    hats = array(
        "d",
        [(rng.random() * 2 - 1) * 0.4 * (1.0 if (i % (sr // 8)) < 300 else 0.05) for i in range(n)],
    )
    padl = sine(440.0, sr, seconds, amp=0.2)
    padr = sine(442.0, sr, seconds, amp=0.2)

    return {
        "Kick": AudioBuffer([sine(55.0, sr, seconds, amp=0.6)], sr),
        "Bass": AudioBuffer([bass], sr),
        "Lead Vocal": AudioBuffer([vox], sr),
        "Hats": AudioBuffer([hats], sr),
        "Pad": AudioBuffer([padl, padr], sr),
    }
