"""Generate a synthetic multitrack session for testing the mixing engine.

The stems are deliberately simple synthesized sources (kick, snare, hats, bass, a vocal
formant tone, and a stereo pad) at a shared tempo, written as WAV files. They are enough
to exercise classification, balancing, EQ, panning and bus processing without needing real
copyrighted audio.
"""

from __future__ import annotations

import math
import random
from array import array
from pathlib import Path

from mixassist.audio.buffer import AudioBuffer
from mixassist.audio.io import save_wav


def _silence(n: int) -> array:
    return array("d", bytes(8 * n))


def _kick(n: int, sr: int, bpm: float) -> array:
    out = _silence(n)
    step = int(sr * 60.0 / bpm)  # one beat
    decay = sr * 0.09
    for start in range(0, n, step):
        for j in range(min(step, n - start)):
            t = j / sr
            freq = 55.0 * math.exp(-t * 12.0) + 40.0  # pitch drop
            out[start + j] += 0.95 * math.sin(2 * math.pi * freq * t) * math.exp(-j / decay)
    return out


def _snare(n: int, sr: int, bpm: float, rng: random.Random) -> array:
    out = _silence(n)
    beat = int(sr * 60.0 / bpm)
    decay = sr * 0.10
    # backbeats on 2 and 4
    for start in range(beat, n, beat * 2):
        for j in range(min(beat, n - start)):
            env = math.exp(-j / decay)
            tone = 0.4 * math.sin(2 * math.pi * 190 * j / sr)
            noise = 0.6 * (rng.random() * 2 - 1)
            out[start + j] += 0.7 * (tone + noise) * env
    return out


def _hats(n: int, sr: int, bpm: float, rng: random.Random) -> array:
    out = _silence(n)
    eighth = int(sr * 30.0 / bpm)
    decay = sr * 0.02
    for start in range(0, n, eighth):
        for j in range(min(eighth, n - start)):
            out[start + j] += 0.35 * (rng.random() * 2 - 1) * math.exp(-j / decay)
    return out


def _bass(n: int, sr: int, bpm: float) -> array:
    out = _silence(n)
    beat = int(sr * 60.0 / bpm)
    notes = [55.0, 55.0, 73.42, 82.41]  # A1, A1, D2, E2
    for idx, start in enumerate(range(0, n, beat)):
        f = notes[idx % len(notes)]
        for j in range(min(beat, n - start)):
            env = min(1.0, j / (sr * 0.01)) * math.exp(-j / (sr * 0.6))
            out[start + j] += 0.5 * math.sin(2 * math.pi * f * j / sr) * env
    return out


def _vocal(n: int, sr: int) -> array:
    """A formant-like sustained tone that reads as 'vocal' spectrally."""
    out = _silence(n)
    f0 = 220.0
    formants = [(700.0, 1.0), (1220.0, 0.5), (2600.0, 0.25)]
    for i in range(n):
        t = i / sr
        vibrato = 1.0 + 0.01 * math.sin(2 * math.pi * 5.0 * t)
        s = 0.0
        # a few harmonics shaped by formant peaks
        for h in range(1, 12):
            fh = f0 * h * vibrato
            amp = 1.0 / h
            shape = 0.0
            for fc, fg in formants:
                shape += fg * math.exp(-((fh - fc) ** 2) / (2 * (250.0**2)))
            s += amp * shape * math.sin(2 * math.pi * fh * t)
        # phrase envelope (sing, rest, sing)
        phrase = 0.0 if (int(t) % 4 == 3) else 1.0
        out[i] = 0.3 * s * phrase
    return out


def _pad(n: int, sr: int) -> tuple[array, array]:
    left = _silence(n)
    right = _silence(n)
    chord = [220.0, 277.18, 329.63]  # A major-ish
    for i in range(n):
        t = i / sr
        sl = 0.0
        sr_ = 0.0
        for f in chord:
            sl += math.sin(2 * math.pi * f * t)
            sr_ += math.sin(2 * math.pi * (f * 1.004) * t)  # slight detune for width
        swell = 0.5 + 0.5 * math.sin(2 * math.pi * 0.25 * t)
        left[i] = 0.12 * sl * swell
        right[i] = 0.12 * sr_ * swell
    return left, right


def generate_session(
    out_dir: str, seconds: float = 5.0, sr: int = 44100, bpm: float = 100.0
) -> Path:
    """Write a synthetic set of stems into ``out_dir/stems`` and return that folder."""
    rng = random.Random(1234)
    n = int(seconds * sr)
    stems_dir = Path(out_dir) / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    save_wav(stems_dir / "Kick.wav", AudioBuffer([_kick(n, sr, bpm)], sr))
    save_wav(stems_dir / "Snare.wav", AudioBuffer([_snare(n, sr, bpm, rng)], sr))
    save_wav(stems_dir / "Hats.wav", AudioBuffer([_hats(n, sr, bpm, rng)], sr))
    save_wav(stems_dir / "Bass.wav", AudioBuffer([_bass(n, sr, bpm)], sr))
    save_wav(stems_dir / "Lead Vocal.wav", AudioBuffer([_vocal(n, sr)], sr))
    pl, pr = _pad(n, sr)
    save_wav(stems_dir / "Synth Pad.wav", AudioBuffer([pl, pr], sr))
    return stems_dir
