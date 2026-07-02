"""WAV round-trip tests across bit depths and formats."""

from __future__ import annotations

import math
from array import array

import pytest

from mixassist.audio.buffer import AudioBuffer, resample_linear
from mixassist.audio.io import load_wav, save_wav


def _sig(sr: int, n: int) -> AudioBuffer:
    left = array("d", [0.5 * math.sin(2 * math.pi * 440 * i / sr) for i in range(n)])
    right = array("d", [0.4 * math.sin(2 * math.pi * 440 * i / sr) for i in range(n)])
    return AudioBuffer([left, right], sr)


@pytest.mark.parametrize("bit_depth,tol", [(16, 3e-4), (24, 2e-6), (32, 1e-8)])
def test_pcm_roundtrip(tmp_path, bit_depth, tol):
    sr = 44100
    buf = _sig(sr, sr // 4)
    path = tmp_path / f"t{bit_depth}.wav"
    save_wav(path, buf, bit_depth=bit_depth)
    back = load_wav(path)
    assert back.num_channels == 2
    assert back.sample_rate == sr
    assert back.num_frames == buf.num_frames
    err = max(abs(back.channels[0][i] - buf.channels[0][i]) for i in range(0, buf.num_frames, 20))
    assert err < tol


def test_float32_roundtrip(tmp_path):
    sr = 48000
    buf = _sig(sr, sr // 4)
    path = tmp_path / "tf.wav"
    save_wav(path, buf, float_output=True)
    back = load_wav(path)
    err = max(abs(back.channels[0][i] - buf.channels[0][i]) for i in range(0, buf.num_frames, 20))
    assert err < 1e-6


def test_clipping_is_bounded(tmp_path):
    sr = 44100
    hot = AudioBuffer([array("d", [1.5, -1.5, 2.0, -2.0] * 100)], sr)
    path = tmp_path / "hot.wav"
    save_wav(path, hot, bit_depth=16)
    back = load_wav(path)
    assert back.peak() <= 1.0 + 1e-6


def test_resample_changes_length():
    sr = 44100
    buf = _sig(sr, sr // 2)
    down = resample_linear(buf, 22050)
    assert down.sample_rate == 22050
    assert abs(down.num_frames - buf.num_frames // 2) <= 2
