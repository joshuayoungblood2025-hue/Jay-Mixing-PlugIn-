"""Block-based real-time bus chain: block invariance, limiter, gain."""

from __future__ import annotations

import math
from array import array

import pytest

from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp.compressor import CompressorSettings
from mixassist.dsp.eq import EQBand
from mixassist.dsp.loudness import peak_dbfs
from mixassist.realtime.blocks import BusChain, StreamingLimiter

_BANDS = [
    EQBand("highpass", 30),
    EQBand("lowshelf", 100, 1.5),
    EQBand("peak", 300, -2, 1.0),
    EQBand("highshelf", 10000, 2.0),
]
_COMP = CompressorSettings(threshold_db=-18, ratio=2.0, attack_ms=30, release_ms=200, knee_db=6)


def _sig(sr: int, n: int, amp: float = 0.6) -> AudioBuffer:
    left = array(
        "d",
        [
            amp * math.sin(2 * math.pi * 220 * i / sr) + 0.3 * math.sin(2 * math.pi * 3000 * i / sr)
            for i in range(n)
        ],
    )
    right = array("d", [amp * math.sin(2 * math.pi * 221 * i / sr) for i in range(n)])
    return AudioBuffer([left, right], sr)


@pytest.mark.parametrize("block_size", [1, 64, 512, 4096])
def test_block_invariance(block_size):
    sr = 44100
    n = sr // 2
    whole = BusChain(sr, _BANDS, _COMP, -1.0, 3.0).process(_sig(sr, n), block_size=n)
    part = BusChain(sr, _BANDS, _COMP, -1.0, 3.0).process(_sig(sr, n), block_size=block_size)
    err = max(abs(whole.channels[0][i] - part.channels[0][i]) for i in range(0, n, 3))
    err += max(abs(whole.channels[1][i] - part.channels[1][i]) for i in range(0, n, 3))
    assert err < 1e-12


def test_streaming_limiter_caps_peak():
    sr = 44100
    n = sr // 4
    hot = AudioBuffer(
        [
            array("d", [1.5 * math.sin(2 * math.pi * 100 * i / sr) for i in range(n)])
            for _ in range(2)
        ],
        sr,
    )
    lim = StreamingLimiter(-1.0, sr)
    # feed in blocks
    pos = 0
    while pos < n:
        cnt = min(256, n - pos)
        lim.process_block(hot.channels, pos, cnt)
        pos += cnt
    assert peak_dbfs(hot) <= -1.0 + 0.1


def test_output_gain_applied():
    sr = 44100
    n = 1000
    buf = AudioBuffer([array("d", [0.1] * n), array("d", [0.1] * n)], sr)
    BusChain(sr, [], None, 0.0, 6.0).process(buf, block_size=128)  # +6 dB, ceiling 0 dBFS
    # 0.1 * 10**(6/20) ~= 0.1995, below the 1.0 ceiling so no limiting
    assert abs(buf.channels[0][500] - 0.1995) < 0.005


def test_reset_clears_state():
    sr = 44100
    chain = BusChain(sr, _BANDS, _COMP, -1.0, 0.0)
    chain.process(_sig(sr, 1000), block_size=128)
    chain.reset()
    assert chain.limiter.env == 0.0
    if chain.comp is not None:
        assert chain.comp.env == 0.0
