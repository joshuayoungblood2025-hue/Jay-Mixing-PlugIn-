"""Streaming (block-based) bus processing that mirrors the plugin signal path.

The chain is: parametric EQ (stateful biquads) -> glue compressor -> output gain ->
brickwall-ish limiter. Every stage carries state across blocks, so processing in 64-sample
or 4096-sample blocks yields the same output as processing the whole buffer at once.
"""

from __future__ import annotations

import math

from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp.biquad import BiquadChain
from mixassist.dsp.compressor import CompressorSettings, _coef, _gain_curve
from mixassist.dsp.eq import EQBand
from mixassist.dsp.gain import db_to_lin


class StreamingCompressor:
    """Feed-forward compressor with state that persists across blocks (stereo-linked)."""

    def __init__(self, settings: CompressorSettings, fs: float) -> None:
        self.s = settings
        self.atk = _coef(settings.attack_ms, fs)
        self.rel = _coef(settings.release_ms, fs)
        self.makeup = 10.0 ** (settings.makeup_db / 20.0)
        self.env = 0.0
        self.max_gr_db = 0.0

    def process_block(self, channels: list, start: int, count: int) -> None:
        s = self.s
        atk, rel, makeup = self.atk, self.rel, self.makeup
        env = self.env
        nch = len(channels)
        thr, ratio, knee = s.threshold_db, s.ratio, s.knee_db
        for i in range(start, start + count):
            peak = 0.0
            for c in range(nch):
                a = channels[c][i]
                if a < 0.0:
                    a = -a
                if a > peak:
                    peak = a
            if peak > env:
                env = atk * env + (1.0 - atk) * peak
            else:
                env = rel * env + (1.0 - rel) * peak
            level_db = 20.0 * math.log10(env) if env > 1e-12 else -120.0
            out_db = _gain_curve(level_db, thr, ratio, knee)
            gr_db = out_db - level_db
            if -gr_db > self.max_gr_db:
                self.max_gr_db = -gr_db
            g = (10.0 ** (gr_db / 20.0)) * makeup
            for c in range(nch):
                channels[c][i] *= g
        self.env = env


class StreamingLimiter:
    """Feedback limiter with state that persists across blocks (stereo-linked)."""

    def __init__(self, ceiling_db: float, fs: float, release_ms: float = 50.0) -> None:
        self.ceiling = 10.0 ** (ceiling_db / 20.0)
        self.rel = _coef(release_ms, fs)
        self.env = 0.0
        self.max_gr_db = 0.0

    def process_block(self, channels: list, start: int, count: int) -> None:
        ceiling, rel = self.ceiling, self.rel
        env = self.env
        nch = len(channels)
        for i in range(start, start + count):
            peak = 0.0
            for c in range(nch):
                a = channels[c][i]
                if a < 0.0:
                    a = -a
                if a > peak:
                    peak = a
            target = ceiling / peak if peak > ceiling else 1.0
            if target < env or env == 0.0:
                env = target
            else:
                env = rel * env + (1.0 - rel) * target
                if env > 1.0:
                    env = 1.0
            if env < 1.0:
                gr = -20.0 * math.log10(env)
                if gr > self.max_gr_db:
                    self.max_gr_db = gr
            for c in range(nch):
                channels[c][i] *= env
        self.env = env


class BusChain:
    """A fixed bus processing chain applied in blocks. Mirrors the plugin's signal path."""

    def __init__(
        self,
        fs: float,
        eq_bands: list[EQBand],
        comp: CompressorSettings | None,
        limiter_ceiling_db: float,
        output_gain_db: float,
        num_channels: int = 2,
    ) -> None:
        self.fs = fs
        self.num_channels = num_channels
        self.eq_chains = [
            BiquadChain([b.to_biquad(fs) for b in eq_bands]) for _ in range(num_channels)
        ]
        self.comp = StreamingCompressor(comp, fs) if comp else None
        self.limiter = StreamingLimiter(limiter_ceiling_db, fs)
        self.output_gain = db_to_lin(output_gain_db)

    def reset(self) -> None:
        for ch in self.eq_chains:
            ch.reset()
        if self.comp:
            self.comp.env = 0.0
        self.limiter.env = 0.0

    def process(self, buf: AudioBuffer, block_size: int = 512) -> AudioBuffer:
        """Process ``buf`` in place in blocks of ``block_size`` and return it."""
        chans = buf.channels
        n = buf.num_frames
        nch = self.num_channels
        gain = self.output_gain
        pos = 0
        while pos < n:
            cnt = min(block_size, n - pos)
            end = pos + cnt
            # EQ per channel (stateful across blocks)
            for c in range(nch):
                seg = chans[c][pos:end]
                self.eq_chains[c].process_inplace(seg)
                chans[c][pos:end] = seg
            # Glue compression
            if self.comp is not None:
                self.comp.process_block(chans, pos, cnt)
            # Output gain (precomputed offline to hit the loudness target)
            if gain != 1.0:
                for c in range(nch):
                    ch = chans[c]
                    for i in range(pos, end):
                        ch[i] *= gain
            # Safety limiter
            self.limiter.process_block(chans, pos, cnt)
            pos = end
        return buf
