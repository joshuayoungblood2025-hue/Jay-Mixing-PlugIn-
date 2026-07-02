"""Deinterleaved floating-point audio buffer.

Samples are stored per-channel as :class:`array.array` of doubles (float64) in the range
``[-1.0, 1.0]`` nominally (values may exceed this before limiting). Keeping channels
separate makes routing, panning and per-channel DSP straightforward.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence

from mixassist.dsp.backend import get_backend

Samples = array  # array('d')


def _new(n: int = 0, fill: float = 0.0) -> Samples:
    a = array("d")
    if n:
        a.extend([fill] * n)
    return a


class AudioBuffer:
    """A multichannel float64 audio buffer at a fixed sample rate."""

    __slots__ = ("channels", "sample_rate")

    def __init__(self, channels: Sequence[Samples], sample_rate: int) -> None:
        if not channels:
            raise ValueError("AudioBuffer requires at least one channel")
        n = len(channels[0])
        for ch in channels:
            if len(ch) != n:
                raise ValueError("all channels must have the same length")
        self.channels: list[Samples] = [
            ch if isinstance(ch, array) and ch.typecode == "d" else array("d", ch)
            for ch in channels
        ]
        self.sample_rate = int(sample_rate)

    # -- construction helpers -------------------------------------------------

    @classmethod
    def silence(cls, num_frames: int, sample_rate: int, num_channels: int = 1) -> AudioBuffer:
        return cls([_new(num_frames) for _ in range(num_channels)], sample_rate)

    @classmethod
    def from_mono(cls, samples: Sequence[float], sample_rate: int) -> AudioBuffer:
        return cls([array("d", samples)], sample_rate)

    # -- properties -----------------------------------------------------------

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def num_frames(self) -> int:
        return len(self.channels[0])

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.sample_rate if self.sample_rate else 0.0

    # -- operations -----------------------------------------------------------

    def copy(self) -> AudioBuffer:
        return AudioBuffer([array("d", ch) for ch in self.channels], self.sample_rate)

    def mono(self) -> Samples:
        """Return a downmixed mono view (equal-weight average of channels)."""
        if self.num_channels == 1:
            return array("d", self.channels[0])
        n = self.num_frames
        nc = self.num_channels
        out = _new(n)
        chans = self.channels
        for i in range(n):
            s = 0.0
            for c in range(nc):
                s += chans[c][i]
            out[i] = s / nc
        return out

    def to_stereo(self) -> AudioBuffer:
        """Return a stereo version (duplicate mono, or pass stereo through)."""
        if self.num_channels == 2:
            return self.copy()
        if self.num_channels == 1:
            m = self.channels[0]
            return AudioBuffer([array("d", m), array("d", m)], self.sample_rate)
        # >2 channels: fold down to stereo by simple even/odd assignment
        left = self.mono()
        return AudioBuffer([left, array("d", left)], self.sample_rate)

    def peak(self) -> float:
        """Absolute sample peak across all channels (linear)."""
        backend = get_backend()
        p = 0.0
        for ch in self.channels:
            cp = backend.peak(ch)
            if cp > p:
                p = cp
        return p

    def apply_gain(self, linear_gain: float) -> None:
        """In-place scalar gain."""
        if linear_gain == 1.0:
            return
        backend = get_backend()
        for ch in self.channels:
            backend.apply_gain(ch, linear_gain)

    def scaled(self, linear_gain: float) -> AudioBuffer:
        out = self.copy()
        out.apply_gain(linear_gain)
        return out


def resample_linear(buf: AudioBuffer, target_rate: int) -> AudioBuffer:
    """Simple linear-interpolation resampler.

    Adequate for aligning stems recorded at different rates. Not a high-quality
    band-limited resampler; a windowed-sinc variant is a future upgrade.
    """
    if buf.sample_rate == target_rate:
        return buf.copy()
    ratio = target_rate / buf.sample_rate
    new_n = max(1, int(round(buf.num_frames * ratio)))
    new_channels: list[Samples] = []
    for ch in buf.channels:
        src_n = len(ch)
        out = _new(new_n)
        for i in range(new_n):
            src_pos = i / ratio
            i0 = int(src_pos)
            frac = src_pos - i0
            s0 = ch[i0] if i0 < src_n else 0.0
            s1 = ch[i0 + 1] if (i0 + 1) < src_n else s0
            out[i] = s0 + (s1 - s0) * frac
        new_channels.append(out)
    return AudioBuffer(new_channels, target_rate)
