"""Parametric EQ built on top of biquad stages.

An :class:`EQBand` is a declarative description of one filter. :func:`apply_eq` builds a
per-channel biquad chain and processes a buffer in place. Bands are also serializable so
the mix report can explain every tonal move.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp import biquad
from mixassist.dsp.biquad import Biquad, BiquadChain

_KINDS = {"highpass", "lowpass", "lowshelf", "highshelf", "peak"}


@dataclass
class EQBand:
    kind: str
    freq: float
    gain_db: float = 0.0
    q: float = 0.7071
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown EQ band kind: {self.kind}")

    def to_biquad(self, fs: float) -> Biquad:
        if self.kind == "highpass":
            return biquad.high_pass(fs, self.freq, self.q)
        if self.kind == "lowpass":
            return biquad.low_pass(fs, self.freq, self.q)
        if self.kind == "lowshelf":
            return biquad.low_shelf(fs, self.freq, self.gain_db)
        if self.kind == "highshelf":
            return biquad.high_shelf(fs, self.freq, self.gain_db)
        return biquad.peaking(fs, self.freq, self.gain_db, self.q)

    def describe(self) -> str:
        if self.kind in ("highpass", "lowpass"):
            return f"{self.kind} @ {self.freq:.0f} Hz"
        sign = "+" if self.gain_db >= 0 else ""
        return f"{self.kind} {sign}{self.gain_db:.1f} dB @ {self.freq:.0f} Hz (Q {self.q:.2f})"

    def as_dict(self) -> dict:
        return asdict(self)


def build_chain(bands: list[EQBand], fs: float) -> BiquadChain:
    return BiquadChain([b.to_biquad(fs) for b in bands])


def apply_eq(buf: AudioBuffer, bands: list[EQBand]) -> None:
    """Apply an EQ (list of bands) to every channel in place."""
    if not bands:
        return
    fs = buf.sample_rate
    for ch in buf.channels:
        chain = build_chain(bands, fs)  # fresh state per channel
        chain.process_inplace(ch)
