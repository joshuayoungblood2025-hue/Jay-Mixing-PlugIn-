"""Algorithmic reverb (Freeverb / Schroeder-Moorer).

A classic, well-proven algorithm: a bank of damped comb filters in parallel feeding a chain
of all-pass filters, run as two channels with a stereo spread. It produces a smooth, musical
tail suitable for a mix reverb send. Controls: room size, damping, width, pre-delay, mix.

Pure standard library. Feedback runs per sample, so rendering long signals is not fast;
the numeric-backend seam is the intended path to speed this up later.
"""

from __future__ import annotations

from array import array

from mixassist.audio.buffer import AudioBuffer

# Freeverb tunings are defined at 44.1 kHz; scaled for other rates.
_COMB_TUNINGS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_ALLPASS_TUNINGS = (556, 441, 341, 225)
_STEREO_SPREAD = 23
_REF_SR = 44100

_SCALE_ROOM = 0.28
_OFFSET_ROOM = 0.7
_SCALE_DAMP = 0.4
_FIXED_GAIN = 0.015
_ALLPASS_FEEDBACK = 0.5


class _Comb:
    __slots__ = ("buf", "idx", "store", "feedback", "damp1", "damp2")

    def __init__(self, size: int, feedback: float, damp: float) -> None:
        self.buf = array("d", bytes(8 * max(1, size)))
        self.idx = 0
        self.store = 0.0
        self.feedback = feedback
        self.damp1 = damp
        self.damp2 = 1.0 - damp

    def process(self, x: float) -> float:
        out = self.buf[self.idx]
        self.store = out * self.damp2 + self.store * self.damp1
        self.buf[self.idx] = x + self.store * self.feedback
        self.idx += 1
        if self.idx >= len(self.buf):
            self.idx = 0
        return out


class _Allpass:
    __slots__ = ("buf", "idx", "feedback")

    def __init__(self, size: int, feedback: float) -> None:
        self.buf = array("d", bytes(8 * max(1, size)))
        self.idx = 0
        self.feedback = feedback

    def process(self, x: float) -> float:
        bufout = self.buf[self.idx]
        out = -x + bufout
        self.buf[self.idx] = x + bufout * self.feedback
        self.idx += 1
        if self.idx >= len(self.buf):
            self.idx = 0
        return out


class Reverb:
    """Stereo Freeverb. Feed a mono (or stereo) send; get a wet stereo buffer."""

    def __init__(
        self,
        sample_rate: int,
        room_size: float = 0.7,
        damping: float = 0.5,
        width: float = 1.0,
        pre_delay_ms: float = 12.0,
    ) -> None:
        self.sr = sample_rate
        scale = sample_rate / _REF_SR
        feedback = room_size * _SCALE_ROOM + _OFFSET_ROOM
        damp1 = damping * _SCALE_DAMP
        self.width = max(0.0, min(1.0, width))
        self.pre_delay = max(0, int(pre_delay_ms * 0.001 * sample_rate))

        self._combs_l = [_Comb(int(t * scale), feedback, damp1) for t in _COMB_TUNINGS]
        self._combs_r = [
            _Comb(int((t + _STEREO_SPREAD) * scale), feedback, damp1) for t in _COMB_TUNINGS
        ]
        self._aps_l = [_Allpass(int(t * scale), _ALLPASS_FEEDBACK) for t in _ALLPASS_TUNINGS]
        self._aps_r = [
            _Allpass(int((t + _STEREO_SPREAD) * scale), _ALLPASS_FEEDBACK) for t in _ALLPASS_TUNINGS
        ]

    def process_send(self, mono_send: array) -> AudioBuffer:
        """Process a mono send signal into a wet stereo :class:`AudioBuffer`."""
        n = len(mono_send)
        pd = self.pre_delay
        # pre-delay via a simple ring buffer
        pre = array("d", bytes(8 * (pd + 1))) if pd > 0 else None
        pre_idx = 0

        left = array("d", bytes(8 * n))
        right = array("d", bytes(8 * n))
        combs_l, combs_r = self._combs_l, self._combs_r
        aps_l, aps_r = self._aps_l, self._aps_r

        for i in range(n):
            x = mono_send[i]
            if pre is not None:
                delayed = pre[pre_idx]
                pre[pre_idx] = x
                pre_idx += 1
                if pre_idx >= len(pre):
                    pre_idx = 0
                x = delayed
            inp = x * _FIXED_GAIN
            outl = 0.0
            outr = 0.0
            for c in combs_l:
                outl += c.process(inp)
            for c in combs_r:
                outr += c.process(inp)
            for a in aps_l:
                outl = a.process(outl)
            for a in aps_r:
                outr = a.process(outr)
            left[i] = outl
            right[i] = outr

        # apply width (mid/side blend)
        w1 = self.width * 0.5 + 0.5
        w2 = (1.0 - self.width) * 0.5
        out_l = array("d", bytes(8 * n))
        out_r = array("d", bytes(8 * n))
        for i in range(n):
            lv = left[i]
            rv = right[i]
            out_l[i] = lv * w1 + rv * w2
            out_r[i] = rv * w1 + lv * w2
        return AudioBuffer([out_l, out_r], self.sr)
