"""Stereo delay with feedback, feedback-path damping, and optional ping-pong.

Designed as a send effect: feed a mono send, get a wet stereo buffer to blend into the mix.
"""

from __future__ import annotations

from array import array

from mixassist.audio.buffer import AudioBuffer


class StereoDelay:
    def __init__(
        self,
        sample_rate: int,
        time_ms: float = 350.0,
        feedback: float = 0.35,
        damping: float = 0.3,
        ping_pong: bool = True,
    ) -> None:
        self.sr = sample_rate
        self.delay = max(1, int(time_ms * 0.001 * sample_rate))
        self.feedback = max(0.0, min(0.95, feedback))
        # one-pole lowpass coefficient in the feedback path (higher damping = darker repeats)
        self.damp = max(0.0, min(0.95, damping))
        self.ping_pong = ping_pong

    def process_send(self, mono_send: array) -> AudioBuffer:
        n = len(mono_send)
        d = self.delay
        fb = self.feedback
        damp = self.damp
        line_l = array("d", bytes(8 * (n + d + 1)))
        line_r = array("d", bytes(8 * (n + d + 1)))
        out_l = array("d", bytes(8 * n))
        out_r = array("d", bytes(8 * n))
        lp_l = 0.0
        lp_r = 0.0

        for i in range(n):
            # delayed taps (already include earlier feedback writes)
            dl = line_l[i]
            dr = line_r[i]
            out_l[i] = dl
            out_r[i] = dr
            # damp the feedback signal (one-pole lowpass)
            lp_l = dl * (1.0 - damp) + lp_l * damp
            lp_r = dr * (1.0 - damp) + lp_r * damp
            wi = i + d
            x = mono_send[i]
            if self.ping_pong:
                # input hits the left tap; repeats bounce L->R->L
                line_l[wi] += x + lp_r * fb
                line_r[wi] += lp_l * fb
            else:
                line_l[wi] += x + lp_l * fb
                line_r[wi] += x + lp_r * fb
        return AudioBuffer([out_l, out_r], self.sr)


def note_ms(bpm: float, note: str = "1/8") -> float:
    """Milliseconds for a musical note at ``bpm`` (e.g. '1/4', '1/8', '1/8.', '1/8T')."""
    table = {"1/4": 1.0, "1/8": 0.5, "1/16": 0.25, "1/2": 2.0}
    base = table.get(note.rstrip(".T"), 0.5)
    beat_ms = 60000.0 / max(1e-6, bpm)
    ms = beat_ms * base
    if note.endswith("."):
        ms *= 1.5  # dotted
    elif note.endswith("T"):
        ms *= 2.0 / 3.0  # triplet
    return ms
