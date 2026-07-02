"""Creative FX: reverb, delay, saturation, and their engine integration."""

from __future__ import annotations

import math
from array import array

from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp.delay import StereoDelay, note_ms
from mixassist.dsp.fft import power_spectrum
from mixassist.dsp.reverb import Reverb
from mixassist.dsp.saturation import saturate
from mixassist.mixing.engine import MixSettings, mix


def _impulse(n: int) -> array:
    a = array("d", bytes(8 * n))
    a[0] = 1.0
    return a


def test_reverb_decays_and_is_stereo():
    sr = 44100
    wet = Reverb(sr, room_size=0.8, damping=0.4).process_send(_impulse(sr))

    def energy(a, s, e):
        return sum(a[i] * a[i] for i in range(s, e))

    early = energy(wet.channels[0], 2000, 6000)
    late = energy(wet.channels[0], 20000, 24000)
    assert early > 0.0
    assert late < early  # tail decays
    assert any(abs(wet.channels[0][i] - wet.channels[1][i]) > 1e-9 for i in range(1000, 3000))


def test_delay_taps_on_time():
    sr = 44100
    wet = StereoDelay(sr, time_ms=100.0, feedback=0.5, damping=0.0, ping_pong=False).process_send(
        _impulse(sr)
    )
    taps = [i for i in range(sr) if abs(wet.channels[0][i]) > 0.2]
    assert taps[:3] == [4410, 8820, 13230]  # multiples of 0.1 s


def test_pingpong_offsets_right_channel():
    sr = 44100
    wet = StereoDelay(sr, time_ms=100.0, feedback=0.5, damping=0.0, ping_pong=True).process_send(
        _impulse(sr)
    )
    first_l = next(i for i in range(sr) if abs(wet.channels[0][i]) > 0.2)
    first_r = next(i for i in range(sr) if abs(wet.channels[1][i]) > 0.2)
    assert first_r > first_l


def test_note_ms():
    assert abs(note_ms(120, "1/8") - 250.0) < 1e-6
    assert abs(note_ms(120, "1/4") - 500.0) < 1e-6


def test_saturation_adds_odd_harmonics_and_is_bounded():
    sr = 44100
    clean = array("d", [0.5 * math.sin(2 * math.pi * 440 * i / sr) for i in range(sr)])
    buf = AudioBuffer([array("d", clean)], sr)
    saturate(buf, drive=0.8, mix=1.0)
    assert max(abs(x) for x in buf.channels[0]) <= 1.0
    f, p = power_spectrum(buf.channels[0], sr, fft_size=8192)

    def bin_at(freq):
        return min(range(len(f)), key=lambda k: abs(f[k] - freq))

    assert p[bin_at(1320)] / p[bin_at(440)] > 1e-3  # 3rd harmonic present


def test_engine_fx_changes_output_and_stays_safe(synth_stems):
    dry = mix(synth_stems, MixSettings(genre="pop", reverb=0.0, delay=0.0, drive=0.0))
    wet = mix(synth_stems, MixSettings(genre="pop", reverb=0.5, delay=0.3, drive=0.2))
    n = min(dry.master.num_frames, wet.master.num_frames)

    assert wet.bus_plan.reverb_amount == 0.5
    assert wet.bus_plan.delay_amount == 0.3
    assert wet.bus_plan.drive_amount == 0.2
    # the FX audibly change the rendered master (not a no-op)
    diff = max(abs(wet.master.channels[0][i] - dry.master.channels[0][i]) for i in range(0, n, 7))
    assert diff > 1e-3
    # and the safety limiter still holds the ceiling
    assert wet.bus_plan.final_peak_dbfs <= wet.bus_plan.peak_ceiling_db + 0.1


def test_fx_off_by_default(synth_stems):
    res = mix(synth_stems, MixSettings(genre="pop"))
    assert res.bus_plan.reverb_amount == 0.0
    assert res.bus_plan.delay_amount == 0.0
    assert res.bus_plan.drive_amount == 0.0
