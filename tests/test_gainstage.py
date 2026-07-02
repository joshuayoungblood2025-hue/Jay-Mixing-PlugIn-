"""Industry-standard input gain staging (~-18 dBFS RMS, peaks <= -6) and intentional panning."""

from __future__ import annotations

import math
from array import array

from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp.loudness import peak_dbfs, rms_dbfs
from mixassist.mixing.engine import MixSettings, mix


def _tone(freq: float, sr: int, n: int, amp: float) -> array:
    return array("d", [amp * math.sin(2 * math.pi * freq * i / sr) for i in range(n)])


def test_quiet_track_staged_up_to_minus18_rms():
    sr = 44100
    n = sr
    raw = _tone(300.0, sr, n, 0.02)  # quiet source (~-36 dBFS RMS)
    rms0 = rms_dbfs(raw)
    pk0 = peak_dbfs(AudioBuffer([array("d", raw)], sr))
    res = mix({"Lead Vocal": AudioBuffer([array("d", raw)], sr)}, MixSettings(master=False))
    stage = res.track_plans[0].input_stage_db
    assert rms0 + stage <= -17.9  # never hotter than -18 dBFS RMS
    assert pk0 + stage <= -5.9  # peaks never above -6 dBFS
    assert abs((rms0 + stage) - (-18.0)) < 0.6  # a quiet track lands on -18 RMS


def test_hot_track_staged_down():
    sr = 44100
    n = sr
    raw = _tone(80.0, sr, n, 0.9)  # hot source
    rms0 = rms_dbfs(raw)
    res = mix({"Bass": AudioBuffer([array("d", raw)], sr)}, MixSettings(master=False))
    stage = res.track_plans[0].input_stage_db
    assert stage < 0.0  # pulled down
    assert rms0 + stage <= -17.9


def test_peak_cap_wins_for_transients():
    sr = 44100
    n = sr
    raw = array("d", bytes(8 * n))
    for k in range(0, n, sr // 4):  # sparse full-scale clicks: high peak, low RMS
        raw[k] = 0.95
    pk0 = peak_dbfs(AudioBuffer([array("d", raw)], sr))
    res = mix({"Perc": AudioBuffer([array("d", raw)], sr)}, MixSettings(master=False))
    stage = res.track_plans[0].input_stage_db
    assert pk0 + stage <= -5.9  # peak ceiling (-6) takes precedence over the RMS target


def test_panning_centers_anchors_and_spreads_others(synth_stems):
    res = mix(synth_stems, MixSettings(master=False))
    pans = {p.name: p.pan for p in res.track_plans}
    assert pans["Kick"] == 0.0
    assert pans["Bass"] == 0.0
    assert pans["Lead Vocal"] == 0.0
    assert abs(pans["Pad"]) > 0.0  # pad is spread off-center
    assert abs(pans["Hats"]) > 0.0  # hats placed off-center
