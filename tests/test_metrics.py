"""Master metrics: stereo correlation, width, loudness time-series, PLR."""

from __future__ import annotations

import math
from array import array

from tests.conftest import sine

from mixassist.analysis.metrics import compute_master_metrics
from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp.loudness import loudness_time_series


def test_correlation_identical_channels(sr):
    s = sine(300.0, sr, 1.0, amp=0.5)
    buf = AudioBuffer([s, array("d", s)], sr)
    m = compute_master_metrics(buf)
    assert m.stereo_correlation > 0.99


def test_correlation_inverted_channels(sr):
    s = sine(300.0, sr, 1.0, amp=0.5)
    buf = AudioBuffer([s, array("d", [-x for x in s])], sr)
    m = compute_master_metrics(buf)
    assert m.stereo_correlation < -0.9


def test_mono_signal_has_low_width(sr):
    s = sine(300.0, sr, 1.0, amp=0.5)
    buf = AudioBuffer([s, array("d", s)], sr)
    m = compute_master_metrics(buf)
    assert m.width_overall < 0.05


def test_plr_is_peak_minus_lufs(sr):
    s = sine(1000.0, sr, 2.0, amp=0.3)
    buf = AudioBuffer([s, array("d", s)], sr)
    m = compute_master_metrics(buf)
    assert math.isclose(m.plr_db, m.peak_dbfs - m.integrated_lufs, abs_tol=0.05)


def test_loudness_time_series_multiple_points(sr):
    # 6 s signal, 3 s window, 0.5 s hop -> several points
    s = sine(1000.0, sr, 6.0, amp=0.3)
    buf = AudioBuffer([s], sr)
    series = loudness_time_series(buf, window_s=3.0, hop_s=0.5)
    assert len(series) >= 5
    assert all(math.isfinite(v) for _, v in series)


def test_short_signal_single_point(sr):
    s = sine(1000.0, sr, 1.0, amp=0.3)
    buf = AudioBuffer([s], sr)
    series = loudness_time_series(buf, window_s=3.0, hop_s=0.5)
    assert len(series) == 1
