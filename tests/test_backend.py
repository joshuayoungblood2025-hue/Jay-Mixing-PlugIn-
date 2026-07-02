"""Numeric backend seam: selection, fallback, and correctness."""

from __future__ import annotations

from array import array

from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp.backend import PurePythonBackend, get_backend, set_backend


def teardown_function() -> None:
    set_backend(None)  # restore auto-selection after each test


def test_auto_select_returns_a_backend():
    set_backend(None)
    b = get_backend()
    assert b.name in ("pure-python", "numpy")
    # In this environment numpy is unavailable, so it must fall back.
    assert b.name == "pure-python"


def test_pure_backend_ops():
    b = PurePythonBackend()
    a = array("d", [0.1, -0.5, 0.25])
    b.apply_gain(a, 2.0)
    assert a[0] == 0.2 and a[1] == -1.0 and a[2] == 0.5

    d = array("d", [1.0, 1.0, 1.0])
    b.mix_add(d, array("d", [0.5, 0.5, 0.5]), 2.0)
    assert list(d) == [2.0, 2.0, 2.0]

    assert b.peak(array("d", [0.2, -0.9, 0.3])) == 0.9
    assert b.peak(array("d", [])) == 0.0


def test_apply_gain_identity_is_noop():
    b = PurePythonBackend()
    a = array("d", [0.3, -0.7])
    b.apply_gain(a, 1.0)
    assert list(a) == [0.3, -0.7]


def test_set_backend_override():
    set_backend(PurePythonBackend())
    assert get_backend().name == "pure-python"


def test_audiobuffer_uses_backend():
    buf = AudioBuffer([array("d", [0.25] * 8), array("d", [-0.5] * 8)], 44100)
    assert buf.peak() == 0.5
    buf.apply_gain(2.0)
    assert buf.peak() == 1.0
