"""Integration test: the C++ plugin DSP port matches the Python engine.

Compiles plugin/tests/dsp_parity.cpp with an available C++ compiler and checks it agrees
with mixassist's real-time BusChain sample-for-sample. Skipped automatically where no C++
compiler is present (the parity is then covered by CI's macOS/ubuntu build).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from shutil import which

import pytest

_HAS_CXX = which("clang++") is not None or which("g++") is not None

pytestmark = pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available")


def _load_verifier():
    path = Path(__file__).resolve().parents[1] / "plugin" / "tests" / "verify_dsp_parity.py"
    spec = importlib.util.spec_from_file_location("verify_dsp_parity", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_dsp_parity"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cpp_dsp_matches_python_engine():
    v = _load_verifier()
    py = v.python_outputs()
    cpp = v.cpp_outputs()
    assert len(py) == len(cpp)
    max_err = max(abs(a - b) for a, b in zip(py, cpp, strict=True))
    assert max_err < 1e-9, f"C++/Python DSP diverged by {max_err:.2e}"
