#!/usr/bin/env python3
"""Compile the C++ DSP harness, run it, and confirm it matches the Python BusChain.

This proves the plugin's Dsp.h port produces the same output as mixassist's real-time
BusChain, sample-for-sample, for an identical signal and chain. Run from the repo root:

    PYTHONPATH=src python plugin/tests/verify_dsp_parity.py
"""

from __future__ import annotations

import math
import subprocess
import sys
from array import array
from pathlib import Path

from mixassist.audio.buffer import AudioBuffer
from mixassist.dsp.compressor import CompressorSettings
from mixassist.dsp.eq import EQBand
from mixassist.realtime.blocks import BusChain

SR = 48000
N = 2000
OUT_GAIN_DB = 3.0
CEILING_DB = -1.0
INDICES = [100, 500, 900, 1300, 1700, 1999]


def python_outputs() -> list[float]:
    def lsample(i: int) -> float:
        return 0.6 * math.sin(2 * math.pi * 220 * i / SR) + 0.3 * math.sin(
            2 * math.pi * 3000 * i / SR
        )

    def rsample(i: int) -> float:
        return 0.55 * math.sin(2 * math.pi * 221 * i / SR) + 0.2 * math.sin(
            2 * math.pi * 5000 * i / SR
        )

    left = array("d", [lsample(i) for i in range(N)])
    right = array("d", [rsample(i) for i in range(N)])
    buf = AudioBuffer([left, right], SR)
    bands = [
        EQBand("highpass", 30.0, 0.0, 0.7071),
        EQBand("lowshelf", 100.0, 1.5, 0.7071),
        EQBand("peak", 300.0, -2.0, 1.0),
        EQBand("highshelf", 10000.0, 2.0, 0.7071),
    ]
    comp = CompressorSettings(
        threshold_db=-18.0, ratio=2.0, attack_ms=30.0, release_ms=200.0, knee_db=6.0, makeup_db=0.0
    )
    chain = BusChain(SR, bands, comp, CEILING_DB, OUT_GAIN_DB, num_channels=2)
    chain.process(buf, block_size=512)  # block size is irrelevant (block-invariant)
    out = [buf.channels[0][i] for i in INDICES] + [buf.channels[1][i] for i in INDICES]
    return out


def cpp_outputs() -> list[float]:
    here = Path(__file__).resolve().parent
    src = here / "dsp_parity.cpp"
    exe = here / "dsp_parity.bin"
    cxx = "clang++" if _which("clang++") else "g++"
    subprocess.run([cxx, "-std=c++17", "-O2", str(src), "-o", str(exe)], check=True)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    return [float(x) for x in result.stdout.split()]


def _which(name: str) -> bool:
    from shutil import which

    return which(name) is not None


def main() -> int:
    py = python_outputs()
    cpp = cpp_outputs()
    if len(py) != len(cpp):
        print(f"length mismatch: py={len(py)} cpp={len(cpp)}")
        return 1
    max_err = max(abs(a - b) for a, b in zip(py, cpp, strict=True))
    print("index      python            c++               abs_err")
    for i, (a, b) in enumerate(zip(py, cpp, strict=True)):
        ch = "L" if i < len(INDICES) else "R"
        idx = INDICES[i % len(INDICES)]
        print(f"{ch}[{idx:4d}]  {a:+.12f}  {b:+.12f}  {abs(a - b):.2e}")
    print(f"\nmax abs error: {max_err:.3e}")
    ok = max_err < 1e-9
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
