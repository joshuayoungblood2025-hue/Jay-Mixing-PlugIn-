"""Drum stem separation via Demucs / DrumSep (runs an external ML model).

This does NOT ship a model — professional separation needs a trained neural network
(Demucs + the DrumSep fine-tune), which requires PyTorch and is installed on the user's
machine. This module is a thin, friendly wrapper: it shells out to ``demucs``, then
collects and renames the resulting stems (kick/snare/cymbals/toms — the DrumSep labels are
Spanish) into a folder that ``mixassist mix`` / the web app can consume directly.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Map a separated source's filename stem (lowercased) to a friendly, mixer-ready name.
# Covers the DrumSep (Spanish) labels, common English drum labels, and Demucs full-mix stems.
_LABELS = {
    "bombo": "Kick",
    "kick": "Kick",
    "redoblante": "Snare",
    "snare": "Snare",
    "platillos": "Cymbals",
    "cymbals": "Cymbals",
    "hihat": "HiHat",
    "hh": "HiHat",
    "hi-hat": "HiHat",
    "toms": "Toms",
    "tom": "Toms",
    "ride": "Ride",
    "crash": "Crash",
    # Demucs full-mix sources (when separating a whole song rather than a drum loop):
    "drums": "Drums",
    "bass": "Bass",
    "vocals": "Vocals",
    "other": "Other",
    "guitar": "Guitar",
    "piano": "Piano",
}


def friendly_name(source_stem: str) -> str:
    """Turn a separated source filename stem into a mixer-ready track name."""
    return _LABELS.get(source_stem.strip().lower(), source_stem.strip().capitalize())


def demucs_available() -> bool:
    """True if Demucs looks runnable (module import or console script present)."""
    if shutil.which("demucs") is not None:
        return True
    try:
        import importlib.util

        return importlib.util.find_spec("demucs") is not None
    except (ImportError, ValueError):
        return False


def collect_outputs(search_root: Path, out_stems_dir: Path) -> list[str]:
    """Copy every separated ``.wav`` under ``search_root`` into ``out_stems_dir`` renamed.

    Demucs writes ``<out>/<model>/<track>/<source>.wav``; we flatten those into friendly
    per-instrument files. Returns the list of created stem names.
    """
    out_stems_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for wav in sorted(search_root.rglob("*.wav")):
        name = friendly_name(wav.stem)
        dest = out_stems_dir / f"{name}.wav"
        # de-duplicate if two sources map to the same friendly name
        n = 2
        while dest.exists():
            dest = out_stems_dir / f"{name} {n}.wav"
            n += 1
        shutil.copyfile(wav, dest)
        created.append(dest.stem)
    return created


def separate(
    input_path: str,
    out_dir: str,
    model: str | None = None,
    repo: str | None = None,
    device: str = "cpu",
    timeout: int = 3600,
) -> dict:
    """Run Demucs (optionally the DrumSep model) and organize the stems for the mixer.

    ``repo`` + ``model`` select the DrumSep fine-tune (drum-piece separation). With neither,
    Demucs' default model runs (splits a full song into drums/bass/vocals/other).
    """
    src = Path(input_path)
    if not src.is_file():
        return {"ok": False, "error": f"input not found: {input_path}"}
    if not demucs_available():
        return {
            "ok": False,
            "error": (
                "Demucs is not installed. Install it on this machine with:\n"
                "  pip install demucs\n"
                "and for drum-piece separation, get the DrumSep model (see docs)."
            ),
        }

    out_root = Path(out_dir)
    tmp = out_root / "_demucs_raw"
    tmp.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "demucs", "-o", str(tmp), "-d", device]
    if repo:
        cmd += ["--repo", repo]
    if model:
        cmd += ["-n", model]
    cmd.append(str(src))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"demucs timed out after {timeout}s"}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        return {"ok": False, "error": "demucs failed:\n" + "\n".join(tail), "cmd": " ".join(cmd)}

    stems_dir = out_root / "stems"
    created = collect_outputs(tmp, stems_dir)
    shutil.rmtree(tmp, ignore_errors=True)
    if not created:
        return {"ok": False, "error": "demucs produced no .wav outputs"}
    return {"ok": True, "stems": created, "stems_dir": str(stems_dir)}
