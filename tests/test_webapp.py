"""Local web app: multipart parsing and the mix pipeline behind /mix."""

from __future__ import annotations

from mixassist.audio.io import save_wav
from mixassist.webapp import _run_mix, parse_multipart


def _multipart(files: list[tuple[str, bytes]], fields: dict[str, str]) -> tuple[bytes, bytes]:
    boundary = b"----testBOUNDARY42"
    parts: list[bytes] = []
    for name, content in files:
        parts.append(b"--" + boundary + b"\r\n")
        parts.append(
            (
                f'Content-Disposition: form-data; name="stems"; filename="{name}"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
            ).encode()
        )
        parts.append(content)
        parts.append(b"\r\n")
    for k, v in fields.items():
        parts.append(b"--" + boundary + b"\r\n")
        parts.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(b"--" + boundary + b"--\r\n")
    return b"".join(parts), boundary


def test_parse_multipart_files_and_fields():
    body, boundary = _multipart(
        [("Kick.wav", b"\x00\x01\x02"), ("Bass.wav", b"\x03\x04")],
        {"genre": "pop", "intensity": "0.6"},
    )
    fields, files = parse_multipart(body, boundary)
    assert fields == {"genre": "pop", "intensity": "0.6"}
    assert [n for n, _ in files] == ["Kick.wav", "Bass.wav"]
    assert files[0][1] == b"\x00\x01\x02"
    assert files[1][1] == b"\x03\x04"


def test_run_mix_end_to_end(tmp_path, synth_stems):
    # Serialize the synthetic stems to WAV bytes, then feed them through the /mix pipeline.
    files: list[tuple[str, bytes]] = []
    for name, buf in synth_stems.items():
        p = tmp_path / f"{name}.wav"
        save_wav(p, buf, bit_depth=16)
        files.append((f"{name}.wav", p.read_bytes()))

    runs = tmp_path / "runs"
    fields = {"genre": "pop", "intensity": "0.6", "vocal": "0.7", "tone": "0.1"}
    res = _run_mix(runs, fields, files)
    assert res["ok"] is True
    mixed = runs / res["id"] / "mixed"
    assert (mixed / "master.wav").exists()
    assert (mixed / "dashboard.html").exists()
    assert (mixed / "mixed_stems.zip").exists()  # processed stems returned as a zip
    assert "<svg" in (mixed / "dashboard.html").read_text()


def test_run_mix_rejects_no_files(tmp_path):
    res = _run_mix(tmp_path / "runs", {"genre": "pop"}, [])
    assert res["ok"] is False


def test_run_mix_rejects_non_wav(tmp_path):
    res = _run_mix(tmp_path / "runs", {"genre": "pop"}, [("notes.txt", b"hello")])
    assert res["ok"] is False
