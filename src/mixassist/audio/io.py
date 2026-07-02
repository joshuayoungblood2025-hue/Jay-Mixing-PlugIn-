"""WAV file I/O and stem-folder loading.

A small, dependency-free RIFF/WAVE reader and writer that understands the formats Logic
Pro (and most DAWs) export:

* PCM 8 / 16 / 24 / 32-bit integer
* IEEE 32-bit float
* WAVE_FORMAT_EXTENSIBLE wrapping either of the above

The Python :mod:`wave` stdlib module cannot read float WAVs, so we parse the RIFF
structure ourselves with :mod:`struct`.
"""

from __future__ import annotations

import os
import struct
import sys
from array import array
from pathlib import Path

from mixassist.audio.buffer import AudioBuffer, resample_linear

_INT24_MAX = 8388608.0  # 2**23
_INT16_MAX = 32768.0
_INT32_MAX = 2147483648.0

_FMT_PCM = 0x0001
_FMT_FLOAT = 0x0003
_FMT_EXTENSIBLE = 0xFFFE


class WavError(ValueError):
    """Raised when a WAV file cannot be parsed."""


def _iter_chunks(data: bytes):
    """Yield ``(chunk_id, payload)`` tuples from a RIFF body."""
    pos = 12  # skip 'RIFF' + size + 'WAVE'
    n = len(data)
    while pos + 8 <= n:
        chunk_id = data[pos : pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        start = pos + 8
        end = start + size
        yield chunk_id, data[start:end]
        pos = end + (size & 1)  # chunks are word-aligned


def load_wav(path: str | os.PathLike[str]) -> AudioBuffer:
    """Read a WAV file into an :class:`AudioBuffer` of float64 samples."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise WavError(f"{path}: not a RIFF/WAVE file")

    fmt_tag = None
    num_channels = 0
    sample_rate = 0
    bits = 0
    data_bytes = b""

    for chunk_id, payload in _iter_chunks(raw):
        if chunk_id == b"fmt ":
            (fmt_tag, num_channels, sample_rate, _byte_rate, _block_align, bits) = (
                struct.unpack_from("<HHIIHH", payload, 0)
            )
            if fmt_tag == _FMT_EXTENSIBLE and len(payload) >= 40:
                # Real format lives in the first 2 bytes of the SubFormat GUID.
                (sub_tag,) = struct.unpack_from("<H", payload, 24)
                fmt_tag = sub_tag
        elif chunk_id == b"data":
            data_bytes = payload

    if fmt_tag is None:
        raise WavError(f"{path}: missing fmt chunk")
    if num_channels < 1:
        raise WavError(f"{path}: invalid channel count")

    samples = _decode_samples(data_bytes, fmt_tag, bits, path)
    channels = _deinterleave(samples, num_channels)
    return AudioBuffer(channels, sample_rate)


def _decode_samples(data: bytes, fmt_tag: int, bits: int, path) -> array:
    """Decode raw interleaved bytes into a flat float64 array in [-1, 1]."""
    out = array("d")
    big_endian = sys.byteorder == "big"

    if fmt_tag == _FMT_FLOAT and bits == 32:
        tmp = array("f")
        tmp.frombytes(data[: len(data) - (len(data) % 4)])
        if big_endian:
            tmp.byteswap()
        out.extend(tmp.tolist())
        return out

    if fmt_tag != _FMT_PCM:
        raise WavError(f"{path}: unsupported format tag {fmt_tag} ({bits}-bit)")

    if bits == 8:
        u = array("B")
        u.frombytes(data)
        out.extend((b - 128) / 128.0 for b in u)
    elif bits == 16:
        s = array("h")
        s.frombytes(data[: len(data) - (len(data) % 2)])
        if big_endian:
            s.byteswap()
        inv = 1.0 / _INT16_MAX
        out.extend(v * inv for v in s)
    elif bits == 24:
        inv = 1.0 / _INT24_MAX
        n = len(data) // 3
        mv = memoryview(data)
        for i in range(n):
            b = mv[i * 3 : i * 3 + 3]
            v = int.from_bytes(b, "little", signed=True)
            out.append(v * inv)
    elif bits == 32:
        s = array("i")
        if s.itemsize != 4:  # pragma: no cover - platform dependent
            raise WavError(f"{path}: cannot decode 32-bit PCM on this platform")
        s.frombytes(data[: len(data) - (len(data) % 4)])
        if big_endian:
            s.byteswap()
        inv = 1.0 / _INT32_MAX
        out.extend(v * inv for v in s)
    else:
        raise WavError(f"{path}: unsupported PCM bit depth {bits}")
    return out


def _deinterleave(flat: array, num_channels: int) -> list[array]:
    if num_channels == 1:
        return [flat]
    frames = len(flat) // num_channels
    channels = [array("d", bytes(8 * frames)) for _ in range(num_channels)]
    for c in range(num_channels):
        ch = channels[c]
        for i in range(frames):
            ch[i] = flat[i * num_channels + c]
    return channels


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def save_wav(
    path: str | os.PathLike[str],
    buf: AudioBuffer,
    bit_depth: int = 24,
    float_output: bool = False,
) -> None:
    """Write an :class:`AudioBuffer` to a WAV file.

    Defaults to 24-bit PCM (standard mix/master delivery). Set ``float_output=True`` to
    write 32-bit float and preserve inter-sample values above 0 dBFS.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nch = buf.num_channels
    n = buf.num_frames
    sr = buf.sample_rate
    chans = buf.channels

    if float_output:
        bits = 32
        fmt_tag = _FMT_FLOAT
        body = _encode_float32(chans, n, nch)
    else:
        bits = bit_depth
        fmt_tag = _FMT_PCM
        if bit_depth == 16:
            body = _encode_pcm16(chans, n, nch)
        elif bit_depth == 24:
            body = _encode_pcm24(chans, n, nch)
        elif bit_depth == 32:
            body = _encode_pcm32(chans, n, nch)
        else:
            raise ValueError("bit_depth must be 16, 24, or 32")

    block_align = nch * (bits // 8)
    byte_rate = sr * block_align
    fmt_chunk = struct.pack("<HHIIHH", fmt_tag, nch, sr, byte_rate, block_align, bits)
    data_size = len(body)
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + data_size)

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", len(fmt_chunk)))
        f.write(fmt_chunk)
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(body)
        if data_size & 1:
            f.write(b"\x00")


def _encode_pcm16(chans, n, nch) -> bytes:
    out = array("h", bytes(2 * n * nch))
    limit = _INT16_MAX - 1.0
    for i in range(n):
        base = i * nch
        for c in range(nch):
            v = _clamp(chans[c][i], -1.0, 1.0) * _INT16_MAX
            iv = int(v)
            if iv > limit:
                iv = int(limit)
            elif iv < -_INT16_MAX:
                iv = int(-_INT16_MAX)
            out[base + c] = iv
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()


def _encode_pcm24(chans, n, nch) -> bytes:
    out = bytearray(3 * n * nch)
    pos = 0
    hi = _INT24_MAX - 1.0
    for i in range(n):
        for c in range(nch):
            v = _clamp(chans[c][i], -1.0, 1.0) * _INT24_MAX
            iv = int(v)
            if iv > hi:
                iv = int(hi)
            elif iv < -_INT24_MAX:
                iv = int(-_INT24_MAX)
            out[pos : pos + 3] = (iv & 0xFFFFFF).to_bytes(3, "little")
            pos += 3
    return bytes(out)


def _encode_pcm32(chans, n, nch) -> bytes:
    out = array("i", bytes(4 * n * nch))
    hi = _INT32_MAX - 1.0
    for i in range(n):
        base = i * nch
        for c in range(nch):
            v = _clamp(chans[c][i], -1.0, 1.0) * _INT32_MAX
            iv = int(v)
            if iv > hi:
                iv = int(hi)
            elif iv < -_INT32_MAX:
                iv = int(-_INT32_MAX)
            out[base + c] = iv
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()


def _encode_float32(chans, n, nch) -> bytes:
    out = array("f", bytes(4 * n * nch))
    for i in range(n):
        base = i * nch
        for c in range(nch):
            out[base + c] = chans[c][i]
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()


_AUDIO_EXTS = {".wav", ".wave"}


def load_stems(
    folder: str | os.PathLike[str], target_rate: int | None = None
) -> dict[str, AudioBuffer]:
    """Load every WAV in ``folder`` into ``{stem_name: AudioBuffer}``.

    Stems are keyed by filename without extension. If ``target_rate`` is given (or the
    stems disagree), everything is resampled to a common rate so they can be summed.
    """
    p = Path(folder)
    if not p.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory")

    files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in _AUDIO_EXTS)
    if not files:
        raise FileNotFoundError(f"no .wav stems found in {folder}")

    stems: dict[str, AudioBuffer] = {}
    for f in files:
        stems[f.stem] = load_wav(f)

    rates = {b.sample_rate for b in stems.values()}
    common = target_rate or max(rates)
    if len(rates) > 1 or target_rate is not None:
        stems = {name: resample_linear(b, common) for name, b in stems.items()}
    return stems
