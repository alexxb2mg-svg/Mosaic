"""Rendu PNG d'une mosaïque — stdlib uniquement (zlib + struct)."""

import struct
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mosaic.index import Index


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload))
    )


def to_png(vec: np.ndarray, grid: tuple[int, int, int]) -> bytes:
    w, h, _c = grid
    img = (vec.reshape(h, w, 3).astype(np.int16) + 128).astype(np.uint8)
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # RGB 8 bits
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


def render_doc(index: "Index", doc_id: str, out: Path) -> None:
    row = index.ids.index(doc_id)
    Path(out).write_bytes(to_png(index.mat[row], index.grid))
