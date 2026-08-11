import struct
import zlib

import numpy as np

from mosaic.render import to_png

GRID = (32, 32, 3)


def test_png_signature_et_dimensions():
    vec = np.zeros(32 * 32 * 3, dtype=np.int8)
    png = to_png(vec, GRID)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR : largeur/hauteur aux offsets 16 et 20
    w, h = struct.unpack(">II", png[16:24])
    assert (w, h) == (32, 32)


def test_pixels_reconstructibles():
    rng = np.random.default_rng(7)
    vec = rng.integers(-127, 128, size=32 * 32 * 3, dtype=np.int8)
    png = to_png(vec, GRID)
    # extraire l'IDAT et vérifier le premier pixel (filtre 0 par ligne)
    idat_start = png.index(b"IDAT") + 4
    (idat_len,) = struct.unpack(">I", png[png.index(b"IDAT") - 4 : png.index(b"IDAT")])
    raw = zlib.decompress(png[idat_start : idat_start + idat_len])
    assert raw[0] == 0  # filtre None
    expected = int(vec[0]) + 128
    assert raw[1] == expected
