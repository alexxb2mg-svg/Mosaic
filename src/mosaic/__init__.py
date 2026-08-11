"""Mosaic — indexation sémantique par mosaïques hyperdimensionnelles."""

GRID_DEFAULT = (64, 64, 3)

from mosaic.index import Index  # noqa: E402

__all__ = ["GRID_DEFAULT", "Index"]
