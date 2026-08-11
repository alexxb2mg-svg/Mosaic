"""Lecteurs de fichiers pour l'indexation : texte brut, tokenisation, conversion markitdown."""

import re
from pathlib import Path

from mosaic import ingest
from mosaic.tokenize import tokenize

_EXTS = {".md", ".txt"}

# v1.5 : tokens de chemin. Séparateurs `/ _ - .` remplacés par des espaces puis
# tokenisation standard (extension incluse) — l'IDF neutralise ce qui est
# partout (ex. "md"/"pdf" sur tout un corpus), mais rend cherchable ce qui ne
# vivait que dans le nom/chemin du document (noms de clients, de fournisseurs, de projets...).
_PATH_TOKEN_SEP_RE = re.compile(r"[/_\-.]")


def _path_tokens(rel_path: str) -> list[str]:
    return tokenize(_PATH_TOKEN_SEP_RE.sub(" ", rel_path))


def _read_text(file: Path) -> str:
    return file.read_text(encoding="utf-8", errors="replace")


def _read_tokens(file: Path) -> list[str]:
    return tokenize(_read_text(file))


def _read_text_convertible(
    file: Path, cache_dir: Path | None, ocr: bool = False
) -> str | None:
    """Fichier CONVERTIBLE_EXTS : conversion markitdown en mémoire (prisme de lecture,
    rien n'est persisté). None si markitdown est indisponible ou la conversion échoue —
    l'appelant compte alors le fichier dans les ignorés, jamais de crash du build.

    `ocr` (v1.5) : relayé à `ingest.to_text` — document muet (< 200 caractères) +
    `ocr=True` -> crochet OCR enfichable (peut lever ValueError sans provider).

    Photos et images (v1.6 §B) : jamais soumises à la garde `ingest.available()`
    (markitdown) — une image n'a aucune couche texte markitdown, `ingest.to_text`
    la traite entièrement via sa branche OCR dédiée, avec ou sans markitdown."""
    if file.suffix.lower() not in ingest.IMAGE_EXTS and not ingest.available():
        return None
    return ingest.to_text(file, cache_dir=cache_dir, ocr=ocr)


def _read_tokens_convertible(file: Path, cache_dir: Path | None) -> list[str] | None:
    text = _read_text_convertible(file, cache_dir)
    if text is None:
        return None
    return tokenize(text)
