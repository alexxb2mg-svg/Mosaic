"""Pont bilingue : canonicalisation des tokens anglais vers leur pivot français."""

import json
from pathlib import Path

from mosaic.tokenize import STOPWORDS

DEFAULT_LEXICON = Path(__file__).resolve().parent / "data" / "lexicon_fr_en.json"


def _read_json_lexicon(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k).lower(): str(v).lower() for k, v in data.items()}


def load_lexicon(path: Path | None = None, extra: Path | None = None) -> dict[str, str]:
    """Charge le noyau (`path` ou `DEFAULT_LEXICON`) et fusionne `extra` en dessous : le noyau gagne."""
    core = _read_json_lexicon(Path(path) if path is not None else DEFAULT_LEXICON)
    if extra is None:
        return core
    merged = _read_json_lexicon(Path(extra))
    merged.update(core)  # le noyau curaté gagne sur les collisions
    return merged


def compile_lexicon(lexicon: dict[str, str]) -> dict[tuple[str, ...], list[str]]:
    """Découpe clés/valeurs en séquences de tokens : clé `_` → tuple, valeur `_` → liste FR sans stopwords."""
    compiled: dict[tuple[str, ...], list[str]] = {}
    for key, value in lexicon.items():
        key_tokens = tuple(key.lower().split("_"))
        if any(t in STOPWORDS for t in key_tokens):
            continue  # garde structurelle : un stopword FR ne doit jamais devenir un mot de contenu
        value_tokens = [t for t in value.lower().split("_") if t not in STOPWORDS]
        if not value_tokens:
            continue  # valeur sans mot plein : mapping inutilisable, ne pas avaler le contenu
        compiled[key_tokens] = value_tokens
    return compiled


def canonicalize(
    tokens: list[str], compiled: dict[tuple[str, ...], list[str]], max_len: int = 4
) -> list[str]:
    """Balayage glouton : essaie les fenêtres de max_len à 1, la clé la plus longue gagne."""
    if not compiled:
        return list(tokens)
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        for length in range(min(max_len, n - i), 0, -1):
            window = tuple(tokens[i : i + length])
            replacement = compiled.get(window)
            if replacement is not None:
                out.extend(replacement)
                i += length
                break
        else:
            out.append(tokens[i])
            i += 1
    return out
