"""Tests — algèbre de requête par connecteurs (mosaic search --connecteurs)."""

from pathlib import Path

from mosaic.connecteurs import analyser, decouper
from mosaic.index import Index

GRID = (32, 32, 3)


def test_analyser_signes():
    assert analyser("disjoncteur sans différentiel") == [
        ("disjoncteur", 1),
        ("différentiel", -1),
    ]
    # « et » garde le signe ; « mais pas » bascule en négatif
    assert decouper("tableau et parafoudre mais pas variateur") == (
        "tableau parafoudre",
        "variateur",
    )
    # « mais » seul re-positive
    assert decouper("pas a mais b") == ("b", "a")
    # sans marqueur de négation : négatif vide
    assert decouper("a et b")[1] == ""


def _corpus(tmp_path: Path) -> Path:
    c = tmp_path / "corpus"
    c.mkdir(parents=True, exist_ok=True)
    (c / "chat.md").write_text(
        "chat félin animal domestique compagnie ronron", encoding="utf-8"
    )
    (c / "chien.md").write_text(
        "chien canin animal domestique compagnie aboiement", encoding="utf-8"
    )
    return c


def test_sans_negatif_equivaut_a_search(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    a = idx.search_connecteurs("animal domestique", k=2)
    b = idx.search("animal domestique", k=2)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_exclusion_fait_descendre_le_document(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    res = idx.search_connecteurs("animal domestique compagnie sans chien", k=2)
    rangs = {r["id"]: i for i, r in enumerate(res)}
    # le document « chien » doit passer SOUS le document « chat »
    assert rangs["chat.md"] < rangs["chien.md"]
    chien = next(r for r in res if r["id"] == "chien.md")
    assert chien["negatif"] > 0.0  # le doc « chien » est bien pénalisé
