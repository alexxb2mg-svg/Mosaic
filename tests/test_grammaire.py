"""Canal grammatical (--grammatical) : build/open/add, gardes, séparation de bout en
bout des clauses à mots identiques et sens opposé. Cf. brief 12/08 + P1-P3 mesurés
(research/canal_grammatical.py)."""

import subprocess
import sys
from pathlib import Path

import pytest

from mosaic.grammaire import analyse
from mosaic.index import Index

PY = [sys.executable, "-m", "mosaic.cli"]
GRID = (32, 32, 3)


def _corpus(tmp_path: Path) -> Path:
    c = tmp_path / "corpus"
    c.mkdir(exist_ok=True)
    (c / "conforme.md").write_text(
        "Le disjoncteur se place en amont du différentiel pour la protection.",
        encoding="utf-8",
    )
    (c / "inverse.md").write_text(
        "Le différentiel se place en amont du disjoncteur pour la protection.",
        encoding="utf-8",
    )
    (c / "peinture.md").write_text(
        "Peinture du couloir et de l'escalier au rouleau.", encoding="utf-8"
    )
    return c


def test_analyse_paire_canonique_roles_opposes():
    a = analyse("le disjoncteur en amont du différentiel")
    b = analyse("le différentiel en amont du disjoncteur")
    assert ("amont", "disjoncteur") in a and ("amont", "différentiel") in b


def test_analyse_abstention_piege():
    assert analyse("les poules du couvent couvent") == []


def test_build_cree_le_canal_et_stats(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, grammatical=True)
    assert (tmp_path / "idx" / "docs_grammatical.msei").is_file()
    assert idx.stats()["grammatical"]["docs_avec_roles"] == 2  # peinture s'abstient


def test_sans_canal_impact_nul_et_refus_recherche(tmp_path):
    a = Index.build(_corpus(tmp_path), tmp_path / "a", grid=GRID)
    b = Index.build(_corpus(tmp_path), tmp_path / "b", grid=GRID, grammatical=True)
    import numpy as np

    assert np.array_equal(a.mat, b.mat), "le canal séparé ne touche pas la grille"
    with pytest.raises(ValueError, match="--grammatical"):
        a.search("amont", k=1, grammatical=True)


def test_separation_de_bout_en_bout(tmp_path):
    """La paire amont/aval à mots identiques : le moteur nu la confond, le canal la
    sépare — la requête structurale « disjoncteur en amont » doit classer la clause
    conforme DEVANT son inverse."""
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, grammatical=True)
    nu = idx.search("le disjoncteur en amont du différentiel", k=2)
    assert {h["id"] for h in nu[:2]} == {"conforme.md", "inverse.md"}
    gram = idx.search("le disjoncteur en amont du différentiel", k=2, grammatical=True)
    assert gram[0]["id"] == "conforme.md"
    assert gram[0]["score_grammatical"] > gram[1]["score_grammatical"]


def test_recherche_identique_apres_reouverture(tmp_path):
    Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, grammatical=True)
    idx = Index.open(tmp_path / "idx")
    avant = idx.search("disjoncteur en amont", k=3, grammatical=True)
    assert avant and "score_grammatical" in avant[0]


def test_add_analyse_le_nouveau_document(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, grammatical=True)
    nouveau = tmp_path / "sans_terre.md"
    nouveau.write_text("Circuit sans conducteur de terre.", encoding="utf-8")
    idx.add(nouveau)
    ré = Index.open(tmp_path / "idx")
    assert ré.stats()["grammatical"]["docs_avec_roles"] == 3


def test_cli_gardes(tmp_path):
    corpus = _corpus(tmp_path)
    idx = tmp_path / "idx"
    r = subprocess.run(
        [*PY, "build", str(corpus), "-o", str(idx), "--grammatical"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    r = subprocess.run(
        [*PY, "search", "amont", str(idx), "--grammatical", "--rerank"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 1 and "exclusif" in r.stderr
