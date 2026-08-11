"""Tests — facette « type de document » (ingest.type_semantique + canal --type-doc au build).

Régimes CI : le test d'injection du token tourne partout (corpus .md/.txt, sans markitdown) ;
le test discriminant multi-types (.html) se saute proprement sans l'extra ingest."""

from pathlib import Path

import pytest

from mosaic.index import Index
from mosaic.ingest import type_semantique

GRID = (32, 32, 3)


def test_type_semantique():
    assert type_semantique(Path("catalogue.xlsx"), "") == "tableur"
    assert type_semantique(Path("photo.jpg"), None) == "photo"
    assert type_semantique(Path("rapport.docx"), "texte") == "document rédigé"
    assert type_semantique(Path("presentation.pptx"), "x") == "présentation"
    assert type_semantique(Path("page.html"), "x") == "page web"
    assert type_semantique(Path("note.txt"), "x") == "note texte"
    # PDF ventilé selon la quantité de texte extraite
    assert type_semantique(Path("scan.pdf"), "abc") == "pdf scanné"
    assert type_semantique(Path("devis.pdf"), "x" * 500) == "pdf numérique"


def test_type_doc_injecte_le_token(tmp_path):
    """Avec type_doc=True, les tokens du type entrent dans le vocabulaire encodé ; sans le
    flag, ils n'y sont pas. Corpus .md/.txt pur (tourne sans markitdown)."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "compte_rendu.md").write_text("disjoncteur pose chantier", encoding="utf-8")
    (c / "specifications.txt").write_text(
        "disjoncteur caracteristiques", encoding="utf-8"
    )

    avec = Index.build(
        c,
        tmp_path / "avec",
        grid=GRID,
        index_paths=False,
        smoothing_rank=0,
        type_doc=True,
    )
    sans = Index.build(
        c,
        tmp_path / "sans",
        grid=GRID,
        index_paths=False,
        smoothing_rank=0,
        type_doc=False,
    )
    assert "note" in set(avec.profiles.rows)  # tokens du type « note texte » injectés
    assert "note" not in set(sans.profiles.rows)  # pas encodés sans le flag


def test_canal_type_doc_rend_le_type_cherchable(tmp_path):
    """Avec --type-doc, une requête par TYPE remonte le bon document parmi des types variés.
    Exige markitdown (le .html doit être ingéré)."""
    pytest.importorskip("markitdown")
    c = tmp_path / "corpus"
    c.mkdir()
    sujet = "disjoncteur tableau electrique protection differentielle"
    (c / "notice.html").write_text(
        f"<html><body>{sujet} installation</body></html>", encoding="utf-8"
    )
    (c / "compte_rendu.md").write_text(f"{sujet} pose chantier", encoding="utf-8")
    (c / "specifications.txt").write_text(f"{sujet} caracteristiques", encoding="utf-8")

    avec = Index.build(
        c,
        tmp_path / "avec",
        grid=GRID,
        index_paths=False,
        smoothing_rank=0,
        type_doc=True,
    )
    r = avec.search("page web notice", k=3)
    assert r[0]["id"] == "notice.html"  # le type « page web » remonte le .html en tête
    assert "web" in set(avec.profiles.rows)
