"""Tests — vérité temporelle (src/mosaic/temporel.py) : la version la plus récente d'un aspect
est canonique, les précédentes sont marquées périmées ; les aspects distincts ne fusionnent pas."""

from mosaic.index import Index
from mosaic.temporel import date_du_chemin, versions_actuelles

GRID = (32, 32, 3)


def test_date_du_chemin():
    assert date_du_chemin("2025-05-12_eclairage_v3.md") == "2025-05-12"
    assert date_du_chemin("sous/dossier/2024-01-02_note.md") == "2024-01-02"
    assert date_du_chemin("sans_date.md") == "0000-00-00"


def test_version_recente_canonique(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    # aspect ÉCLAIRAGE : 3 versions datées (vocabulaire partagé -> regroupées)
    (c / "2025-01-15_eclairage_v1.md").write_text(
        "éclairage spots LED encastrés 3000K blanc chaud bureaux circulation gradation",
        encoding="utf-8",
    )
    (c / "2025-03-10_eclairage_v2.md").write_text(
        "éclairage spots LED encastrés 4000K blanc neutre bureaux détection présence",
        encoding="utf-8",
    )
    (c / "2025-05-12_eclairage_v3.md").write_text(
        "éclairage spots LED encastrés 3500K bureaux 4000K circulation gradation DALI",
        encoding="utf-8",
    )
    # aspect TABLEAU : distinct, ne doit PAS fusionner avec l'éclairage
    (c / "2025-02-20_tableau_v1.md").write_text(
        "tableau électrique 3 rangées 36 modules disjoncteur 40A différentiel 30mA",
        encoding="utf-8",
    )
    idx = Index.build(
        c, tmp_path / "idx", grid=GRID, index_paths=False, smoothing_rank=0
    )

    groupes = versions_actuelles(idx, "spécification éclairage retenue", k=8)

    # la version d'éclairage canonique est la plus RÉCENTE (mai)
    eclairage = next(g for g in groupes if "eclairage" in g["canonique"])
    assert eclairage["canonique"] == "2025-05-12_eclairage_v3.md"
    assert eclairage["date"] == "2025-05-12"
    perimees = {p["id"] for p in eclairage["perimees"]}
    assert perimees == {"2025-01-15_eclairage_v1.md", "2025-03-10_eclairage_v2.md"}

    # le tableau n'est PAS fusionné avec l'éclairage (aspect distinct)
    tableau = next(g for g in groupes if "tableau" in g["canonique"])
    assert tableau["perimees"] == []  # une seule version
    assert "eclairage" not in tableau["canonique"]
