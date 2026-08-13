"""Tests — vérité temporelle (src/mosaic/temporel.py) : la version la plus récente d'un aspect
est canonique, les précédentes sont marquées périmées ; les aspects distincts ne fusionnent pas."""

from mosaic.index import Index
from mosaic.temporel import date_du_chemin, versions_actuelles

GRID = (32, 32, 3)


def test_date_du_chemin():
    assert date_du_chemin("2025-05-12_eclairage_v3.md") == "2025-05-12"
    assert date_du_chemin("sous/dossier/2024-01-02_note.md") == "2024-01-02"
    assert date_du_chemin("sans_date.md") == "0000-00-00"


def test_date_du_chemin_format_compact():
    # Convention massive des corpus compta/chantiers : AAAAMMJJ collé en préfixe ou
    # suffixe du nom (973/1271 docs compta étaient sans date à cause de ce format).
    assert date_du_chemin("20260622_Fournisseur_BL_9990001.pdf") == "2026-06-22"
    assert (
        date_du_chemin(
            "Bons de Livraison/FOURNISSEUR/2026/06-Juin/20260622_Fournisseur_BL_9990001.pdf"
        )
        == "2026-06-22"
    )
    assert date_du_chemin("bilan_puissance_tarif-jaune_20260630.md") == "2026-06-30"
    # le format explicite à tirets garde la priorité sur un compact présent ailleurs
    assert date_du_chemin("2026-05-12_client_BC_20250101.PDF") == "2026-05-12"


def test_date_du_chemin_dossier_classement():
    # Classement par dossier AAAA/MM-Mois (convention des corpus compta/chantiers) :
    # granularité mois, jour « 00 » — trie en tête du mois, derrière rien du mois,
    # devant la sentinelle 0000-00-00. Ne s'applique qu'en l'absence de date complète.
    assert (
        date_du_chemin("Factures clients/2024/06-Juin/F24069901 client.pdf")
        == "2024-06-00"
    )
    assert (
        date_du_chemin("2026_05-Mai_CCI_AFFAIRE_A_MEMOIRE_CHANTIER.md") == "2026-05-00"
    )
    # une date complète (compacte ou à tirets) garde la priorité sur le dossier
    assert (
        date_du_chemin(
            "Bons de Livraison/FOURNISSEUR/2026/06-Juin/20260622_Fournisseur_BL_9990001.pdf"
        )
        == "2026-06-22"
    )
    # le tiret doit ouvrir un libellé (lettre), jamais un nombre
    assert date_du_chemin("archives/2024/12-31_note.md") == "0000-00-00"


def test_date_du_chemin_compact_sans_faux_positifs():
    # une référence numérique n'est jamais une date : fenêtre de 8 chiffres NON isolée,
    # année hors 19xx/20xx, mois ou jour invalides
    assert date_du_chemin("BC_009990001.PDF") == "0000-00-00"  # 9 chiffres collés
    assert date_du_chemin("AR-4050499010.pdf") == "0000-00-00"  # 10 chiffres collés
    assert (
        date_du_chemin("F26070002_facture.pdf") == "0000-00-00"
    )  # année 2607, mois 00
    assert date_du_chemin("ref_20261325_x.pdf") == "0000-00-00"  # mois 13
    assert date_du_chemin("ref_20260100_x.pdf") == "0000-00-00"  # jour 00
    assert date_du_chemin("ref_20260232_x.pdf") == "0000-00-00"  # jour 32


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
