"""Tests — canal facettes (src/mosaic/facettes.py + intégration Index) : métadonnées exactes
type/date persistées au build, filtre par type et fusion récence à la recherche.

Deux régimes (piège CI no-extras) : les tests sur .md/.txt tournent PARTOUT (job cœur lean,
sans markitdown) — la date y est le discriminant ; les tests exigeant un convertible (.html,
type « page web ») se sautent proprement via importorskip (job complet uniquement)."""

import json

import pytest

from mosaic.facettes import appliquer, charger
from mosaic.index import Index

GRID = (32, 32, 3)


def _corpus_date(tmp_path):
    """Corpus lean (.md/.txt seulement — ingérable sans markitdown), daté dans les noms."""
    c = tmp_path / "corpus"
    c.mkdir()
    sujet = "disjoncteur tableau electrique protection differentielle"
    (c / "2025-01-10_notice.md").write_text(
        f"{sujet} notice installation", encoding="utf-8"
    )
    (c / "2025-03-15_compte_rendu.md").write_text(
        f"{sujet} pose chantier", encoding="utf-8"
    )
    (c / "2025-06-20_specifications.txt").write_text(
        f"{sujet} specifications techniques", encoding="utf-8"
    )
    return c


def test_facettes_persistees_au_build(tmp_path):
    c = _corpus_date(tmp_path)
    Index.build(c, tmp_path / "idx", grid=GRID, index_paths=False, smoothing_rank=0)
    fac = charger(tmp_path / "idx")
    assert fac is not None
    assert fac["2025-01-10_notice.md"]["type"] == "note texte"
    assert fac["2025-01-10_notice.md"]["date"] == "2025-01-10"
    assert fac["2025-06-20_specifications.txt"]["date"] == "2025-06-20"
    # le fichier est du JSON valide sur disque
    brut = json.loads((tmp_path / "idx" / "facettes.json").read_text(encoding="utf-8"))
    assert len(brut) == 3


def test_search_fusion_recence(tmp_path):
    c = _corpus_date(tmp_path)
    idx = Index.build(
        c, tmp_path / "idx", grid=GRID, index_paths=False, smoothing_rank=0
    )
    # récence à fond : le doc le plus récent (juin) doit dominer le classement
    r = idx.search("disjoncteur tableau", k=3, recence=1.0)
    assert r[0]["id"] == "2025-06-20_specifications.txt"
    assert "score_fusion" in r[0]
    # sans récence : ordre sémantique pur, aucune clé de fusion
    r0 = idx.search("disjoncteur tableau", k=3)
    assert "score_fusion" not in r0[0]


def test_search_recence_apres_reouverture(tmp_path):
    """Les facettes survivent au cycle save/open (persistance réelle, pas l'objet en RAM)."""
    c = _corpus_date(tmp_path)
    Index.build(c, tmp_path / "idx", grid=GRID, index_paths=False, smoothing_rank=0)
    idx = Index.open(tmp_path / "idx")
    r = idx.search("disjoncteur tableau", k=3, recence=1.0)
    assert r[0]["id"] == "2025-06-20_specifications.txt"


def test_search_filtre_par_type_convertible(tmp_path):
    """Filtre exact par type sur corpus mixte — exige markitdown (le .html doit être ingéré
    pour porter le type « page web »)."""
    pytest.importorskip("markitdown")
    c = _corpus_date(tmp_path)
    (c / "2025-02-01_fiche.html").write_text(
        "<html><body>disjoncteur tableau fiche web</body></html>", encoding="utf-8"
    )
    idx = Index.build(
        c, tmp_path / "idx", grid=GRID, index_paths=False, smoothing_rank=0
    )
    fac = charger(tmp_path / "idx")
    assert fac is not None and fac["2025-02-01_fiche.html"]["type"] == "page web"
    r = idx.search("disjoncteur tableau", k=4, type_filtre="page web")
    assert len(r) == 1  # seul le .html est une page web
    assert r[0]["id"] == "2025-02-01_fiche.html"
    assert r[0]["type"] == "page web"  # facette exposée pour l'explicabilité


def test_index_sans_facettes_refuse_loud(tmp_path):
    """Un index antérieur (sans facettes.json) refuse type/recence avec une erreur claire."""
    c = _corpus_date(tmp_path)
    Index.build(c, tmp_path / "idx", grid=GRID, index_paths=False, smoothing_rank=0)
    (tmp_path / "idx" / "facettes.json").unlink()  # simule un index d'avant
    idx2 = Index.open(tmp_path / "idx")
    with pytest.raises(ValueError, match="facettes"):
        idx2.search("disjoncteur", k=3, type_filtre="tableur")
    # sans les options, la recherche marche comme avant (zéro régression)
    assert idx2.search("disjoncteur", k=3)


def test_appliquer_validation():
    with pytest.raises(ValueError, match="recence"):
        appliquer([], {}, recence=1.5)


def test_refs_du_texte_criteres():
    """Réf probable = >=5 mixtes ou >=6 chiffres ; jamais un calibre court ni une année."""
    from mosaic.facettes import refs_du_texte

    refs = refs_du_texte(
        "disjoncteur MFN710 calibre 16A pose 2025 ref 086027 D26050018"
    )
    assert "MFN710" in refs
    assert "086027" in refs
    assert "D26050018" in refs
    assert "16A" not in refs  # calibre court : pas une réf
    assert "2025" not in refs  # année : pas une réf


def test_boost_ref_exacte(tmp_path):
    """Une requête portant une réf exacte propulse le document porteur en tête, même si le
    flou sémantique le classait derrière. Corpus lean .md (tourne partout)."""
    c = tmp_path / "corpus"
    c.mkdir()
    # doc CIBLE : porte la réf MFN710 mais un vocabulaire pauvre
    (c / "2025-01-05_bon_commande.md").write_text(
        "commande fourniture MFN710 quantite deux", encoding="utf-8"
    )
    # docs DISTRACTEURS : riches sémantiquement sur « disjoncteur », sans la réf
    for i in range(3):
        (c / f"2025-02-0{i + 1}_note{i}.md").write_text(
            "disjoncteur tableau electrique protection disjoncteur differentiel "
            f"disjoncteur pose chantier variante {i}",
            encoding="utf-8",
        )
    idx = Index.build(
        c, tmp_path / "idx", grid=GRID, index_paths=False, smoothing_rank=0
    )

    r = idx.search("disjoncteur MFN710", k=4)
    assert r[0]["id"] == "2025-01-05_bon_commande.md"  # le porteur de la réf en tête
    assert r[0]["ref_exacte"] == ["MFN710"]  # explicabilité du boost
    # sans réf dans la requête : pas de champ ref_exacte, ordre sémantique
    r2 = idx.search("disjoncteur tableau", k=4)
    assert all("ref_exacte" not in h for h in r2)


def test_add_maintient_facettes(tmp_path):
    c = _corpus_date(tmp_path)
    idx = Index.build(
        c, tmp_path / "idx", grid=GRID, index_paths=False, smoothing_rank=0
    )
    nouveau = tmp_path / "2025-07-01_ajout.txt"
    nouveau.write_text("disjoncteur nouveau document ajout", encoding="utf-8")
    idx.add(nouveau)
    fac = charger(tmp_path / "idx")
    assert fac is not None and fac["2025-07-01_ajout.txt"]["date"] == "2025-07-01"
    assert fac["2025-07-01_ajout.txt"]["type"] == "note texte"
