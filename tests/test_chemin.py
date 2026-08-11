"""Tests — parcours multi-sauts v3 (queries.chemin) : doc -> entités (déliage vectoriel du
canal de relations) -> documents frères. Corpus .md arborescent (tourne partout, sans extras)."""

import pytest

from mosaic.index import Index
from mosaic.relations import entites_du_canal

GRID = (32, 32, 3)


def _corpus_arborescent(tmp_path):
    """Deux chantiers, deux années : ATLAS/2025, ATLAS/2026, HELIOS/2026 — de quoi traverser
    « même dossier » et « même année »."""
    c = tmp_path / "corpus"
    for chemin, texte in [
        ("ATLAS/2025/note_cablage.md", "cablage tableau electrique atlas"),
        ("ATLAS/2026/note_eclairage.md", "eclairage led atlas couloirs"),
        ("ATLAS/2026/note_reception.md", "reception travaux atlas proces verbal"),
        ("HELIOS/2026/note_chauffage.md", "chauffage radiateurs helios bureaux"),
    ]:
        f = c / chemin
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(texte, encoding="utf-8")
    return c


def test_deliage_vectoriel_retrouve_les_entites(tmp_path):
    """Saut 1 : les entités d'un document sont retrouvées par DÉLIAGE de son canal (pur
    vectoriel), sans lire son chemin — et collent à la vérité du chemin."""
    c = _corpus_arborescent(tmp_path)
    idx = Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        index_paths=False,
        smoothing_rank=0,
        relations=True,
    )
    row = idx.ids.index("ATLAS/2026/note_eclairage.md")
    assert idx.relations_mat is not None
    entites = entites_du_canal(
        idx.relations_mat[row], idx.relations_manifest, idx.relations_mat.shape[1]
    )
    trouve = {(r, e) for r, e, _cos in entites}
    assert ("dossier", "atlas") in trouve
    assert ("annee", "2026") in trouve
    assert ("dossier", "helios") not in trouve  # pas d'entité d'un autre document


def test_chemin_meme_dossier_et_meme_annee(tmp_path):
    """Saut 1 + saut 2 : depuis une note ATLAS/2026, la traversée rend les frères du même
    dossier (les autres notes ATLAS) et de la même année (dont HELIOS/2026), départ exclu."""
    c = _corpus_arborescent(tmp_path)
    idx = Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        index_paths=False,
        smoothing_rank=0,
        relations=True,
    )
    groupes = idx.chemin("ATLAS/2026/note_eclairage.md", k=5)
    par_cle = {(g["role"], g["entite"]): g for g in groupes}

    dossier = par_cle[("dossier", "atlas")]
    ids_dossier = {d["id"] for d in dossier["documents"]}
    assert "ATLAS/2026/note_reception.md" in ids_dossier
    assert "ATLAS/2025/note_cablage.md" in ids_dossier
    assert "ATLAS/2026/note_eclairage.md" not in ids_dossier  # départ exclu

    annee = par_cle[("annee", "2026")]
    ids_annee = {d["id"] for d in annee["documents"]}
    assert "HELIOS/2026/note_chauffage.md" in ids_annee  # autre chantier, même année
    assert "ATLAS/2025/note_cablage.md" not in ids_annee  # autre année, hors groupe

    # restriction par rôle
    seul_annee = idx.chemin("ATLAS/2026/note_eclairage.md", k=5, role="annee")
    assert all(g["role"] == "annee" for g in seul_annee)


def test_chemin_sans_relations_refuse_loud(tmp_path):
    c = _corpus_arborescent(tmp_path)
    idx = Index.build(
        c, tmp_path / "idx", grid=GRID, index_paths=False, smoothing_rank=0
    )  # relations=False
    with pytest.raises(ValueError, match="relations"):
        idx.chemin("ATLAS/2026/note_eclairage.md")


def test_chemin_document_inconnu(tmp_path):
    c = _corpus_arborescent(tmp_path)
    idx = Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        index_paths=False,
        smoothing_rank=0,
        relations=True,
    )
    with pytest.raises(ValueError, match="inconnu"):
        idx.chemin("nexiste/pas.md")
