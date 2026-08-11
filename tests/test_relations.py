"""Canal de relations (v2.0) — liaison, extraction de chemin, stockage, Index.related()."""

import sys
from pathlib import Path

import numpy as np
import pytest

from mosaic.index import Index
from mosaic.relations import (
    bind,
    decalage,
    document_channel,
    entities_from_path,
    normalize_entity,
)
from mosaic.signatures import signature
from mosaic.store import load_relations, save_relations

GRID = (32, 32, 3)
DIM = 3072


# -- bind() / decalage() -------------------------------------------------------------------


def test_decalage_deterministe_et_dans_les_bornes():
    d1 = decalage("dossier", DIM)
    d2 = decalage("dossier", DIM)
    assert d1 == d2
    assert 1 <= d1 < DIM


def test_decalage_varie_par_role():
    assert decalage("dossier", DIM) != decalage("annee", DIM)


def test_bind_deterministe():
    a = bind("dossier", "atlas_nord", DIM)
    b = bind("dossier", "atlas_nord", DIM)
    assert np.array_equal(a, b)


def test_bind_est_une_permutation_circulaire_de_la_signature():
    sig = signature("atlas_nord", DIM)
    off = decalage("dossier", DIM)
    assert np.array_equal(bind("dossier", "atlas_nord", DIM), np.roll(sig, off))


def test_bind_role_different_donne_vecteur_different():
    a = bind("dossier", "atlas_nord", DIM)
    b = bind("annee", "atlas_nord", DIM)
    assert not np.array_equal(a, b)


def test_bind_entite_differente_quasi_orthogonale():
    a = bind("dossier", "atlas_nord", DIM)
    b = bind("dossier", "affaire_a", DIM)
    # signatures creuses (40 non-nuls/12288 dims) : produit scalaire attendu ~0
    assert abs(int(a.astype(np.int64) @ b.astype(np.int64))) <= 12


# -- document_channel() ---------------------------------------------------------------------


def test_document_channel_vide_sans_relation():
    q, n = document_channel([], DIM)
    assert q.dtype == np.int8
    assert np.all(q == 0)
    assert n == 0.0


def test_document_channel_non_vide():
    q, n = document_channel([("dossier", "atlas_nord"), ("annee", "2026")], DIM)
    assert q.dtype == np.int8
    assert n > 0.0
    assert int(np.abs(q).max()) == 127  # quantification pic-127


def test_document_channel_deterministe():
    rels = [("dossier", "atlas_nord"), ("mois", "2026-08")]
    q1, n1 = document_channel(rels, DIM)
    q2, n2 = document_channel(rels, DIM)
    assert np.array_equal(q1, q2) and n1 == n2


# -- normalize_entity() ----------------------------------------------------------------------


def test_normalize_entity_minuscule_et_join_underscore():
    assert normalize_entity("ATLAS_NORD") == "atlas_nord"


def test_normalize_entity_accents_conserves():
    assert normalize_entity("Août") == "août"


def test_normalize_entity_idempotente():
    v = normalize_entity("Devis")
    assert normalize_entity(v) == v


# -- entities_from_path() --------------------------------------------------------------------


def test_entities_from_path_exemple_spec():
    doc_id = "2026/08-Août/ATLAS_NORD/03_Devis/devis.pdf"
    rels = entities_from_path(doc_id)
    rel_set = set(rels)
    assert ("annee", "2026") in rel_set
    assert (
        "dossier",
        "2026",
    ) not in rel_set  # pur marqueur de date : pas de dossier (revue v2.0)
    assert ("dossier", "août") in rel_set
    assert ("mois", "2026-08") in rel_set
    assert ("dossier", "atlas_nord") in rel_set
    assert ("dossier", "devis") in rel_set
    # le nom de fichier n'est jamais une relation
    assert not any(e == "devis.pdf" or e == "devis_pdf" for _, e in rels)


def test_entities_from_path_mm_point_aaaa():
    doc_id = "08.2026/ATLAS/devis.pdf"
    rels = set(entities_from_path(doc_id))
    assert ("mois", "2026-08") in rels
    assert ("dossier", "atlas") in rels


def test_entities_from_path_mois_sans_annee_prealable_pas_de_relation_mois():
    doc_id = "08-Août/ATLAS/devis.pdf"
    rels = set(entities_from_path(doc_id))
    assert not any(role == "mois" for role, _ in rels)
    assert ("dossier", "août") in rels


def test_entities_from_path_fichier_a_la_racine_vide():
    assert entities_from_path("devis.pdf") == []


def test_entities_from_path_deduplique_par_document():
    doc_id = "ATLAS/ATLAS/devis.pdf"
    rels = entities_from_path(doc_id)
    assert rels.count(("dossier", "atlas")) == 1


def test_entities_from_path_annee_seule_relation_temporelle_seule():
    # un segment "2026" pur est un marqueur temporel : (annee, 2026) SEUL, pas de dossier
    # (revue v2.0 — évite la collision avec la convention devis 08.2026)
    rels = set(entities_from_path("2026/devis.pdf"))
    assert ("annee", "2026") in rels
    assert ("dossier", "2026") not in rels


# -- store.py : relations.msrel round-trip ----------------------------------------------------


def test_relations_absent_retourne_none(tmp_path):
    assert load_relations(tmp_path) is None


def test_relations_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    mat = rng.integers(-127, 128, size=(3, DIM), dtype=np.int8)
    norms = np.linalg.norm(mat.astype(np.float32), axis=1)
    manifest = {"atlas_nord": {"dossier"}, "2026": {"annee", "dossier"}}
    save_relations(tmp_path, mat, norms, manifest, GRID)
    mat2, norms2, manifest2, grid2 = load_relations(tmp_path)
    assert np.array_equal(mat, mat2)
    assert np.allclose(norms, norms2)
    assert grid2 == GRID
    assert manifest2 == {"atlas_nord": ["dossier"], "2026": ["annee", "dossier"]}


def test_relations_deterministe_entre_deux_ecritures(tmp_path):
    mat = np.ones((2, DIM), dtype=np.int8)
    norms = np.linalg.norm(mat.astype(np.float32), axis=1)
    manifest = {"atlas": {"dossier"}}
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    save_relations(dir_a, mat, norms, manifest, GRID)
    save_relations(dir_b, mat, norms, manifest, GRID)
    assert (dir_a / "relations.msrel").read_bytes() == (
        dir_b / "relations.msrel"
    ).read_bytes()


def test_relations_magic_invalide(tmp_path):
    (tmp_path / "relations.msrel").write_bytes(b"XXXX" + b"\x00" * 32)
    with pytest.raises(ValueError):
        load_relations(tmp_path)


def test_relations_tronque_leve_valueerror(tmp_path):
    mat = np.ones((2, DIM), dtype=np.int8)
    norms = np.linalg.norm(mat.astype(np.float32), axis=1)
    save_relations(tmp_path, mat, norms, {"atlas": {"dossier"}}, GRID)
    chemin = tmp_path / "relations.msrel"
    data = bytearray(chemin.read_bytes())
    del data[-4:]
    chemin.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        load_relations(tmp_path)


def test_relations_vmin_non_supporte_leve_valueerror(tmp_path):
    import json
    import struct

    meta = {"manifest": {}}
    blob = json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
    header = struct.Struct("<4sBBHHBBI").pack(
        b"MSRL", 1, 99, GRID[0], GRID[1], GRID[2], 0, 1
    )
    with open(tmp_path / "relations.msrel", "wb") as f:
        f.write(header)
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(np.zeros(1, dtype=np.float32).tobytes())
        f.write(np.zeros(DIM, dtype=np.int8).tobytes())
    with pytest.raises(ValueError):
        load_relations(tmp_path)


# -- Index.build(relations=True) / Index.related() ---------------------------------------------


def _chantiers_corpus(tmp_path: Path) -> Path:
    c = tmp_path / "corpus"
    atlas = c / "2026" / "08-Août" / "ATLAS_NORD"
    (atlas / "03_Devis").mkdir(parents=True)
    (atlas / "03_Devis" / "devis.md").write_text(
        "pose interrupteur differentiel tableau protection circuit", encoding="utf-8"
    )
    (atlas / "05_Facture").mkdir(parents=True)
    (atlas / "05_Facture" / "facture.md").write_text(
        "reglement facture acompte chantier atlas", encoding="utf-8"
    )
    affaire_a = c / "2026" / "07-Juillet" / "AFFAIRE_A" / "03_Devis"
    affaire_a.mkdir(parents=True)
    (affaire_a / "devis.md").write_text(
        "cablage armoire electrique local technique affaire_a", encoding="utf-8"
    )
    return c


def test_build_sans_relations_aucun_fichier_msrel(tmp_path):
    idx = Index.build(_chantiers_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    assert not (tmp_path / "idx" / "relations.msrel").is_file()
    assert idx.relations_mat is None


def test_build_avec_relations_cree_le_fichier(tmp_path):
    idx = Index.build(
        _chantiers_corpus(tmp_path), tmp_path / "idx", grid=GRID, relations=True
    )
    assert (tmp_path / "idx" / "relations.msrel").is_file()
    assert idx.relations_mat is not None
    assert idx.relations_mat.shape == (3, DIM)
    assert "atlas_nord" in idx.relations_manifest


def test_related_remonte_les_bons_documents(tmp_path):
    idx = Index.build(
        _chantiers_corpus(tmp_path), tmp_path / "idx", grid=GRID, relations=True
    )
    hits = idx.related("atlas_nord", k=5)
    ids = [h["id"] for h in hits[:2]]
    assert set(ids) == {
        "2026/08-Août/ATLAS_NORD/03_Devis/devis.md",
        "2026/08-Août/ATLAS_NORD/05_Facture/facture.md",
    }
    affaire_a_id = "2026/07-Juillet/AFFAIRE_A/03_Devis/devis.md"
    scores = {h["id"]: h["score"] for h in hits}
    assert scores[ids[0]] > scores.get(affaire_a_id, -1.0)


def test_related_avec_role_explicite(tmp_path):
    idx = Index.build(
        _chantiers_corpus(tmp_path), tmp_path / "idx", grid=GRID, relations=True
    )
    hits = idx.related("2026", k=10, role="annee")
    assert len(hits) == 3  # les 3 docs sont tous en 2026


def test_related_entite_inconnue_liste_vide(tmp_path):
    idx = Index.build(
        _chantiers_corpus(tmp_path), tmp_path / "idx", grid=GRID, relations=True
    )
    assert idx.related("entreprise_qui_n_existe_pas") == []


def test_related_sans_relations_leve_valueerror_clair(tmp_path):
    idx = Index.build(_chantiers_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    with pytest.raises(ValueError, match="relations"):
        idx.related("atlas_nord")


def test_related_apres_reouverture(tmp_path):
    Index.build(
        _chantiers_corpus(tmp_path), tmp_path / "idx", grid=GRID, relations=True
    )
    idx = Index.open(tmp_path / "idx")
    hits = idx.related("atlas_nord", k=2)
    assert len(hits) == 2


def test_stats_expose_relations_quand_actif(tmp_path):
    idx_on = Index.build(
        _chantiers_corpus(tmp_path / "on"),
        tmp_path / "idx_on",
        grid=GRID,
        relations=True,
    )
    assert idx_on.stats().get("relations") is True
    idx_off = Index.build(
        _chantiers_corpus(tmp_path / "off"), tmp_path / "idx_off", grid=GRID
    )
    assert "relations" not in idx_off.stats()


def test_add_met_a_jour_relations_msrel(tmp_path):
    idx = Index.build(
        _chantiers_corpus(tmp_path), tmp_path / "idx", grid=GRID, relations=True
    )
    extra = tmp_path / "plomberie.md"
    extra.write_text(
        "chauffe-eau ballon eau chaude raccordement cuivre", encoding="utf-8"
    )
    idx.add(extra)
    assert idx.relations_mat.shape[0] == 4
    idx2 = Index.open(tmp_path / "idx")
    assert idx2.relations_mat.shape[0] == 4


def test_build_deterministe_avec_relations(tmp_path):
    Index.build(
        _chantiers_corpus(tmp_path / "corpus_a"),
        tmp_path / "ia",
        grid=GRID,
        relations=True,
    )
    Index.build(
        _chantiers_corpus(tmp_path / "corpus_b"),
        tmp_path / "ib",
        grid=GRID,
        relations=True,
    )
    assert (tmp_path / "ia" / "relations.msrel").read_bytes() == (
        tmp_path / "ib" / "relations.msrel"
    ).read_bytes()


# -- garantie « zéro changement » sur les fichiers existants -----------------------------------


def _fixed_corpus(tmp_path: Path) -> Path:
    c = tmp_path / "corpus"
    d = c / "2026" / "08-Août" / "ATLAS_NORD" / "03_Devis"
    d.mkdir(parents=True)
    (d / "devis.md").write_text(
        "pose interrupteur differentiel tableau protection circuit", encoding="utf-8"
    )
    (c / "2026" / "08-Août" / "ATLAS_NORD" / "autre.md").write_text(
        "chantier atlas nord renovation electrique", encoding="utf-8"
    )
    return c


# Empreintes capturées AVANT l'introduction du canal relations (v1.6, même corpus/grille) —
# garantit que --relations OFF (défaut) ne change STRICTEMENT rien aux fichiers existants.
_SHA_DOCS_MSEI_V16 = "76ed763d578be5e1ca81265cdca63604c541b61fa7430399f1677cfc6cb781cb"
_SHA_VOCAB_MSEV_V16 = "5fb3c9018b9b0847e16ccf9f74d93376560531ae32ac423340dc35af550efb62"


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="SHA de référence capturé sur la machine de production (Windows) ; "
    "l'identité octet-à-octet du docs.msei dépend du backend BLAS de numpy "
    "(SVD flottante) et ne se transporte pas telle quelle vers un runner Linux",
)
def test_relations_off_bytes_identiques_a_v16(tmp_path):
    import hashlib

    idx_dir = tmp_path / "idx"
    Index.build(_fixed_corpus(tmp_path), idx_dir, grid=GRID)
    docs_sha = hashlib.sha256((idx_dir / "docs.msei").read_bytes()).hexdigest()
    vocab_sha = hashlib.sha256((idx_dir / "vocab.msev").read_bytes()).hexdigest()
    assert docs_sha == _SHA_DOCS_MSEI_V16
    assert vocab_sha == _SHA_VOCAB_MSEV_V16


def test_entities_from_path_date_pure_pas_de_dossier_collision():
    """Revue v2.0 : segments purs de date (2026, 08.2026) → relations temporelles
    seules, jamais (dossier, "2026") — sinon les conventions devis (08.2026) et
    chantiers (08-Août) collisionneraient."""
    from mosaic.relations import entities_from_path

    devis = entities_from_path("2026/08.2026/D26089903_KUMQUAT.pdf")
    assert ("annee", "2026") in devis and ("mois", "2026-08") in devis
    assert not any(role == "dossier" for role, _ in devis)  # aucun faux dossier
    chantier = entities_from_path("2026/08-Août/ATLAS_NORD/devis.pdf")
    assert ("dossier", "août") in chantier  # mois libellé = dossier cherchable
    assert ("dossier", "atlas_nord") in chantier
    assert ("mois", "2026-08") in chantier
