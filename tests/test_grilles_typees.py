"""Index à grilles typées (v4) : build/open/add, recherche typée (pondération + préséance
gatée), stats, CLI, compatibilité index historique. Cf. spec 2026-08-12-grilles-typees-v4.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mosaic.index import Index

PY = [sys.executable, "-m", "mosaic.cli"]


def _corpus(tmp_path: Path) -> Path:
    c = tmp_path / "corpus"
    c.mkdir(parents=True, exist_ok=True)
    (c / "elec.md").write_text(
        "pose interrupteur differentiel tableau protection reference a9f77216",
        encoding="utf-8",
    )
    (c / "carrelage.md").write_text(
        "achat carrelage gris cuisine colle joint pose sol", encoding="utf-8"
    )
    (c / "devis.md").write_text(
        "devis chantier peinture couloir escalier", encoding="utf-8"
    )
    return c


def _typee(tmp_path: Path) -> Index:
    return Index.build(_corpus(tmp_path), tmp_path / "idx", grilles_typees=True)


# -- build / stockage -----------------------------------------------------------------------


def test_build_cree_les_fichiers_par_grille(tmp_path):
    _typee(tmp_path)
    for nom in (
        "docs.msei",
        "vocab.msev",
        "docs_ref.msei",
        "vocab_ref.msev",
        "docs_chemin.msei",
        "vocab_chemin.msev",
    ):
        assert (tmp_path / "idx" / nom).is_file(), nom


def test_stats_expose_les_dims_effectives(tmp_path):
    s = _typee(tmp_path).stats()
    assert s["grilles_typees"] == {"sens": 3072, "ref": 768, "chemin": 768}


def test_index_historique_intact(tmp_path):
    """Sans le drapeau : aucun fichier typé, aucun changement de comportement."""
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx")
    assert idx.grilles is None
    assert not (tmp_path / "idx" / "docs_ref.msei").exists()
    assert "grilles_typees" not in idx.stats()


# -- recherche ------------------------------------------------------------------------------


def test_recherche_sens_et_ref(tmp_path):
    idx = _typee(tmp_path)
    assert idx.search("carrelage cuisine", k=1)[0]["id"] == "carrelage.md"
    hit = idx.search("a9f77216", k=1)[0]
    assert hit["id"] == "elec.md"
    assert hit["lectures"]["ref"] == 1.0  # signature pure : identifiant net


def test_preseance_identifiant_contre_la_noyade(tmp_path):
    """La réf + des mots d'un AUTRE document : le porteur de la réf passe devant
    (préséance mesurée au banc produits — la raison d'être des grilles typées)."""
    idx = _typee(tmp_path)
    hits = idx.search("a9f77216 carrelage gris cuisine", k=2)
    assert hits[0]["id"] == "elec.md"


def test_requete_sans_type_ref_ponderation_simple(tmp_path):
    idx = _typee(tmp_path)
    hits = idx.search("peinture couloir", k=1)
    assert hits[0]["id"] == "devis.md"
    assert (
        "ref" not in hits[0]["lectures"]
    )  # grille silencieuse : jamais dans la sortie


def test_rerank_refuse_sur_index_type(tmp_path):
    idx = _typee(tmp_path)
    with pytest.raises(ValueError, match="typées"):
        idx.search("pose", k=1, rerank=True)


def test_recherche_identique_apres_reouverture(tmp_path):
    idx = _typee(tmp_path)
    avant = idx.search("a9f77216 carrelage", k=3)
    relu = Index.open(tmp_path / "idx")
    assert relu.search("a9f77216 carrelage", k=3) == avant


def test_facettes_composent_avec_la_recherche_typee(tmp_path):
    idx = _typee(tmp_path)
    hits = idx.search("pose", k=3, type_filtre="note texte")
    assert hits  # le pipeline facettes reçoit les hits typés


# -- add ------------------------------------------------------------------------------------


def test_add_met_a_jour_toutes_les_grilles(tmp_path):
    idx = _typee(tmp_path)
    nouveau = tmp_path / "plomberie.md"
    nouveau.write_text(
        "robinet thermostatique salle de bain code zz88k77", encoding="utf-8"
    )
    idx.add(nouveau)
    assert idx.search("zz88k77", k=1)[0]["id"] == "plomberie.md"
    relu = Index.open(tmp_path / "idx")  # persistance (y compris memmap Windows coupé)
    assert relu.search("zz88k77", k=1)[0]["id"] == "plomberie.md"
    assert relu.stats()["docs"] == 4
    assert relu.grilles is not None
    assert all(g.mat.shape[0] == 4 for g in relu.grilles.values())


def test_add_sur_index_reouvert_grille_ref_sans_paires(tmp_path):
    """Le piège memmap résolu à la racine : une grille dont les tokens n'ont jamais
    co-occurru (une réf seule par doc) a un vocabulaire de profils VIDE — état normal,
    la signature porte tout. finalize doit quand même couper le memmap du chargement
    paresseux, sinon _save échoue sous Windows (mesuré)."""
    _typee(tmp_path)
    relu = Index.open(tmp_path / "idx")
    assert relu.grilles is not None and not relu.grilles["ref"].profiles.rows
    nouveau = tmp_path / "n.md"
    nouveau.write_text("document neuf reference qk99z31", encoding="utf-8")
    relu.add(nouveau)  # ne doit PAS lever PermissionError
    assert relu.search("qk99z31", k=1)[0]["id"] == "n.md"


# -- profil : type custom -------------------------------------------------------------------


def test_grille_custom_du_profil(tmp_path):
    profil = {
        "nom": "test",
        "grilles": {"norme": {"motif": r"nfc[0-9]+", "dim": 192}},
    }
    c = _corpus(tmp_path)
    (c / "norme.md").write_text("mise en conformite nfc15100 locaux", encoding="utf-8")
    idx = Index.build(c, tmp_path / "idx", grilles_typees=True, profil=profil)
    assert idx.stats()["grilles_typees"]["norme"] == 192
    hit = idx.search("nfc15100", k=1)[0]
    assert hit["id"] == "norme.md" and "norme" in hit["lectures"]
    relu = Index.open(tmp_path / "idx")
    assert relu.search("nfc15100", k=1)[0]["id"] == "norme.md"


# -- CLI ------------------------------------------------------------------------------------


def test_cli_build_grilles_typees_puis_search(tmp_path):
    corpus = _corpus(tmp_path)
    idx = str(tmp_path / "idx")
    r = subprocess.run(
        [*PY, "build", str(corpus), "-o", idx, "--grilles-typees"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    stats = json.loads(r.stdout)
    assert stats["grilles_typees"]["sens"] == 3072
    r2 = subprocess.run(
        [*PY, "search", "a9f77216", idx, "--top", "1"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    hits = json.loads(r2.stdout)
    assert hits[0]["id"] == "elec.md" and "lectures" in hits[0]
