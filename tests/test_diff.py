"""Diff sémantique (mosaic diff) : spécificité stricte, sensibilité, gardes, CLI.
Mécanisme validé par banc planté — research/diff_semantique.py (P1-P3)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mosaic.diff import diff_corpus, diff_indexes
from mosaic.index import Index

PY = [sys.executable, "-m", "mosaic.cli"]


def _corpus(base: Path, nom: str) -> Path:
    c = base / nom
    c.mkdir()
    (c / "elec.md").write_text(
        "pose interrupteur differentiel tableau protection beurre disjoncteur",
        encoding="utf-8",
    )
    (c / "cuisine.md").write_text(
        "sauce beurre jaune huile emulsion fouet", encoding="utf-8"
    )
    (c / "peinture.md").write_text(
        "peinture couloir escalier plafond rouleau", encoding="utf-8"
    )
    return c


def test_specificite_stricte_corpus_identique(tmp_path):
    """La garantie fondatrice : corpus identique des deux côtés ⇒ diff STRICTEMENT
    vide (zéro dérive, zéro apparition) — le déterminisme la porte au sens
    contractuel, pas à un ulp près."""
    c = _corpus(tmp_path, "c")
    d = diff_corpus(c, c)
    assert d["docs_ajoutes"] == [] and d["vocab_apparu"] == []
    assert d["derive_mots"] == []
    assert d["derive_contexte"] == []
    assert d["docs_modifies"] == []


def test_sensibilite_substitution_plantee(tmp_path):
    """beurre -> margarine dans un doc : margarine apparaît, beurre décline, et le
    document au contenu INTACT qui parle de beurre dérive en contexte."""
    t1 = _corpus(tmp_path, "t1")
    t2 = _corpus(tmp_path, "t2")
    (t2 / "cuisine.md").write_text(
        "sauce margarine jaune huile emulsion fouet", encoding="utf-8"
    )
    d = diff_corpus(t1, t2)
    assert "margarine" in d["vocab_apparu"]
    assert any(m["id"] == "cuisine.md" for m in d["docs_modifies"])
    # elec.md est INTACT mais parle de beurre : sa grille doit bouger (contexte)
    assert any(x["id"] == "elec.md" for x in d["derive_contexte"])


def test_diff_indexes_sans_hashes_lecture_fusionnee(tmp_path):
    c1 = _corpus(tmp_path, "c1")
    c2 = _corpus(tmp_path, "c2")
    (c2 / "quatre.md").write_text("chauffe eau ballon cuivre", encoding="utf-8")
    ia = Index.build(c1, tmp_path / "i1")
    ib = Index.build(c2, tmp_path / "i2")
    d = diff_indexes(ia, ib)
    assert d["docs_ajoutes"] == ["quatre.md"]
    assert "derive_documents" in d and "docs_modifies" not in d


def test_diff_refuse_espaces_differents(tmp_path):
    """Deux index d'espaces d'encodage différents (grid) : delta sans signification,
    refus net."""
    c = _corpus(tmp_path, "c")
    ia = Index.build(c, tmp_path / "i64")
    ib = Index.build(c, tmp_path / "i32", grid=(32, 32, 3))
    with pytest.raises(ValueError, match="espace d'encodage"):
        diff_indexes(ia, ib)


def test_cli_diff_corpus_et_gardes(tmp_path):
    t1 = _corpus(tmp_path, "t1")
    t2 = _corpus(tmp_path, "t2")
    (t2 / "nouveau.md").write_text("chantier terrasse dalle beton", encoding="utf-8")
    r = subprocess.run(
        [*PY, "diff", str(t1), str(t2)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["docs_ajoutes"] == ["nouveau.md"]
    # mixte corpus/index : refus
    idx = tmp_path / "idx"
    assert (
        subprocess.run(
            [*PY, "build", str(t1), "-o", str(idx)], capture_output=True
        ).returncode
        == 0
    )
    r = subprocess.run(
        [*PY, "diff", str(t1), str(idx)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 1
    assert "pas de sens" in json.loads(r.stderr.strip().splitlines()[-1])["error"]


def test_cli_actuel_relaye_les_options_de_recherche(tmp_path):
    """`actuel` n'est plus un search amputé : --rerank est relayé et ses exigences
    (rerank.msrv) s'appliquent — refus loud sur un index sans vecteurs."""
    c = _corpus(tmp_path, "c")
    idx = tmp_path / "idx"
    assert (
        subprocess.run(
            [*PY, "build", str(c), "-o", str(idx)], capture_output=True
        ).returncode
        == 0
    )
    r = subprocess.run(
        [*PY, "actuel", "beurre", str(idx), "--rerank"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 1
    assert "rerank.msrv" in json.loads(r.stderr.strip().splitlines()[-1])["error"]
