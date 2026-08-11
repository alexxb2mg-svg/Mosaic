"""Fusion hybride à trois canaux (grille + BM25 + embeddings) — module bm25, persistance
bm25.msbm, build --hybride, search(fusion=True), add() incrémental, CLI.

Comme test_index_rerank : un faux modèle model2vec (déterministe, seedé par hash du texte)
pour les tests Index — seuls les tests CLI (subprocess) exigent le vrai model2vec
(importorskip). Le banc Alloprof reste la mesure de QUALITÉ de la fusion ; ici on teste
la MÉCANIQUE (exactitude BM25, invariants add ≡ rebuild, déterminisme, refus nets).
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mosaic import rerank
from mosaic.bm25 import B, K1, Bm25
from mosaic.index import Index
from mosaic.store import load_bm25, save_bm25

GRID = (32, 32, 3)
PY = [sys.executable, "-m", "mosaic.cli"]


def _fake_model(monkeypatch) -> None:
    class _FakeModel:
        def encode(self, texts):
            out = np.zeros((len(texts), rerank.DIM), dtype=np.float32)
            for i, t in enumerate(texts):
                out[i] = np.random.default_rng(hash(t) & 0xFFFFFFFF).normal(
                    size=rerank.DIM
                )
            return out

    monkeypatch.setattr(rerank, "StaticModel", _FakeModel)
    monkeypatch.setattr(rerank, "_get_model", lambda: _FakeModel())


def _corpus(tmp_path: Path) -> Path:
    c = tmp_path / "corpus"
    c.mkdir(parents=True, exist_ok=True)
    (c / "elec.md").write_text(
        "pose interrupteur differentiel tableau protection circuit disjoncteur",
        encoding="utf-8",
    )
    (c / "carrelage.md").write_text(
        "achat carrelage gris cuisine colle joint pose sol", encoding="utf-8"
    )
    (c / "devis_xkr914.md").write_text(
        "devis reference xkr914 chantier peinture couloir", encoding="utf-8"
    )
    return c


# -- module bm25 ----------------------------------------------------------------------------


def test_bm25_scores_calcul_manuel():
    docs = [("a", ["chat", "chien", "chat"]), ("b", ["chien", "oiseau"])]
    bm = Bm25.from_docs(docs)
    scores = bm.scores(["chat"])
    # BM25 Okapi à la main : N=2, df(chat)=1, idf=ln(1 + (2-1+0.5)/(1+0.5)) = ln(2)
    # doc a : tf=2, len=3, avgdl=2.5 -> tf*(K1+1) / (tf + K1*(1-B+B*3/2.5))
    idf = math.log(2.0)
    denom = K1 * (1 - B + B * 3 / 2.5)
    attendu = idf * 2 * (K1 + 1) / (2 + denom)
    assert scores[0] == pytest.approx(attendu, rel=1e-5)
    assert scores[1] == 0.0


def test_bm25_terme_inconnu_ignore():
    bm = Bm25.from_docs([("a", ["chat"]), ("b", ["chien"])])
    assert not np.any(bm.scores(["zebre"]))
    assert not np.any(bm.scores([]))


def test_bm25_occurrences_multiples_dans_la_requete_comptent():
    bm = Bm25.from_docs([("a", ["chat", "chien"]), ("b", ["chien", "chien"])])
    une = bm.scores(["chat"])
    deux = bm.scores(["chat", "chat"])
    assert deux[0] == pytest.approx(2 * une[0], rel=1e-6)


def test_bm25_roundtrip_persistance(tmp_path):
    docs = [("a", ["chat", "chien", "chat"]), ("b", ["chien", "oiseau"])]
    bm = Bm25.from_docs(docs)
    save_bm25(tmp_path, bm)
    relu = load_bm25(tmp_path)
    assert relu is not None
    assert relu.vocab_termes == bm.vocab_termes
    np.testing.assert_array_equal(relu.indptr, bm.indptr)
    np.testing.assert_array_equal(relu.doc_idx, bm.doc_idx)
    np.testing.assert_array_equal(relu.tf, bm.tf)
    np.testing.assert_array_equal(relu.doc_lens, bm.doc_lens)


def test_bm25_load_absent_rend_none(tmp_path):
    assert load_bm25(tmp_path) is None


def test_bm25_add_doc_equivaut_au_rebuild():
    """Invariant fort : add_doc() produit EXACTEMENT les mêmes structures qu'un from_docs
    sur le corpus complet — jamais deux mondes selon le chemin d'arrivée du document."""
    d1 = ("a", ["chat", "chien", "chat"])
    d2 = ("b", ["chien", "oiseau"])
    d3 = ("c", ["oiseau", "poisson", "chat", "poisson"])
    increments = Bm25.from_docs([d1, d2])
    increments.add_doc(d3[1])
    rebuild = Bm25.from_docs([d1, d2, d3])
    assert increments.vocab_termes == rebuild.vocab_termes
    np.testing.assert_array_equal(increments.indptr, rebuild.indptr)
    np.testing.assert_array_equal(increments.doc_idx, rebuild.doc_idx)
    np.testing.assert_array_equal(increments.tf, rebuild.tf)
    np.testing.assert_array_equal(increments.doc_lens, rebuild.doc_lens)


# -- build / open / add ---------------------------------------------------------------------


def test_build_hybride_cree_les_deux_fichiers(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, hybride=True)
    assert (tmp_path / "idx" / "bm25.msbm").is_file()
    assert (tmp_path / "idx" / "rerank.msrv").is_file()  # --hybride implique rerank
    assert idx.stats().get("hybride") is True


def test_open_recharge_le_canal_bm25(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, hybride=True)
    relu = Index.open(tmp_path / "idx")
    assert relu.bm25 is not None and relu.bm25.n_docs == 3


def test_add_maintient_bm25(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, hybride=True)
    nouveau = tmp_path / "plomberie.md"
    nouveau.write_text(
        "remplacement robinet thermostatique zzqv77 salle de bain", encoding="utf-8"
    )
    idx.add(nouveau)
    assert idx.bm25 is not None and idx.bm25.n_docs == 4
    relu = Index.open(tmp_path / "idx")  # persistance de l'add
    assert relu.bm25 is not None and relu.bm25.n_docs == 4
    hits = relu.search("zzqv77", k=1, fusion=True)
    assert hits and hits[0]["id"] == "plomberie.md"


# -- search(fusion=True) --------------------------------------------------------------------


def test_fusion_retrouve_le_terme_exact_rare(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, hybride=True)
    hits = idx.search("xkr914", k=3, fusion=True)
    assert hits and hits[0]["id"] == "devis_xkr914.md"
    assert "bm25" in hits[0]["rangs"] and hits[0]["rangs"]["bm25"] == 1


def test_fusion_deterministe_et_identique_apres_reouverture(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, hybride=True)
    a = idx.search("pose interrupteur", k=3, fusion=True)
    b = idx.search("pose interrupteur", k=3, fusion=True)
    assert a == b
    relu = Index.open(tmp_path / "idx")
    assert relu.search("pose interrupteur", k=3, fusion=True) == a


def test_fusion_expose_les_rangs_par_canal(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, hybride=True)
    hits = idx.search("pose carrelage cuisine", k=1, fusion=True)
    assert set(hits[0]["rangs"]) <= {"grille", "bm25", "embed"}
    assert all(r >= 1 for r in hits[0]["rangs"].values())


def test_fusion_sans_index_hybride_refuse(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    with pytest.raises(ValueError, match="hybride"):
        idx.search("pose", k=1, fusion=True)


def test_fusion_et_rerank_exclusifs(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, hybride=True)
    with pytest.raises(ValueError, match="exclusif"):
        idx.search("pose", k=1, fusion=True, rerank=True)


def test_fusion_sans_model2vec_au_runtime_refuse(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, hybride=True)
    monkeypatch.setattr(rerank, "StaticModel", None)
    with pytest.raises(ValueError, match="model2vec"):
        idx.search("pose", k=1, fusion=True)


def test_fusion_avec_type_filtre(tmp_path, monkeypatch):
    """La fusion passe par le même pipeline facettes que la recherche normale."""
    _fake_model(monkeypatch)
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, hybride=True)
    hits = idx.search("pose", k=3, fusion=True, type_filtre="note texte")
    assert hits  # tous les docs du corpus sont des .md -> type « note texte »


# -- CLI (vrai model2vec, comme les tests --rerank de test_cli) -----------------------------


def _run(*args: str) -> str:
    r = subprocess.run(
        [*PY, *args], capture_output=True, text=True, encoding="utf-8", check=True
    )
    return r.stdout


def test_cli_build_hybride_puis_search_fusion(tmp_path):
    pytest.importorskip("model2vec")
    corpus = _corpus(tmp_path)
    idx = str(tmp_path / "idx")
    _run("build", str(corpus), "-o", idx, "--grid", "32x32", "--hybride")
    assert (Path(idx) / "bm25.msbm").is_file()
    assert (Path(idx) / "rerank.msrv").is_file()
    out = _run("search", "xkr914", idx, "--top", "1", "--fusion")
    hits = json.loads(out)
    assert hits and hits[0]["id"] == "devis_xkr914.md" and "rangs" in hits[0]


def test_cli_fusion_sans_hybride_echoue_proprement(tmp_path):
    corpus = _corpus(tmp_path)
    idx = str(tmp_path / "idx")
    _run("build", str(corpus), "-o", idx, "--grid", "32x32")
    r = subprocess.run(
        [*PY, "search", "pose", idx, "--fusion"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode != 0 and "hybride" in (r.stderr + r.stdout)
