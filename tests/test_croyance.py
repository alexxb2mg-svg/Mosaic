"""Tests — mémoire de croyance VSA (src/mosaic/croyance.py) + CLI."""

import json
import subprocess
import sys

from mosaic.croyance import MemoireCroyance

DIM = 512
PY = [sys.executable, "-m", "mosaic.cli"]


def test_courant_supersession():
    m = MemoireCroyance(dim=DIM)
    m.asserter("disjoncteur_roger", "etat", "defectueux", t=0)
    m.asserter("disjoncteur_roger", "etat", "repare", t=1)
    c = m.courant("disjoncteur_roger", "etat")
    assert c["valeur"] == "repare"  # valeur EXACTE (fait le plus récent)
    assert not c["conteste"]
    assert not c["a_preciser"]  # net -> pas besoin de préciser
    assert c["confiance"] > 0.1


def test_historique_ordonne():
    m = MemoireCroyance(dim=DIM)
    m.asserter("e", "a", "v2", t=2)
    m.asserter("e", "a", "v1", t=1)  # asserté après mais t antérieur
    h = m.historique("e", "a")
    assert [x["valeur"] for x in h] == ["v1", "v2"]  # trié par t


def test_conteste_meme_t():
    m = MemoireCroyance(dim=DIM)
    m.asserter("client", "ville", "orleans", t=5)
    m.asserter("client", "ville", "chartres", t=5)  # même t -> se disputent
    c = m.courant("client", "ville")
    assert c["conteste"]
    assert c["a_preciser"]  # incertain -> invite à préciser
    assert c["valeur"] is None  # aucune valeur exacte ne tranche
    assert set(c["candidats"]) == {
        "orleans",
        "chartres",
    }  # les concurrentes, pour la question
    assert c["confiance"] < 0.05


def test_emplacement_inconnu():
    m = MemoireCroyance(dim=DIM)
    assert m.courant("rien", "rien") is None
    assert m.historique("rien", "rien") == []


def test_save_load_deterministe(tmp_path):
    m = MemoireCroyance(dim=DIM)
    m.asserter("client", "ville", "orleans", t=0)
    m.asserter("client", "ville", "saran", t=1)
    m.asserter("client", "statut", "actif", t=0)
    p = tmp_path / "croyance.jsonl"
    m.sauver(p)
    m2 = MemoireCroyance.charger(p, dim=DIM)
    assert m2.courant("client", "ville") == m.courant("client", "ville")
    assert m2.courant("client", "statut") == m.courant("client", "statut")
    assert m2.historique("client", "ville") == m.historique("client", "ville")


def test_cli_croyance(tmp_path):
    store = str(tmp_path / "cr.jsonl")

    def run(*a):
        return subprocess.run(
            [*PY, "croyance", *a], capture_output=True, text=True, encoding="utf-8"
        )

    run(
        "assert",
        store,
        "--entite",
        "e",
        "--attribut",
        "a",
        "--valeur",
        "v1",
        "--t",
        "0",
    )
    run(
        "assert",
        store,
        "--entite",
        "e",
        "--attribut",
        "a",
        "--valeur",
        "v2",
        "--t",
        "1",
    )
    r = run("courant", store, "--entite", "e", "--attribut", "a")
    assert json.loads(r.stdout)["valeur"] == "v2"
    h = run("historique", store, "--entite", "e", "--attribut", "a")
    assert [x["valeur"] for x in json.loads(h.stdout)] == ["v1", "v2"]


def test_determinisme_cross_instance():
    # même symbole -> même vecteur -> même réponse, sur deux instances distinctes
    a = MemoireCroyance(dim=DIM)
    b = MemoireCroyance(dim=DIM)
    for mem in (a, b):
        mem.asserter("x", "y", "alpha", t=0)
        mem.asserter("x", "y", "beta", t=1)
    assert a.courant("x", "y") == b.courant("x", "y")


def test_faisceau_incremental_deterministe():
    """L'assert incrémental (O(1)) est BYTE-EXACT et reproductible : la même séquence
    monotone bâtie sur deux instances donne le faisceau identique au bit près."""
    import numpy as np

    vals = ["en_cours", "en_attente", "termine", "bloque"]
    a, b = MemoireCroyance(dim=DIM), MemoireCroyance(dim=DIM)
    for mem in (a, b):
        for i in range(
            200
        ):  # historique profond sur UN emplacement (chemin incrémental)
            mem.asserter("E", "etat", vals[i % 4], t=float(i))
    assert np.array_equal(a._acc[("E", "etat")], b._acc[("E", "etat")])
    assert (
        a.courant("E", "etat")["valeur"] == "bloque"
    )  # dernier fait (t=199 -> 199%4=3)


def test_calibrer_conteste_conforme():
    """Le seuil « contesté » est CALIBRÉ sur le store lui-même (vérité-terrain = le canal
    exact) : après calibration à alpha, l'erreur des slots « confiants » est <= alpha, et
    courant() utilise le seuil d'instance. Déterministe : recalibrer rend le même seuil."""
    import pytest

    m = MemoireCroyance(dim=DIM)
    vals = ["a", "b", "c", "d"]
    # 30 slots NETS (une valeur dominante, marge haute, VSA correct)
    for i in range(30):
        m.asserter(f"N{i}", "etat", vals[i % 4], t=0)
        m.asserter(f"N{i}", "etat", vals[i % 4], t=1)  # renforcée -> marge nette
    # 10 slots PIÉGÉS : une valeur répétée 3x puis CHANGÉE — la superposition favorise
    # encore l'ancienne (0.6+0.36+0.216 = 1.176 > 1.0) -> le VSA se TROMPE, marge faible.
    # C'est exactement ce que le seuil calibré doit apprendre à écarter.
    for i in range(10):
        for t in range(3):
            m.asserter(f"P{i}", "etat", vals[i % 4], t=float(t))
        m.asserter(f"P{i}", "etat", vals[(i + 1) % 4], t=3.0)

    rapport = m.calibrer_conteste(alpha=0.10)
    assert rapport["slots_exploites"] == 40
    assert rapport["erreur_sur_retenus"] <= 0.10  # la garantie conforme tient
    assert rapport["taux_reponse"] < 1.0  # les slots piégés (VSA faux) sont écartés
    assert (
        rapport["seuil_calibre"] > 0.05
    )  # le seuil a monté au-dessus de l'heuristique
    assert m.seuil_conteste == rapport["seuil_calibre"]  # courant() l'utilisera
    # et courant() sur un slot piégé lève désormais a_preciser (marge sous le seuil calibré)
    c = m.courant("P0", "etat")
    assert c is not None and c["a_preciser"]
    # déterminisme : recalibrer sur les mêmes faits rend le même seuil
    assert m.calibrer_conteste(alpha=0.10)["seuil_calibre"] == rapport["seuil_calibre"]

    # garde-fou : trop peu de slots -> refus loud
    petit = MemoireCroyance(dim=DIM)
    petit.asserter("x", "y", "v1", t=0)
    petit.asserter("x", "y", "v2", t=1)
    with pytest.raises(ValueError, match="fiable"):
        petit.calibrer_conteste()


def test_round_trip_historique_profond_et_hors_ordre(tmp_path):
    """Persistance exacte même avec un slot à historique profond (chemin incrémental) ET un
    slot construit hors-ordre (repli _recompute) : charger reproduit courant() à l'identique."""
    m = MemoireCroyance(dim=DIM)
    for i in range(300):
        m.asserter("E", "phase", ["etude", "chantier"][i % 2], t=float(i))
    m.asserter("E", "prio", "haute", t=5.0)
    m.asserter("E", "prio", "basse", t=2.0)  # hors-ordre -> _recompute
    p = tmp_path / "cr.jsonl"
    m.sauver(p)
    m2 = MemoireCroyance.charger(p, dim=DIM)
    assert m2.courant("E", "phase") == m.courant("E", "phase")
    assert m2.courant("E", "prio") == m.courant("E", "prio")
    assert m.courant("E", "prio")["valeur"] == "haute"  # t=5 domine t=2
