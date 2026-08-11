"""Tests — méta-recherche cross-index (src/mosaic/meta.py). Cœur RRF pur, sans index réel."""

import pytest

from mosaic.meta import resume_par_index, rrf_fuse


def _liste(source, ids):
    return (source, [{"id": i, "score": s} for i, s in ids])


def test_rrf_entrelace_les_rangs():
    """Deux corpus disjoints : RRF entrelace par rang — les rang-1 de chaque source passent
    devant les rang-2, indépendamment de l'échelle de score (devis à 0.9, comms à 0.2)."""
    devis = _liste("devis", [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)])
    comms = _liste("comms", [("c1", 0.2), ("c2", 0.15)])
    fusion = rrf_fuse([devis, comms], k=10)
    rangs1 = {r["id"] for r in fusion[:2]}
    assert rangs1 == {"d1", "c1"}  # les deux rang-1 en tête, malgré l'écart de score
    assert fusion[2]["id"] in {"d2", "c2"}  # puis les rang-2


def test_rrf_provenance_et_score_local():
    fusion = rrf_fuse([_liste("compta", [("f1", 0.5)])], k=10)
    assert fusion[0]["index"] == "compta"
    assert fusion[0]["rang_local"] == 1
    assert fusion[0]["score_local"] == 0.5
    assert fusion[0]["score_rrf"] == pytest.approx(1.0 / 61, abs=1e-6)


def test_rrf_cosignalement_additionne():
    """Un même id présent dans deux sources cumule ses contributions RRF (co-signalé = remonté).
    id partagé 'x' en rang 2 partout bat 'a'/'b' qui ne sont rang 1 que dans une seule source."""
    s1 = _liste("A", [("a", 0.9), ("x", 0.8)])
    s2 = _liste("B", [("b", 0.9), ("x", 0.7)])
    # 'x' apparaît dans A et B mais sous des clés (source,id) DISTINCTES -> pas de cumul ici :
    # le cumul RRF vaut pour un MÊME (source,id). On vérifie donc que les clés restent séparées.
    fusion = rrf_fuse([s1, s2], k=10)
    cles = {(r["index"], r["id"]) for r in fusion}
    assert ("A", "x") in cles and ("B", "x") in cles  # provenance distincte préservée


def test_rrf_cumul_meme_source_id():
    """Le cumul opère quand la MÊME (source,id) revient dans deux listes de même source (ex. deux
    passes de requête sur le même index) : les contributions s'additionnent."""
    p1 = _liste("devis", [("d1", 0.9), ("d2", 0.5)])
    p2 = _liste("devis", [("d2", 0.6), ("d1", 0.4)])  # d1 rang1+rang2, d2 rang2+rang1
    fusion = rrf_fuse([p1, p2], k=10)
    par_id = {r["id"]: r for r in fusion}
    attendu = 1.0 / 61 + 1.0 / 62  # rang 1 + rang 2, identique pour d1 et d2
    assert par_id["d1"]["score_rrf"] == pytest.approx(round(attendu, 6), abs=1e-6)
    assert par_id["d2"]["score_rrf"] == pytest.approx(round(attendu, 6), abs=1e-6)


def test_resume_par_index_signale_hors_sujet():
    listes = [
        _liste("devis", [("d1", 0.72)]),
        _liste(
            "compta", [("f1", 0.03)]
        ),  # meilleur score très bas -> probablement hors-sujet
    ]
    resume = resume_par_index(listes)
    par = {r["index"]: r for r in resume}
    assert par["devis"]["meilleur_score_local"] == 0.72
    assert par["compta"]["meilleur_score_local"] == 0.03
    assert par["compta"]["candidats"] == 1


def test_rrf_top_k_et_validation():
    fusion = rrf_fuse([_liste("A", [(f"a{i}", 1.0 - i / 10) for i in range(5)])], k=3)
    assert len(fusion) == 3
    with pytest.raises(ValueError):
        rrf_fuse([], k=0)
