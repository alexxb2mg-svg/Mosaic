"""Fusion par scores z-normalisés (`mosaic meta --fusion znorm`).

Ce que ces tests protègent : la fusion mesurée gagnante sur des sources HOMOGÈNES
(Alloprof découpé en 4 tranches équilibrées : 0.3832 R@10 / 0.2229 MRR contre
0.3216 / 0.2164 pour l'index unique) — et le fait que le RRF reste le défaut,
parce qu'il est la seule monnaie juste entre corpus hétérogènes.
"""

import pytest

from mosaic.meta import rrf_fuse, znorm_fuse


def test_znorm_rend_les_echelles_comparables():
    """Deux sources aux échelles très différentes : la source « basse » doit pouvoir
    placer son meilleur document devant les médiocres de la source « haute ».
    Le score brut concaténé laisserait la source haute tout rafler."""
    haute = [
        {"id": "h1", "score": 0.90},
        {"id": "h2", "score": 0.88},
        {"id": "h3", "score": 0.87},
    ]
    basse = [
        {"id": "b1", "score": 0.30},
        {"id": "b2", "score": 0.10},
        {"id": "b3", "score": 0.09},
    ]
    top = znorm_fuse([("haute", haute), ("basse", basse)], k=2)
    assert {r["id"] for r in top} == {"h1", "b1"}


def test_znorm_conserve_la_provenance_et_le_score_local():
    res = znorm_fuse(
        [("A", [{"id": "x", "score": 0.5}, {"id": "y", "score": 0.1}])], k=1
    )
    assert res[0]["index"] == "A"
    assert res[0]["rang_local"] == 1
    assert res[0]["score_local"] == 0.5
    assert "score_z" in res[0]


def test_znorm_source_a_scores_identiques_ne_divise_pas_par_zero():
    """σ = 0 : rien à corriger, et surtout aucune division par zéro."""
    plate = [{"id": "a", "score": 0.4}, {"id": "b", "score": 0.4}]
    res = znorm_fuse([("plate", plate)], k=2)
    assert len(res) == 2 and all(r["score_z"] == 0.0 for r in res)


def test_znorm_ignore_une_source_vide():
    res = znorm_fuse([("vide", []), ("pleine", [{"id": "p", "score": 1.0}])], k=5)
    assert [r["id"] for r in res] == ["p"]


def test_znorm_refuse_k_invalide():
    with pytest.raises(ValueError, match="k doit être"):
        znorm_fuse([("A", [{"id": "a", "score": 1.0}])], k=0)


def test_rrf_reste_le_defaut_et_differe_de_znorm():
    """Les deux fusions ne sont PAS interchangeables. Le RRF ne voit que les rangs :
    les deux têtes de liste sont à égalité, quoi qu'elles valent. La z-norm regarde
    combien un document DÉTACHE de ses voisins : ici b1 domine largement sa source
    (+1,37 σ) alors que a1 est au coude à coude avec a2 (+0,73 contre +0,71) — donc
    b1 passe devant. C'est le comportement voulu, et c'est aussi la source du biais
    mesuré : une petite source aux scores resserrés se hisse facilement."""
    a = [
        {"id": "a1", "score": 0.99},
        {"id": "a2", "score": 0.98},
        {"id": "a3", "score": 0.10},
    ]
    b = [
        {"id": "b1", "score": 0.50},
        {"id": "b2", "score": 0.20},
        {"id": "b3", "score": 0.10},
    ]
    par_rang = [r["id"] for r in rrf_fuse([("a", a), ("b", b)], k=6)]
    par_score = [r["id"] for r in znorm_fuse([("a", a), ("b", b)], k=6)]
    assert par_rang != par_score
    assert par_rang[0] == "a1"  # RRF : premier de la première source citée
    assert par_score[0] == "b1"  # z-norm : celui qui détache le plus de ses voisins
