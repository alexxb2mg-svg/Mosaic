"""Tests — boucle de résolution de bout en bout (src/mosaic/resolution.py).

Corpus SYNTHÉTIQUE (aucune donnée client) : prouve le cycle croyance contestée -> recherche de
preuve dans l'index -> ré-assertion de la valeur gagnante -> croyance résolue, ET le garde-fou
« non tranché » quand la preuve est trop mince.
"""

from mosaic.croyance import MemoireCroyance
from mosaic.index import Index
from mosaic.resolution import resoudre

GRID = (32, 32, 3)
DIM = 512


def _index_termine(tmp_path):
    """Corpus où CHT_DEMO est TERMINÉ : 3 docs riches (réception/PV/facture) + 1 vieux 'en cours'."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "reception.md").write_text(
        "CHT_DEMO reception des travaux proces-verbal signe chantier livre client "
        "facture solde termine acheve",
        encoding="utf-8",
    )
    (c / "pv.md").write_text(
        "CHT_DEMO PV reception signe garantie parfait achevement termine livre solde "
        "reglement final",
        encoding="utf-8",
    )
    (c / "facture.md").write_text(
        "CHT_DEMO facture finale emise chantier termine solde paiement recu cloture livre",
        encoding="utf-8",
    )
    (c / "vieux.md").write_text("CHT_DEMO pose tableau travaux cours", encoding="utf-8")
    return Index.build(
        c, tmp_path / "idx", grid=GRID, index_paths=False, smoothing_rank=0
    )


FORMULATIONS = {
    "termine": "reception proces-verbal signe livre facture solde acheve",
    "en_cours": "pose tableau travaux cours",
}


def test_resolution_tranche_par_la_preuve(tmp_path):
    """Deux agents assertent des états contradictoires au MÊME instant -> contesté. La boucle
    cherche la preuve dans le corpus, 'termine' est mieux corroboré, ré-asserté, résolu."""
    idx = _index_termine(tmp_path)
    mem = MemoireCroyance(dim=DIM)
    mem.asserter("CHT_DEMO", "etat", "en_cours", t=5)
    mem.asserter("CHT_DEMO", "etat", "termine", t=5)  # même t -> se disputent
    assert mem.courant("CHT_DEMO", "etat")["a_preciser"]  # incertain AVANT

    rap = resoudre(mem, "CHT_DEMO", "etat", idx, formulations=FORMULATIONS)

    assert rap["statut"] == "resolu"
    assert rap["valeur"] == "termine"  # la preuve la mieux corroborée l'emporte
    assert rap["marge"] > 0.05
    # la croyance est désormais NETTE (ré-assertion plus récente que le litige)
    apres = mem.courant("CHT_DEMO", "etat")
    assert apres["valeur"] == "termine"
    assert not apres["conteste"]
    assert not apres["a_preciser"]


def test_resolution_sabstient_si_preuve_mince(tmp_path):
    """Garde-fou : preuves équilibrées (deux états également peu ancrés) -> 'non tranché', jamais
    un pari silencieux. La croyance reste contestée, à escalader."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "a.md").write_text("CHT_X etat alpha situation", encoding="utf-8")
    (c / "b.md").write_text("CHT_X etat beta situation", encoding="utf-8")
    idx = Index.build(
        c, tmp_path / "idx", grid=GRID, index_paths=False, smoothing_rank=0
    )
    mem = MemoireCroyance(dim=DIM)
    mem.asserter("CHT_X", "etat", "alpha", t=1)
    mem.asserter("CHT_X", "etat", "beta", t=1)

    rap = resoudre(
        mem,
        "CHT_X",
        "etat",
        idx,
        formulations={"alpha": "alpha situation", "beta": "beta situation"},
    )

    assert rap["statut"] == "non_tranche"
    assert rap["marge"] < 0.05
    assert mem.courant("CHT_X", "etat")["conteste"]  # reste contesté, non deviné


def test_resolution_noop_si_deja_net(tmp_path):
    """Une croyance déjà nette n'est pas touchée : statut 'deja_net', aucune ré-assertion."""
    idx = _index_termine(tmp_path)
    mem = MemoireCroyance(dim=DIM)
    mem.asserter("CHT_DEMO", "etat", "en_cours", t=0)
    mem.asserter("CHT_DEMO", "etat", "termine", t=1)  # plus récent, net
    avant = len(mem.historique("CHT_DEMO", "etat"))
    rap = resoudre(mem, "CHT_DEMO", "etat", idx, formulations=FORMULATIONS)
    assert rap["statut"] == "deja_net"
    assert rap["valeur"] == "termine"
    assert len(mem.historique("CHT_DEMO", "etat")) == avant  # rien ré-asséré


def test_resolution_emplacement_inconnu(tmp_path):
    idx = _index_termine(tmp_path)
    mem = MemoireCroyance(dim=DIM)
    rap = resoudre(mem, "RIEN", "rien", idx)
    assert rap["statut"] == "inconnu"
