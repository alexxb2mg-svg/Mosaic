"""Tests — magasin structurel (src/mosaic/structurel.py) : compter, ordonner, joindre.

Les questions d'AGRÉGATION (« combien », « le plus récent », « qui porte cette
référence ») ne sont pas des questions de similarité : aucun canal sémantique ne
peut y répondre, ils rendent des documents, jamais des comptes. Le magasin est
DÉRIVÉ des facettes de l'index — jamais une source à maintenir à la main.
"""

import json

import pytest

from mosaic.structurel import Magasin

FACETTES = {
    "Factures/2024/06-Juin/F24069901 client.pdf": {
        "type": "pdf numérique",
        "date": "2024-06-00",
        "refs": ["F24069901"],
    },
    "BL/FOURNISSEUR/2026/06-Juin/20260622_BL_9990001.pdf": {
        "type": "pdf numérique",
        "date": "2026-06-22",
        "refs": ["9990001", "9990004"],
    },
    "BL/FOURNISSEUR/2026/06-Juin/20260609_BL_9990002.pdf": {
        "type": "pdf numérique",
        "date": "2026-06-09",
        "refs": ["9990002"],
    },
    "BL/FOURNISSEUR/2026/07-Juillet/20260713_BL_9990003.pdf": {
        "type": "pdf numérique",
        "date": "2026-07-13",
        "refs": ["9990003", "9990004"],
    },
    "Chantier/photos/IMG_2043.jpg": {"type": "photo", "date": "0000-00-00"},
}


@pytest.fixture
def magasin(tmp_path):
    d = tmp_path / "idx"
    d.mkdir()
    (d / "facettes.json").write_text(json.dumps(FACETTES), encoding="utf-8")
    m = Magasin()
    m.charger("test", d)
    return m


def test_charger_rend_le_nombre_de_documents(tmp_path):
    d = tmp_path / "idx"
    d.mkdir()
    (d / "facettes.json").write_text(json.dumps(FACETTES), encoding="utf-8")
    m = Magasin()
    assert m.charger("test", d) == 5


def test_charger_index_sans_facettes_refuse_clairement(tmp_path):
    d = tmp_path / "vide"
    d.mkdir()
    m = Magasin()
    with pytest.raises(FileNotFoundError, match="facettes.json"):
        m.charger("test", d)


def test_compter_sans_filtre_rend_tout(magasin):
    assert magasin.compter("test") == 5


def test_compter_par_chemin(magasin):
    assert magasin.compter("test", chemin_contient="BL/FOURNISSEUR/2026/06-Juin") == 2
    assert magasin.compter("test", chemin_contient="BL/FOURNISSEUR") == 3


def test_compter_par_type_et_par_date(magasin):
    assert magasin.compter("test", type_doc="photo") == 1
    assert magasin.compter("test", date_prefixe="2026-06") == 2
    assert magasin.compter("test", date_prefixe="2026") == 3


def test_compter_filtres_cumulatifs(magasin):
    assert magasin.compter("test", chemin_contient="BL", date_prefixe="2026-07") == 1


def test_compter_index_inconnu_rend_zero_jamais_une_erreur(magasin):
    assert magasin.compter("absent") == 0


def test_plus_recents_ordonne_et_exclut_les_sans_date(magasin):
    top = magasin.plus_recents("test", k=3)
    assert [d for d, _ in top] == [
        "BL/FOURNISSEUR/2026/07-Juillet/20260713_BL_9990003.pdf",
        "BL/FOURNISSEUR/2026/06-Juin/20260622_BL_9990001.pdf",
        "BL/FOURNISSEUR/2026/06-Juin/20260609_BL_9990002.pdf",
    ]
    # la photo sans date ne remonte JAMAIS : « le plus récent » n'a pas de sens
    # pour un document dont on ignore la date — l'omettre plutôt que mentir
    assert all("IMG_2043" not in d for d, _ in magasin.plus_recents("test", k=99))


def test_plus_recents_respecte_les_filtres(magasin):
    # « 06-Juin » matche AUSSI Factures/2024/06-Juin : le filtre est un fragment de
    # chemin, pas un mois — c'est voulu, et l'ordre par date fait le reste.
    top = magasin.plus_recents("test", k=5, chemin_contient="06-Juin")
    assert len(top) == 3
    assert top[0][1] == "2026-06-22"
    assert top[-1][1] == "2024-06-00"
    # pour un vrai « juin 2026 », c'est le filtre de DATE qui répond
    assert magasin.compter("test", date_prefixe="2026-06") == 2


def test_documents_portant_ref_joint_et_ordonne(magasin):
    docs = magasin.documents_portant_ref("9990004")
    assert [d for _, d, _ in docs] == [
        "BL/FOURNISSEUR/2026/07-Juillet/20260713_BL_9990003.pdf",
        "BL/FOURNISSEUR/2026/06-Juin/20260622_BL_9990001.pdf",
    ]
    assert magasin.documents_portant_ref("inexistante") == []


def test_repartition_par_mois(magasin):
    rep = magasin.repartition_par_mois("test", chemin_contient="BL")
    assert rep == {"2026-06": 2, "2026-07": 1}


def test_sans_date_compte_le_trou_de_couverture(magasin):
    assert magasin.sans_date("test") == 1


def test_deux_index_coexistent_sans_se_melanger(tmp_path):
    m = Magasin()
    for nom in ("a", "b"):
        d = tmp_path / nom
        d.mkdir()
        (d / "facettes.json").write_text(json.dumps(FACETTES), encoding="utf-8")
        m.charger(nom, d)
    assert m.compter("a") == 5
    assert m.compter("b") == 5
    assert m.compter("a", type_doc="photo") == 1
    # une référence traverse les index : la jointure les nomme tous les deux
    assert {idx for idx, _, _ in m.documents_portant_ref("9990001")} == {"a", "b"}


def test_rechargement_est_idempotent(tmp_path):
    """Recharger le même index ne DOUBLE pas les documents — le magasin est un
    dérivé, pas un journal : deux chargements = un seul état."""
    d = tmp_path / "idx"
    d.mkdir()
    (d / "facettes.json").write_text(json.dumps(FACETTES), encoding="utf-8")
    m = Magasin()
    m.charger("test", d)
    m.charger("test", d)
    assert m.compter("test") == 5
    assert len(m.documents_portant_ref("9990001")) == 1


def test_caractere_joker_sql_est_litteral(magasin):
    """`%` et `_` sont des jokers SQL : un chemin qui en contient ne doit pas
    élargir la recherche silencieusement."""
    assert magasin.compter("test", chemin_contient="BL%FOURNISSEUR") == 0
    assert magasin.compter("test", chemin_contient="IMG_2043") == 1
    assert magasin.compter("test", chemin_contient="IMGx2043") == 0


def test_types_disponibles_dit_le_vocabulaire_reel(magasin):
    """Un agent qui filtre sur un type inexistant reçoit zéro résultat et croit
    que le document n'existe pas. Pour se corriger, il lui faut le vocabulaire
    RÉEL du domaine — c'est ce que rend cette méthode, avec les effectifs."""
    assert magasin.types_disponibles("test") == {"pdf numérique": 4, "photo": 1}
    assert magasin.types_disponibles("domaine-inconnu") == {}
