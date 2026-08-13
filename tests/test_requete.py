"""Prétraitement déterministe de la requête (`mosaic.requete`).

Ce que ces tests protègent : le nettoyage doit retirer le bruit conversationnel
SANS jamais toucher aux termes porteurs, et sans jamais rendre une requête vide.
Les cas viennent de requêtes réelles du banc Alloprof.
"""

from mosaic.requete import nettoyer, taux_de_bruit


def test_requete_reelle_alloprof():
    """Cas réel du banc : 15 mots de bruit avant le premier terme utile."""
    brut = (
        "Bonjour! Je voulais d'aide avec ces question je n'arrive pas a comprendre "
        "comment les faire Vrai ou Faux - entiers relatifs"
    )
    net, retires = nettoyer(brut)
    assert "entiers relatifs" in net
    assert "Vrai ou Faux" in net
    assert "Bonjour" not in net
    assert "arrive pas" not in net
    assert len(retires) >= 2


def test_termes_porteurs_intacts():
    """Aucun mot du domaine ne doit disparaître — c'est la garantie centrale."""
    brut = "bonjour svp je n'arrive pas à comprendre le disjoncteur différentiel 30mA"
    net, _ = nettoyer(brut)
    for terme in ("disjoncteur", "différentiel", "30mA"):
        assert terme in net


def test_jamais_de_requete_vide():
    """Une requête qui n'est QUE de la politesse est rendue INCHANGÉE : chercher
    du bruit vaut mieux que ne rien chercher."""
    for brut in ("bonjour", "merci beaucoup", "bonjour, s'il vous plaît, merci"):
        net, retires = nettoyer(brut)
        assert net == brut
        assert retires == []


def test_requete_propre_inchangee():
    """Une requête déjà nette ne doit subir AUCUNE transformation."""
    brut = "ALDH1 expression is associated with poorer prognosis for breast cancer"
    net, retires = nettoyer(brut)
    assert net == brut and retires == []


def test_requete_technique_francaise_inchangee():
    brut = "tableau divisionnaire garage disjoncteur différentiel 30mA en amont"
    net, retires = nettoyer(brut)
    assert net == brut and retires == []


def test_ponctuation_expressive_reduite():
    net, _ = nettoyer("comment câbler un va-et-vient ???")
    assert net.endswith("?") and "???" not in net


def test_meta_discours_retire():
    net, retires = nettoyer("Ma question est : comment dimensionner un câble ?")
    assert "dimensionner" in net and "câble" in net
    assert "question est" not in net.lower()
    assert retires


def test_incomprehension_retiree():
    net, _ = nettoyer("j'ai du mal à comprendre le théorème de Thalès")
    assert "Thalès" in net and "théorème" in net
    assert "mal à comprendre" not in net


def test_entrees_degenerees():
    for brut in ("", "   ", "?"):
        net, retires = nettoyer(brut)
        assert net == brut and retires == []


def test_taux_de_bruit_diagnostic():
    """Le diagnostic doit distinguer un corpus bruité d'un corpus propre — c'est
    lui qui dira si le prétraitement mérite d'être activé sur un corpus donné."""
    bruitee = "Bonjour, je n'arrive pas à comprendre les fractions, merci d'avance"
    propre = "conversion des fractions en nombres décimaux"
    assert taux_de_bruit(bruitee) > 0.3
    assert taux_de_bruit(propre) == 0.0
