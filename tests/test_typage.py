"""Routeur des grilles typées (v4) : routage déterministe, config par profil, dims auto."""

from mosaic.typage import (
    DIM_MAX_AUTO,
    TYPES_DEFAUT,
    config_grilles,
    dim_effective,
    router,
    router_flux,
)


def test_router_ref_et_sens():
    assert router("a9f77216") == "ref"  # mixte alphanum >= 5
    assert router("086027045054") == "ref"  # chiffres >= 6
    assert router("disjoncteur") == "sens"
    assert router("16a") == "sens"  # calibre, jamais une réf (règle moteur)


def test_router_regles_refs_du_profil():
    assert router("ab12", {"min_mixte": 4}) == "ref"
    assert router("ab12") == "sens"


def test_router_flux_provenance_chemin():
    config = config_grilles(None)
    flux = router_flux(["pose", "a9f77216"], ["devis", "2026"], config)
    assert flux["sens"] == ["pose"]
    assert flux["ref"] == ["a9f77216"]
    # provenance : « 2026 » irait en sens par la règle, mais le chemin prime
    assert flux["chemin"] == ["devis", "2026"]


def test_config_grilles_defauts():
    config = config_grilles(None)
    assert set(config) == set(TYPES_DEFAUT)
    assert config["ref"]["poids"] == (1.0, 0.0, 0.0)
    assert config["ref"]["lissage"] == 0  # JAMAIS lisser des identifiants
    assert config["sens"]["poids"] is None  # hérite de l'index


def test_config_grilles_surcharge_et_type_custom():
    profil = {
        "grilles": {
            "ref": {"dim": 2048},
            "norme": {"motif": r"nfc[0-9]+", "dim": 512},
        }
    }
    config = config_grilles(profil)
    assert config["ref"]["dim"] == 2048
    assert config["ref"]["poids"] == (1.0, 0.0, 0.0)  # le reste des défauts tenu
    assert config["norme"]["dim"] == 512
    flux = router_flux(["nfc15100", "cable", "a9f77216"], [], config)
    assert flux["norme"] == ["nfc15100"]
    assert flux["ref"] == ["a9f77216"]
    assert flux["sens"] == ["cable"]


def test_dim_effective_grandit_par_cote_jusqu_a_la_borne():
    assert dim_effective(768, 200) == 768  # vocab confortable : inchangé
    assert dim_effective(768, 600) == 3072  # vocab > dim/2 : côté doublé (dim x4)
    assert dim_effective(768, 2000) == 12288  # 768 -> 3072 (2000>1536) -> 12288
    assert dim_effective(768, 10**6) == DIM_MAX_AUTO  # borné, jamais au-delà


def test_dim_effective_signature_pure_jamais_de_croissance():
    """Une grille en signature pure (ref) n'a pas de profils à loger : sa capacité est
    la superposition par document, jamais le vocabulaire global (mesuré : sans cette
    règle, ref gonflait à 12288 sur un corpus riche en tokens numériques)."""
    assert dim_effective(768, 10**6, (1.0, 0.0, 0.0)) == 768
    assert dim_effective(768, 600, (0.6, 0.4, 0.0)) == 3072  # cooc apprise : grandit
    assert dim_effective(768, 600, None) == 3072  # poids hérités de l'index : grandit


def test_grille_de_dim_famille():
    from mosaic.typage import grille_de_dim

    assert grille_de_dim(3072) == (32, 32, 3)
    assert grille_de_dim(768) == (16, 16, 3)
    import pytest

    with pytest.raises(ValueError, match="hors famille"):
        grille_de_dim(1024)


def test_grille_custom_sans_motif_jamais_routee():
    config = config_grilles({"grilles": {"muette": {"dim": 256}}})
    flux = router_flux(["cable"], [], config)
    assert flux["muette"] == []  # présente mais jamais alimentée par le routage
