"""Tests — calibration niveau 3 (src/mosaic/calibration.py) : les poids par la mesure."""

import pytest

from mosaic.calibration import (
    GRILLE_STANDARD,
    calibrer,
    expliquer_calibration,
)
from mosaic.encoder import WEIGHTS_DEFAULT

GRID = (32, 32, 3)


def _corpus(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir(exist_ok=True)
    (c / "disjoncteurs.md").write_text(
        "disjoncteur differentiel 30ma protection tableau electrique calibre",
        encoding="utf-8",
    )
    (c / "eclairage.md").write_text(
        "eclairage led spots luminaires couloirs detection presence",
        encoding="utf-8",
    )
    (c / "chauffage.md").write_text(
        "chauffage radiateurs inertie thermostat zones puissance",
        encoding="utf-8",
    )
    return c


REQUETES = [
    {"query": "protection differentielle tableau", "relevant": ["disjoncteurs.md"]},
    {"query": "spots led detection", "relevant": ["eclairage.md"]},
    {"query": "radiateurs thermostat", "relevant": ["chauffage.md"]},
]


def test_calibrer_rapport_complet(tmp_path):
    """Le sweep tourne, le défaut est TOUJOURS dans le classement, le rapport est complet et
    le verdict de fiabilité honnête (3 requêtes < 10 -> non fiable)."""
    rapport = calibrer(
        _corpus(tmp_path),
        REQUETES,
        grille=[(0.5, 0.3, 0.2)],
        grid=GRID,
        smoothing_rank=0,
    )
    assert rapport["n_documents"] == 3
    assert rapport["fiable"] is False  # 3 requêtes < seuil : signalé, pas caché
    poids_vus = {tuple(r["weights"]) for r in rapport["classement"]}
    assert WEIGHTS_DEFAULT in poids_vus  # le défaut est toujours la référence
    assert rapport["recommandation"] in ("changer", "garder_defaut")
    assert "gain_mrr_vs_defaut" in rapport


def test_calibrer_valide_les_requetes(tmp_path):
    with pytest.raises(ValueError, match="relevant"):
        calibrer(_corpus(tmp_path), [{"query": "x"}], grid=GRID, smoothing_rank=0)


def test_expliquer_calibration_fr_en(tmp_path):
    rapport = calibrer(
        _corpus(tmp_path),
        REQUETES,
        grille=[(0.5, 0.3, 0.2)],
        grid=GRID,
        smoothing_rank=0,
    )
    fr = expliquer_calibration(rapport, langue="fr")
    assert "Recommandation" in fr and "MRR" in fr
    assert "ATTENTION" in fr  # < 10 requêtes -> l'avertissement de fiabilité est dit
    en = expliquer_calibration(rapport, langue="en")
    assert "Recommendation" in en and "WARNING" in en


def test_grille_standard_contient_le_defaut():
    assert (
        GRILLE_STANDARD[0] == WEIGHTS_DEFAULT
    )  # la référence d'abord, toujours comparée


def _corpus_long(tmp_path):
    """Docs assez longs pour le held-out (>= 40 tokens chacun, deux moitiés qui partagent le
    sujet sans partager toutes les phrases)."""
    c = tmp_path / "corpus"
    c.mkdir(exist_ok=True)
    (c / "electricite.md").write_text(
        "le disjoncteur differentiel protege le circuit contre les fuites de courant "
        "le tableau electrique regroupe les protections modulaires par rangee "
        "chaque depart alimente une zone precise du batiment avec sa section de cable "
        "la protection differentielle trente milliamperes reste obligatoire pour les prises "
        "un interrupteur sectionneur permet la coupure generale de l installation "
        "le calibre des disjoncteurs depend de la section des conducteurs du circuit",
        encoding="utf-8",
    )
    (c / "cuisine.md").write_text(
        "la pate feuilletee demande un beurre froid incorpore par tours successifs "
        "chaque tour de pliage cree des couches alternees de beurre et de detrempe "
        "le repos au froid entre les tours evite que le beurre ne fonde dans la pate "
        "une cuisson a four chaud fait lever les feuillets par la vapeur emprisonnee "
        "le feuilletage reussi se reconnait a ses couches fines et croustillantes "
        "la detrempe se prepare avec de la farine de l eau et une pincee de sel",
        encoding="utf-8",
    )
    return c


def test_verite_auto_held_out(tmp_path):
    """La vérité déterministe (sans LLM) : générée du corpus, chaque requête retrouve son
    document parent, et le rapport porte la mention de sa nature et de sa limite."""
    rapport = calibrer(
        _corpus_long(tmp_path),
        grille=[(0.5, 0.3, 0.2)],
        grid=GRID,
        smoothing_rank=0,
        verite_auto=True,
    )
    assert rapport["verite"] == "auto (held-out déterministe)"
    assert rapport["n_requetes"] == 2  # une par doc assez long
    # reproductibilité : deux runs -> même rapport (déterminisme de la génération)
    rapport2 = calibrer(
        _corpus_long(tmp_path),
        grille=[(0.5, 0.3, 0.2)],
        grid=GRID,
        smoothing_rank=0,
        verite_auto=True,
    )
    assert rapport == rapport2
    # la limite est dite dans les deux langues
    assert "étalon-or" in expliquer_calibration(rapport, langue="fr")
    assert "gold standard" in expliquer_calibration(rapport, langue="en")


def test_verite_auto_exclusif_avec_requetes(tmp_path):
    with pytest.raises(ValueError, match="verite_auto"):
        calibrer(_corpus_long(tmp_path), REQUETES, verite_auto=True, grid=GRID)
    with pytest.raises(ValueError, match="requêtes-vérité|verite_auto"):
        calibrer(_corpus_long(tmp_path), None, grid=GRID)


def test_calibrer_index_paths_false_ignore_les_tokens_de_chemin(tmp_path):
    """`index_paths=False` doit calibrer sur le MÊME espace qu'un build --no-path-tokens :
    une requête qui ne matche QUE le nom de fichier trouve avec les tokens de chemin,
    plus rien sans eux — si les deux modes rendaient le même score, le paramètre serait
    décoratif et la calibration resterait hors-phase avec le build."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "zzximo.md").write_text("contenu generique sans le terme", encoding="utf-8")
    (c / "autre.md").write_text("document voisin egalement generique", encoding="utf-8")
    requetes = [{"query": "zzximo", "relevant": ["zzximo.md"]}]

    avec = calibrer(c, requetes, grid=GRID, smoothing_rank=0)
    sans = calibrer(c, requetes, grid=GRID, smoothing_rank=0, index_paths=False)
    assert avec["defaut"]["mrr"] > sans["defaut"]["mrr"]
