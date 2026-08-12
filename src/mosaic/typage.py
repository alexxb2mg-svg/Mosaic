"""Typage des tokens — le routeur des grilles typées (spec v4, idée Alex 12/08).

Chaque token d'un document est routé vers UNE grille selon son type. Le routage est
déterministe et se fait à l'ÉCRITURE (l'encodage est un aller simple : une fois
superposé dans une grille mélangée, le type n'est plus récupérable — c'est la raison
d'être du tri au build, cf. spec 2026-08-12-grilles-typees-v4-design.md).

Ordre de routage (première règle gagnante) :
  1. provenance : les tokens de CHEMIN sont routés « chemin » par l'appelant (ils ne
     passent jamais par cette fonction — la provenance prime sur toute règle) ;
  2. types custom du profil (clé `grilles`, motif regex fullmatch, ordre de déclaration) ;
  3. « ref » : la règle `_est_ref` du moteur (profil-aware, même critère que les facettes) ;
  4. « sens » : tout le reste.

Chaque grille porte SA recette d'encodage (spec §2-3, leçons mesurées) :
  - poids : `ref` = signature quasi pure (un identifiant est lexical — jamais
    d'embeddings ni de cooccurrence qui fabriqueraient des confusions) ;
  - lissage : JAMAIS lisser des identifiants (rapprocher deux réfs voisines = créer
    des collisions) — rang 0 pour ref/chemin, rang de l'index pour sens ;
  - dimension : taillée au vocabulaire réel, mesurée au build (`dim_effective`).
"""

import re

from mosaic.facettes import _est_ref

# Recettes par défaut des trois types de base. `poids`/`lissage` à None = hérite des
# réglages de l'index (la grille sens EST la grille historique, embeddings compris).
# Les dims restent dans la famille (c, c, 3) — côtés 8/16/32/64, dims 192/768/3072/
# 12288 : chaque grille typée demeure une VRAIE grille (rendu PNG, métaphore intacte).
TYPES_DEFAUT: dict[str, dict] = {
    "sens": {"dim": 3072, "poids": None, "lissage": None, "embeddings": True},
    "ref": {"dim": 768, "poids": (1.0, 0.0, 0.0), "lissage": 0, "embeddings": False},
    "chemin": {
        "dim": 768,
        "poids": (0.6, 0.4, 0.0),
        "lissage": 0,
        "embeddings": False,
    },
}
# borne de la croissance automatique : la dimension de la grille unique historique
DIM_MAX_AUTO = 12288
# préséance de la lecture ref : un token « ref » présent dans <= 2 documents est un
# IDENTIFIANT ; au-delà c'est un descripteur technique partagé (u1000r2v) — préséance
# aveugle mesurée nuisible. Courbe mesurée (500 articles réels, moteur v4) :
#   df<=3 -> désignation 0.925 / noyade 0.9083 ; df<=2 -> 0.9333 / 0.90 ;
#   df<=1 -> 0.9417 / 0.90. Défaut 2 : tient la barre de la spec (désignation >= 0.93)
# ET garde la préséance pour une réf vivant dans DEUX documents (le cas métier
# courant : le même code sur un BL et sa facture).
DF_MAX_IDENTIFIANT = 2

_CLES_GRILLE = {"dim", "poids", "lissage", "embeddings", "motif"}


def config_grilles(profil: dict | None) -> dict[str, dict]:
    """Configuration des grilles : défauts + surcharges/ajouts du profil (clé `grilles`).

    Un type custom du profil DOIT porter un `motif` (sa règle de routage) ; les trois
    types de base ne peuvent surcharger que dim/poids/lissage/embeddings. L'ordre du
    dict rendu est l'ordre de routage des types custom (insertion du profil)."""
    config = {t: dict(v) for t, v in TYPES_DEFAUT.items()}
    for nom, brut in ((profil or {}).get("grilles") or {}).items():
        if nom in config:
            config[nom].update({k: v for k, v in brut.items() if k != "motif"})
        else:
            custom = dict(brut)
            custom.setdefault("poids", (1.0, 0.0, 0.0))
            custom.setdefault("lissage", 0)
            custom.setdefault("embeddings", False)
            custom.setdefault("dim", 1024)
            config[nom] = custom
    return config


def _motifs_custom(config: dict[str, dict]) -> list[tuple[str, "re.Pattern[str]"]]:
    return [
        (nom, re.compile(str(cfg["motif"])))
        for nom, cfg in config.items()
        if nom not in TYPES_DEFAUT and "motif" in cfg
    ]


def router(
    token: str,
    regles_refs: dict | None = None,
    motifs_custom: list[tuple[str, "re.Pattern[str]"]] | None = None,
) -> str:
    """Type d'un token de CONTENU (les tokens de chemin sont routés par provenance,
    jamais par cette fonction)."""
    for nom, motif in motifs_custom or ():
        if motif.fullmatch(token):
            return nom
    return "ref" if _est_ref(token, regles_refs) else "sens"


def router_flux(
    contenu: list[str],
    chemin: list[str],
    config: dict[str, dict],
    regles_refs: dict | None = None,
) -> dict[str, list[str]]:
    """Répartit un document (contenu canonique + tokens de chemin) entre les grilles.
    Chaque grille de la config reçoit sa liste (vide si rien ne la concerne)."""
    motifs = _motifs_custom(config)
    flux: dict[str, list[str]] = {t: [] for t in config}
    flux["chemin"] = list(chemin)
    for t in contenu:
        flux[router(t, regles_refs, motifs)].append(t)
    return flux


class GrilleTypee:
    """Une grille typée matérialisée : ses profils, sa matrice documents int8, ses
    normes, et sa recette (config). La grille « sens » n'est PAS portée par cette
    classe — elle est la grille historique de l'Index (mat/profiles principaux)."""

    def __init__(self, profiles, mat, norms, config: dict) -> None:
        self.profiles = profiles
        self.mat = mat
        self.norms = norms
        self.config = config

    @property
    def dim(self) -> int:
        return int(self.mat.shape[1])


def dim_effective(dim: int, vocab: int, poids: tuple | None = None) -> int:
    """Dimension taillée au vocabulaire réel (spec §4) : tant que le vocabulaire dépasse
    la moitié de la dimension, doubler le CÔTÉ de la grille (dim ×4 — la famille
    (c, c, 3) est préservée : 192/768/3072/12288) — borné à DIM_MAX_AUTO. Mesuré au
    build, rapporté dans stats() : l'auto-dimensionnement est visible, jamais silencieux.

    NE s'applique qu'aux grilles qui APPRENNENT la cooccurrence (poids[1] > 0, ou poids
    hérités de l'index) : le vocabulaire y loge des profils, il faut de la place. Une
    grille en SIGNATURE PURE (ref : poids (1,0,0)) n'a rien à loger — sa capacité se
    joue à la superposition PAR DOCUMENT (4 réfs tiennent large dans 768 dims), pas au
    vocabulaire global. Mesuré (banc privé réel) : sans cette règle, la grille ref
    gonflait à 12288 sur un corpus riche en tokens numériques (SIRET, dates, normes) —
    plus de mémoire que la grille unique, pour rien."""
    d = int(dim)
    if poids is not None and float(poids[1]) == 0.0:
        return d  # signature pure : jamais de croissance au vocabulaire
    while vocab > d // 2 and d < DIM_MAX_AUTO:
        d = min(d * 4, DIM_MAX_AUTO)
    return d


def grille_de_dim(dim: int) -> tuple[int, int, int]:
    """Le triplet (c, c, 3) d'une dimension de la famille — pour les formats de stockage
    et le rendu. Refuse loud une dimension hors famille (jamais un reshape silencieusement
    faux)."""
    cote = int((dim / 3) ** 0.5)
    if cote * cote * 3 != dim:
        raise ValueError(f"dimension {dim} hors famille (c, c, 3)")
    return (cote, cote, 3)
