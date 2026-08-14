"""CLI Mosaic — toutes les sorties en JSON UTF-8, consommables par les agents.

STRUCTURE (refonte 12/08, retombées de l'audit CLI — 34 findings) :
- un handler `_cmd_<nom>(args)` par sous-commande, routé par la table `_COMMANDES`
  (l'ancienne cascade if/elif de 315 lignes culminait à 86 de complexité cyclomatique) ;
- toutes les vérifications d'arguments AVANT tout `Index.open` — zéro I/O payée pour
  apprendre qu'un drapeau est invalide, et les combinaisons inopérantes sont REFUSÉES
  (jamais un drapeau ignoré en silence) ;
- `_ouvrir_index` : ce qui n'est pas un index Mosaic est refusé avec le geste de
  construction, jamais un errno brut ;
- socle de sortie : tout résultat classé porte `id` + `score` ; `{"ok", "out"}` pour
  les commandes qui produisent un fichier ; les erreurs sortent en `{"error": ...}`
  mono-ligne sur stderr, rc 1 — y compris les fichiers d'index corrompus/amputés.
"""

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

from mosaic import carte
from mosaic.croyance import MemoireCroyance
from mosaic.diff import diff_corpus, diff_indexes
from mosaic.embeddings import Embeddings, prepare
from mosaic.index import PROFILE_WEIGHTING_DEFAULT, SMOOTHING_RANK_DEFAULT, Index
from mosaic.lexicon import load_lexicon
from mosaic import profil as profil_module
from mosaic.meta import resume_par_index, rrf_fuse, znorm_fuse
from mosaic.render import render_doc
from mosaic.temporel import SEUIL_VERSION_DEFAUT, versions_actuelles

_GRIDS = {"64x64": (64, 64, 3), "32x32": (32, 32, 3)}

# Raccourci "wikdict" pour --lexicon-extra : le lexique importé embarqué dans le
# package si présent (committé, < 5 Mo), sinon celui laissé dans data_externes/
# (repo de dev uniquement — voir src/mosaic/data/LICENSE_wikdict.txt).
_WIKDICT_LEXICON_PACKAGE = (
    Path(__file__).resolve().parent / "data" / "lexicon_wikdict_fr_en.json"
)
_WIKDICT_LEXICON_EXTERNAL = Path("data_externes") / "lexicon_wikdict_fr_en.json"


def _out(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def _resolve_lexicon_extra(value: str) -> Path:
    if value != "wikdict":
        return Path(value)
    if _WIKDICT_LEXICON_PACKAGE.exists():
        return _WIKDICT_LEXICON_PACKAGE
    if _WIKDICT_LEXICON_EXTERNAL.exists():
        return _WIKDICT_LEXICON_EXTERNAL
    raise ValueError(
        f"lexique WikDict introuvable (ni {_WIKDICT_LEXICON_PACKAGE} ni {_WIKDICT_LEXICON_EXTERNAL})"
    )


def _parse_weights(value: str) -> tuple[float, float, float]:
    """Parse "a,b,g" (ex. "0.25,0.15,0.60") en triplet de poids > 0."""
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(
            f"--weights doit contenir exactement 3 valeurs séparées par des virgules "
            f"(ex. 0.25,0.15,0.60) : {value!r}"
        )
    try:
        a, b, g = (float(p) for p in parts)
    except ValueError:
        raise ValueError(
            f"--weights doit contenir 3 nombres valides : {value!r}"
        ) from None
    if a <= 0 or b <= 0 or g <= 0:
        raise ValueError(
            f"--weights doit contenir 3 valeurs strictement positives : {value!r}"
        )
    return (a, b, g)


def _parse_profile_weighting(value: str) -> str:
    """Valide --profile-weighting : choix brut|ppmi."""
    if value not in ("brut", "ppmi"):
        raise ValueError(f"--profile-weighting doit être 'brut' ou 'ppmi' : {value!r}")
    return value


def _parse_int_nonnegative(value: int, param_name: str) -> int:
    """Valide un entier >= 0 (argparse fournit déjà l'int)."""
    if value < 0:
        raise ValueError(f"--{param_name} doit être >= 0 : {value!r}")
    return value


def _parse_int_positive(value: int, param_name: str) -> int:
    """Valide un entier >= 1 (argparse fournit déjà l'int)."""
    if value < 1:
        raise ValueError(f"--{param_name} doit être >= 1 : {value!r}")
    return value


def _parse_abtt(value: int) -> int:
    """Valide --abtt : entier dans [0, 255] — la borne du FORMAT (octet u8 du header
    .msee) est appliquée ICI, avant tout scan de corpus (elle ne tombait qu'après)."""
    if not (0 <= value <= 255):
        raise ValueError(f"--abtt doit être dans [0, 255] (octet u8) : {value!r}")
    return value


def _parse_doc_weight(value: float) -> float:
    """Valide --doc-weight : flottant dans [0, 1) (argparse fournit déjà le float)."""
    if not (0.0 <= value < 1.0):
        raise ValueError(f"--doc-weight doit être dans [0, 1) : {value!r}")
    return value


def _parse_rerank_lambda(value: float) -> float:
    """Valide --rerank-lambda : flottant dans [0, 1] (argparse fournit déjà le float)."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"--rerank-lambda doit être dans [0, 1] : {value!r}")
    return value


def _parse_rerank_depth(value: int) -> int:
    """Valide --rerank-depth : entier >= 10 (argparse fournit déjà l'int)."""
    if value < 10:
        raise ValueError(f"--rerank-depth doit être >= 10 : {value!r}")
    return value


def _parse_cos_01(value: float, param_name: str) -> float:
    """Valide un cosinus dans [0, 1] (--conn-lambda, --seuil-version)."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"--{param_name} doit être dans [0, 1] : {value!r}")
    return value


def _merge_extra_lexicon(core: dict[str, str], extra_path: Path) -> dict[str, str]:
    """Fusionne `extra_path` sous `core` (le noyau gagne) — équivalent CLI de
    load_lexicon(extra=...) quand `core` n'est pas issu d'un simple chemin de
    fichier (ex : --lexicon none -> noyau vide)."""
    extra_data = json.loads(extra_path.read_text(encoding="utf-8"))
    merged = {str(k).lower(): str(v).lower() for k, v in extra_data.items()}
    merged.update(core)
    return merged


def _ingest_cache_root() -> Path:
    """Racine du cache d'ingestion. Surchargeable par MOSAIC_INGEST_CACHE
    (isolation des tests / builds concurrents) ; défaut : temp système."""
    override = os.environ.get("MOSAIC_INGEST_CACHE")
    return Path(override) if override else Path(tempfile.gettempdir()) / "mosaic_ingest"


def _ouvrir_index(path: Path, **kwargs) -> Index:
    """Ouvre un index en refusant EN CLAIR ce qui n'en est pas un — l'errno brut
    (`[Errno 2] ... docs.msei`) n'est pas un message actionnable, et une typo d'ordre
    positionnel (requête prise pour un index) doit se lire immédiatement."""
    if not path.is_dir() or not (path / "docs.msei").is_file():
        raise ValueError(
            f"{path} n'est pas un index Mosaic (docs.msei absent) — construire avec "
            f"`mosaic build <corpus> -o {path}`"
        )
    return Index.open(path, **kwargs)


def _ajouter_rerank_params(p: argparse.ArgumentParser) -> None:
    """λ et profondeur du repêcheur — déclarés UNE fois pour search/like/meta
    (l'audit a montré la divergence : meta avait --rerank sans ses réglages)."""
    p.add_argument(
        "--rerank-lambda",
        type=float,
        default=0.70,
        help="λ du mélange λ·cos_mosaic + (1-λ)·cos_m2v, dans [0, 1]",
    )
    p.add_argument(
        "--rerank-depth",
        type=int,
        default=50,
        help="profondeur du reclassement (top-N mosaic re-classé), entier >= 10",
    )


def _ajouter_cache_ingestion(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cache-ingestion",
        action="store_true",
        help="cache opt-in des conversions markitdown sous tempfile.gettempdir()/mosaic_ingest/ "
        "(jamais à côté des documents, jamais activé par défaut, "
        "surchargeable par MOSAIC_INGEST_CACHE)",
    )


def _ajouter_top(p: argparse.ArgumentParser, defaut: int = 10) -> None:
    p.add_argument("--top", type=int, default=defaut, help="nombre de résultats rendus")


# ---------------------------------------------------------------------------------
# Parser


def _construire_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mosaic")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def ajouter(nom: str, aide: str) -> argparse.ArgumentParser:
        return sub.add_parser(
            nom,
            help=aide,
            description=aide,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    p_build = ajouter("build", "construit un index depuis un dossier de documents")
    p_build.add_argument("corpus", help="dossier de documents à indexer")
    p_build.add_argument("-o", "--output", required=True, help="dossier de l'index")
    p_build.add_argument(
        "--grid", choices=sorted(_GRIDS), default="64x64", help="géométrie de la grille"
    )
    p_build.add_argument(
        "--lexicon", default=None, help="lexique de canonicalisation (chemin ou 'none')"
    )
    p_build.add_argument(
        "--lexicon-extra",
        default=None,
        help="lexique additionnel fusionné sous le noyau (chemin ou 'wikdict')",
    )
    p_build.add_argument(
        "--embeddings", default=None, help="table d'embeddings .msee (canal γ)"
    )
    p_build.add_argument(
        "--weights",
        default=None,
        help="a,b,g (ex. 0.25,0.15,0.60) ; None = défauts calibrés du moteur",
    )
    p_build.add_argument(
        "--profile-weighting",
        default=PROFILE_WEIGHTING_DEFAULT,
        help="brut ou ppmi",
    )
    p_build.add_argument(
        "--smoothing-rank",
        type=int,
        default=SMOOTHING_RANK_DEFAULT,
        help="rang du lissage SVD, entier >= 0 (0 = sans lissage)",
    )
    p_build.add_argument(
        "--abtt", type=int, default=0, help="all-but-the-top, entier dans [0, 255]"
    )
    p_build.add_argument(
        "--doc-weight",
        type=float,
        default=None,
        help="δ, poids du canal document (niveau sujet) dans [0, 1), défaut 0 (v1.2 exact)",
    )
    _ajouter_cache_ingestion(p_build)
    p_build.add_argument(
        "--rerank-vectors",
        action="store_true",
        help="encode le texte de chaque document via model2vec (potion) dans rerank.msrv, "
        "pour permettre `search --rerank` (défaut désactivé, nécessite model2vec)",
    )
    p_build.add_argument(
        "--grilles-typees",
        action="store_true",
        help="index à GRILLES TYPÉES (v4) : chaque type de donnée dans SA grille "
        "(sens/réf/chemin, extensible par le profil), poids et lissage par grille, "
        "dimensions taillées au vocabulaire — la recherche route et synthétise "
        "(pondération idf + préséance identifiant). Mesuré sur le vrai moteur "
        "(500 articles réels) : noyade de réfs 0.825 -> 0.90, grille sens 4x plus "
        "petite quand le vocabulaire le permet (croissance auto sinon)",
    )
    p_build.add_argument(
        "--hybride",
        action="store_true",
        help="index hybride : stocke les postings BM25 (bm25.msbm) ET les vecteurs "
        "model2vec (implique --rerank-vectors) pour permettre `search --fusion` — la "
        "fusion RRF à trois canaux validée au banc Alloprof (0.517 R@10 contre 0.498 "
        "pour le standard BM25+embeddings)",
    )
    p_build.add_argument(
        "--atlas",
        action="store_true",
        help="canal atlas (#367, exige --hybride) : SOM sémantique sur les profils du "
        "vocabulaire, cartes de chaleur par document — 4e canal de `search --fusion`. "
        "Mesuré : +2.84 pts R@10 et +3.65 MRR sur Alloprof complet au-dessus du trio. "
        "Coût AU BUILD : la SOM travaille sur tout le vocabulaire (~4 Go de pic RAM et "
        "~20 min à 72k tokens) — jamais un défaut",
    )
    p_build.add_argument(
        "--grammatical",
        action="store_true",
        help="canal GRAMMATICAL déterministe (opt-in) : rôles à règles fermées "
        "(négation à portée, amont/aval ordonné, agent/patient) analysés sur le texte "
        "brut et liés aux signatures (bind du canal relations). Mesuré : 25/34 paires "
        "à mots identiques/sens opposé sont invisibles au moteur nu (cos 1.0000) ; "
        "33/34 séparées avec le canal ; analyseur à 2.1 %% d'erreur avec abstention "
        "propre. Canal SÉPARÉ : sans le drapeau de recherche, impact nul",
    )
    p_build.add_argument(
        "--no-path-tokens",
        action="store_true",
        help="désactive l'injection des tokens du chemin relatif en tête du document "
        "(comportement v1.4 ; par défaut, l'injection est active)",
    )
    p_build.add_argument(
        "--ocr",
        action="store_true",
        help="active le crochet OCR — pour les convertibles dont la conversion rend "
        "< 200 caractères, ET pour les photos/images (.jpg/.jpeg/.png/.tiff), qui ne "
        "sont indexées que si ce drapeau est actif (sinon ignorées+comptées) ; "
        "nécessite un moteur OCR installé (rapidocr ou rapidocr_onnxruntime)",
    )
    p_build.add_argument(
        "--relations",
        action="store_true",
        help="active le canal de relations (v2.0) : relations tirées gratuitement de "
        "l'arborescence du corpus (dossier/année/mois), écrites dans relations.msrel, "
        "utilisables par `mosaic related` (défaut désactivé, aucun changement sinon)",
    )
    p_build.add_argument(
        "--profil",
        default=None,
        help="profil d'index (JSON) : paramétrage déclaratif métier — rôles d'arborescence, "
        "types de documents custom, critère de références. Persisté dans l'index (build/add/"
        "search relisent le même). Voir `mosaic profil --explique` et `--suggere`.",
    )
    p_build.add_argument(
        "--type-doc",
        action="store_true",
        help="ajoute la FACETTE type de document à l'encodage (tableur, pdf scanné/numérique, "
        "photo, document rédigé, présentation, page web) — permet de chercher par nature de "
        "document, complémentaire du sens (défaut désactivé)",
    )

    p_embed = ajouter(
        "embed-prepare", "prépare une table d'embeddings .msee depuis un .vec.gz"
    )
    p_embed.add_argument("vec_gz", help="fichier fastText .vec.gz source")
    p_embed.add_argument("-o", "--output", required=True, help="table .msee produite")
    p_embed.add_argument(
        "--keep", type=int, default=200_000, help="nombre de mots gardés, entier >= 1"
    )
    p_embed.add_argument(
        "--abtt",
        type=int,
        default=0,
        help="applique all-but-the-top À LA PRÉPARATION (table déjà nettoyée à l'écriture, "
        "cf. mosaic build --abtt) — entier dans [0, 255]",
    )

    p_add = ajouter("add", "ajoute un document à un index existant")
    p_add.add_argument("file", help="document à ajouter")
    p_add.add_argument("index", help="dossier de l'index")
    _ajouter_cache_ingestion(p_add)

    p_search = ajouter("search", "recherche sémantique dans un index")
    p_search.add_argument(
        "query", nargs="?", default=None, help="requête (absente en mode --batch)"
    )
    p_search.add_argument("index", help="dossier de l'index")
    _ajouter_top(p_search)
    p_search.add_argument(
        "--fusion",
        action="store_true",
        help="fusion RRF multi-canaux (grille + BM25 + embeddings, + atlas si présent) — "
        "nécessite un index construit avec --hybride ; exclusif avec --rerank",
    )
    p_search.add_argument(
        "--rerank",
        action="store_true",
        help="repêcheur : reclasse le top --rerank-depth par mélange avec model2vec "
        "(nécessite un index construit avec --rerank-vectors)",
    )
    _ajouter_rerank_params(p_search)
    p_search.add_argument(
        "--batch",
        action="store_true",
        help="lit les requêtes depuis stdin (une par ligne, UTF-8), affiche un tableau JSON "
        "par ligne — un seul process, un seul chargement pour N requêtes (agents "
        "faisant plusieurs recherches : amortit le coût de démarrage). Pas de `query` "
        "positionnelle dans ce mode. Une requête en échec émet {'error', 'requete'} sur "
        "SA ligne et le flux continue (rc final 1 si au moins un échec).",
    )
    p_search.add_argument(
        "--grammatical",
        action="store_true",
        help="active le canal grammatical à la recherche (nécessite un index construit "
        "avec --grammatical) : score = grille + 0.5·structural — sépare les clauses à "
        "mots identiques et sens opposé (amont/aval, négation à portée). Exclusif (v1) "
        "avec --fusion/--rerank/--type/--recence/--connecteurs",
    )
    p_search.add_argument(
        "--nettoyer-requete",
        action="store_true",
        help="retire le bruit conversationnel de la requête AVANT de chercher "
        "(salutations, demandes d'aide, aveux d'incompréhension — classes fermées du "
        "français, aucun modèle). Mesuré sur Alloprof, 2 316 requêtes réelles : "
        "+6.50 pts de rappel et +4.82 de MRR, SANS reconstruire l'index. Sans effet "
        "sur une requête déjà propre (1 %% des requêtes SciFact touchées)",
    )
    p_search.add_argument(
        "--connecteurs",
        action="store_true",
        help="algèbre de connecteurs : « et/ou » renforcent, « sans/pas/ni », « mais pas » "
        "FONT DESCENDRE le score. score = cos(doc, voulu) - λ·cos(doc, exclu). Incompatible "
        "avec --rerank/--batch. Résultats enrichis (positif/négatif par doc).",
    )
    p_search.add_argument(
        "--conn-lambda",
        type=float,
        default=None,
        help="λ de l'algèbre de connecteurs (poids de la soustraction), dans [0, 1] — "
        "défaut effectif 0.7 ; refusé sans --connecteurs",
    )
    p_search.add_argument(
        "--type",
        dest="type_filtre",
        default=None,
        help="facette : ne garde que les documents de ce type exact (tableur, pdf numérique, "
        "pdf scanné, photo, document rédigé, présentation, page web, note texte). Nécessite "
        "un index avec facettes.json (reconstruit depuis cette version).",
    )
    p_search.add_argument(
        "--recence",
        type=float,
        default=0.0,
        help="facette : poids de la fraîcheur dans [0,1] (0 = ordre sémantique pur). "
        "Fusion de rangs sens/date — les documents récents remontent, utile sur les dossiers "
        "qui évoluent (la date est lue dans le nom de fichier AAAA-MM-JJ).",
    )

    p_like = ajouter("like", "recherche par l'exemple : un document comme requête")
    p_like.add_argument(
        "docs",
        nargs="+",
        help="1+ id(s) déjà indexé(s) ou chemin(s) de fichier externe, suivi(s) de l'index "
        "(dernier positionnel) ; 2+ documents = mélange (moyenne normalisée)",
    )
    _ajouter_top(p_like)
    p_like.add_argument(
        "--rerank",
        action="store_true",
        help="repêcheur : id interne réutilise son empreinte rerank stockée (aucun modèle "
        "requis) ; fichier externe encodé via model2vec (nécessite un index construit "
        "avec --rerank-vectors)",
    )
    _ajouter_rerank_params(p_like)
    _ajouter_cache_ingestion(p_like)

    p_related = ajouter("related", "documents liés à une entité (canal de relations)")
    p_related.add_argument("entite", help="entité (dossier/année/mois normalisés)")
    p_related.add_argument("index", help="dossier de l'index")
    p_related.add_argument(
        "--role",
        choices=("dossier", "annee", "mois"),
        default=None,
        help="restreindre au rôle",
    )
    _ajouter_top(p_related)

    p_chemin = ajouter(
        "chemin", "traversée multi-sauts : les documents frères via les entités"
    )
    p_chemin.add_argument("doc_id", help="document de départ (id indexé)")
    p_chemin.add_argument("index", help="dossier de l'index")
    p_chemin.add_argument(
        "--role",
        choices=("dossier", "annee", "mois"),
        default=None,
        help="restreindre la traversée à un rôle (défaut : toutes les entités du document)",
    )
    _ajouter_top(p_chemin)

    p_render = ajouter("render", "rend la grille d'un document en image PNG")
    p_render.add_argument("doc_id", help="document à rendre")
    p_render.add_argument("index", help="dossier de l'index")
    p_render.add_argument("-o", "--output", required=True, help="PNG produit")

    p_stats = ajouter("stats", "statistiques et configuration d'un index")
    p_stats.add_argument("index", help="dossier de l'index")

    p_compter = ajouter(
        "compter",
        "combien de documents ? (comptage exact, pas une recherche) — rend aussi "
        "la couverture : le nombre de documents SANS date, exclus des comptes datés",
    )
    p_compter.add_argument("index", help="dossier de l'index")
    p_compter.add_argument(
        "--chemin", default="", help="fragment de chemin (cherché littéralement)"
    )
    p_compter.add_argument(
        "--type", dest="type_doc", default="", help="type de document"
    )
    p_compter.add_argument(
        "--date", default="", help="préfixe de date : 2026, 2026-06, 2026-06-22"
    )
    p_compter.add_argument(
        "--par-mois", action="store_true", help="détailler la répartition par mois"
    )

    p_recents = ajouter(
        "recents",
        "les k documents les plus RÉCENTS (ordre exact par date, pas par pertinence) ; "
        "les documents sans date sont exclus, jamais classés au hasard",
    )
    p_recents.add_argument("index", help="dossier de l'index")
    p_recents.add_argument("-k", type=int, default=5, help="combien en rendre")
    p_recents.add_argument("--chemin", default="", help="fragment de chemin")
    p_recents.add_argument(
        "--type", dest="type_doc", default="", help="type de document"
    )

    p_refs = ajouter(
        "refs",
        "quels documents portent cette référence ? (jointure exacte à travers "
        "PLUSIEURS index : une réf relie un devis, un BL et une facture)",
    )
    p_refs.add_argument("ref", help="référence exacte (n° de BL, code article…)")
    p_refs.add_argument("index", nargs="+", help="un ou plusieurs dossiers d'index")

    p_explain = ajouter(
        "explain", "pourquoi ce document ? (contributions par token, démélange)"
    )
    p_explain.add_argument("doc_id", help="document à expliquer")
    p_explain.add_argument("index", help="dossier de l'index")
    _ajouter_top(p_explain, defaut=20)
    p_explain.add_argument(
        "--query", default=None, help="explique le MATCH avec cette requête"
    )

    p_carte = ajouter(
        "carte",
        "carte d'identité sémantique d'un dossier (écrit <dossier>/_MOSAIC/ : "
        "index jetable + cartes.html)",
    )
    p_carte.add_argument(
        "dossier",
        help="dossier à cartographier — l'artefact est écrit DANS ce dossier, "
        "sous _MOSAIC/ (seule commande qui écrit à côté des documents)",
    )
    p_carte.add_argument(
        "--top-concepts", type=int, default=8, help="concepts par document"
    )
    _ajouter_cache_ingestion(p_carte)
    p_carte.add_argument(
        "--profile-weighting",
        default=PROFILE_WEIGHTING_DEFAULT,
        help="brut ou ppmi",
    )
    p_carte.add_argument(
        "--smoothing-rank",
        type=int,
        default=SMOOTHING_RANK_DEFAULT,
        help="rang du lissage SVD, entier >= 0",
    )
    p_carte.add_argument(
        "--grid",
        choices=sorted(_GRIDS),
        default="64x64",
        help="géométrie de la grille de l'index jetable",
    )
    p_carte.add_argument(
        "--embeddings",
        default=None,
        help="table d'embeddings .msee (canal γ) pour l'index jetable — mêmes "
        "concepts, similarités plus fines",
    )
    p_carte.add_argument(
        "--ocr",
        action="store_true",
        help="active le crochet OCR de l'ingestion (photos et PDF scannés du dossier "
        "cartographié — sans lui ils sont ignorés, donc SANS carte)",
    )
    p_carte.add_argument(
        "--type-doc",
        action="store_true",
        help="ajoute la facette type de document à l'encodage de l'index jetable",
    )
    p_carte.add_argument(
        "--profil",
        default=None,
        help="profil d'index (JSON) appliqué à l'index jetable — types custom, "
        "critère de références, listes grammaticales",
    )

    p_proches = ajouter("proches", "voisins d'un mot (table .msee ou index de corpus)")
    p_proches.add_argument("mot", help="mot dont on veut les voisins")
    p_proches.add_argument(
        "table",
        help="table .msee (voisins génériques, ex. data_externes/potion_fr.msee) OU un "
        "dossier d'index (voisins SÉMANTIQUES appris de ce corpus, ex. ./index_devis)",
    )
    _ajouter_top(p_proches)
    p_proches.add_argument(
        "--abtt",
        type=int,
        default=None,
        help="all-but-the-top au chargement d'une table .msee (défaut effectif 0 = table "
        "déjà nettoyée) — refusé sur un dossier d'index (sans effet là-bas)",
    )
    p_proches.add_argument(
        "--dico",
        action="store_true",
        help="voisinage strictement LEXICAL : ne garde que les vrais mots français (vocab de la "
        "table potion embarquée), écarte le boilerplate de contenu (collocations, codes, dates) "
        "au prix des codes-métier hors dictionnaire. Nécessite un index construit avec "
        "--embeddings ; sans effet (donc refusé) ailleurs.",
    )

    p_actuel = ajouter(
        "actuel", "recherche récence-aware : versions groupées, la canonique en tête"
    )
    p_actuel.add_argument("query", help="requête")
    p_actuel.add_argument("index", help="dossier de l'index")
    _ajouter_top(p_actuel)
    p_actuel.add_argument(
        "--seuil-version",
        type=float,
        default=SEUIL_VERSION_DEFAUT,
        help="cosinus dans [0, 1] au-dessus duquel deux documents sont des versions du "
        "même aspect (date lue dans le nom de fichier AAAA-MM-JJ)",
    )
    p_actuel.add_argument(
        "--rerank",
        action="store_true",
        help="repêcheur sur la recherche sous-jacente (mêmes exigences que `search --rerank`)",
    )
    _ajouter_rerank_params(p_actuel)
    p_actuel.add_argument(
        "--type",
        dest="type_filtre",
        default=None,
        help="facette : même filtre de type exact que `search --type`",
    )
    p_actuel.add_argument(
        "--recence",
        type=float,
        default=0.0,
        help="facette : même fusion de fraîcheur que `search --recence`",
    )

    p_diff = ajouter(
        "diff",
        "diff SÉMANTIQUE entre deux états d'un corpus — ce qui a changé de sens, "
        "pas seulement de contenu",
    )
    p_diff.add_argument(
        "avant", help="état t1 : dossier de CORPUS (build éphémère) ou d'INDEX"
    )
    p_diff.add_argument(
        "apres",
        help="état t2 : même nature que `avant` (corpus avec corpus, index avec index)",
    )
    _ajouter_top(p_diff, defaut=20)

    p_meta = ajouter(
        "meta", "méta-recherche : plusieurs index fusionnés par rangs (RRF)"
    )
    p_meta.add_argument("query", help="requête")
    p_meta.add_argument(
        "indexes",
        nargs="+",
        help="2+ dossiers d'index à interroger ensemble (ex. ./index_devis "
        "./index_courriels). Le nom affiché = nom du dossier.",
    )
    _ajouter_top(p_meta)
    p_meta.add_argument(
        "--profondeur",
        type=int,
        default=20,
        help="nombre de résultats tirés de CHAQUE index avant fusion",
    )
    p_meta.add_argument(
        "--rerank",
        action="store_true",
        help="applique le repêcheur sur chaque index qui le peut (rerank.msrv présent) — "
        "un index sans repêcheur DÉGRADE proprement (listé dans `sans_rerank`) au lieu "
        "de faire échouer la fusion entière",
    )
    _ajouter_rerank_params(p_meta)
    p_meta.add_argument(
        "--fusion",
        choices=("rrf", "znorm"),
        default="rrf",
        help="méthode de fusion. « rrf » (défaut) : fusion par RANGS, la monnaie juste "
        "entre corpus HÉTÉROGÈNES dont les scores ne sont pas comparables. « znorm » : "
        "scores z-normalisés par source — pour des sources HOMOGÈNES (un même corpus "
        "découpé en tranches). Mesuré sur Alloprof en 4 tranches équilibrées : znorm "
        "0.3832 R@10 / 0.2229 MRR contre 0.3216 / 0.2164 pour l'index unique — il bat "
        "l'index entier sur les DEUX métriques, là où le RRF gagne le rappel en payant "
        "la précision de tête. À ne pas employer sur des corpus de natures différentes",
    )
    p_meta.add_argument(
        "--resume",
        action="store_true",
        help="ajoute un diagnostic de rappel par index (candidats, meilleur score local)",
    )

    p_profil = ajouter("profil", "profil d'un index, ou suggestion depuis un corpus")
    p_profil.add_argument(
        "cible",
        help="dossier d'INDEX (voir son profil) ou de CORPUS (avec --suggere : proposer un "
        "profil calibré sur cet environnement)",
    )
    p_profil.add_argument(
        "--explique",
        action="store_true",
        help="mode humain : raconte en français ce que le profil fait faire au moteur "
        "(sortie {'explication': ...})",
    )
    p_profil.add_argument(
        "--suggere",
        action="store_true",
        help="mode agent : scanne le corpus cible et propose un profil candidat (JSON) — "
        "extensions à mapper, motifs d'arborescence observés — à ajuster puis passer à "
        "`mosaic build --profil`",
    )
    p_profil.add_argument(
        "--langue",
        choices=("fr", "en"),
        default=None,
        help="langue du mode --explique (défaut effectif fr ; refusé sans --explique)",
    )

    p_calib = ajouter(
        "calibrer", "choisit les poids d'encodage par la mesure sur TES requêtes-vérité"
    )
    p_calib.add_argument("corpus", help="dossier de documents à calibrer")
    p_calib.add_argument(
        "--requetes",
        default=None,
        help='jeu de requêtes-vérité (JSONL : {"query": ..., "relevant": [doc_ids]}) — '
        "c'est LUI qui choisit les poids, jamais un curseur manuel ; exclusif avec "
        "--verite-auto",
    )
    p_calib.add_argument(
        "--verite-auto",
        action="store_true",
        help="génère la vérité DÉTERMINISTIQUEMENT (held-out : moitié indexée, requête = "
        "termes distinctifs de l'autre moitié) — zéro LLM, zéro humain. Plancher utile "
        "sans requêtes rédigées ; celles-ci restent l'étalon-or. Exclusif avec --requetes.",
    )
    p_calib.add_argument(
        "--embeddings", default=None, help="table .msee incluse dans la calibration"
    )
    p_calib.add_argument(
        "--abtt", type=int, default=0, help="all-but-the-top, entier dans [0, 255]"
    )
    p_calib.add_argument(
        "--no-path-tokens",
        action="store_true",
        help="calibre SANS les tokens de chemin — à utiliser si l'index visé sera "
        "construit avec ce même drapeau (noms de fichiers opaques) : calibrer sur un "
        "espace différent de celui du build choisirait des poids pour un autre monde",
    )
    p_calib.add_argument(
        "--explique",
        action="store_true",
        help="verdict en clair (mode humain, sortie {'explication': ...}) au lieu du "
        "rapport JSON",
    )
    p_calib.add_argument(
        "--langue",
        choices=("fr", "en"),
        default=None,
        help="langue du mode --explique (défaut effectif fr ; refusé sans --explique)",
    )

    p_croy = ajouter(
        "croyance", "mémoire de croyance : asserter, lire, historiser, calibrer"
    )
    p_croy.add_argument(
        "action",
        choices=["assert", "courant", "historique", "calibrer"],
        help="assert écrit (et crée le store au besoin) ; courant/historique/calibrer "
        "exigent un store existant",
    )
    p_croy.add_argument("store", help="fichier de mémoire de croyance (.jsonl)")
    p_croy.add_argument("--entite", default=None, help="entité du fait")
    p_croy.add_argument("--attribut", default=None, help="attribut du fait")
    p_croy.add_argument(
        "--valeur", default=None, help="valeur du fait (assert uniquement)"
    )
    p_croy.add_argument(
        "--t",
        type=float,
        default=None,
        help="ordre temporel (assert uniquement ; défaut : à la suite)",
    )
    p_croy.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="calibrer uniquement : taux d'erreur cible (le seuil « contesté » est CALIBRÉ "
        "sur les faits du store — prédiction conforme, sans donnée externe ; défaut "
        "effectif 0.05)",
    )

    return parser


# ---------------------------------------------------------------------------------
# Handlers — un par sous-commande, gardes d'arguments AVANT toute I/O.


def _cmd_build(args) -> int:
    corpus = Path(args.corpus)
    if not corpus.is_dir():
        raise ValueError(f"corpus introuvable : {corpus}")
    if args.lexicon == "none":
        lex: dict[str, str] | None = {}
    elif args.lexicon:
        lex = load_lexicon(Path(args.lexicon))
    else:
        lex = None
    if args.lexicon_extra:
        extra_path = _resolve_lexicon_extra(args.lexicon_extra)
        base = lex if lex is not None else load_lexicon()
        lex = _merge_extra_lexicon(base, extra_path)
    embeddings_path = Path(args.embeddings) if args.embeddings else None
    weights = _parse_weights(args.weights) if args.weights else None
    profile_weighting = _parse_profile_weighting(args.profile_weighting)
    smoothing_rank = _parse_int_nonnegative(args.smoothing_rank, "smoothing-rank")
    abtt = _parse_abtt(args.abtt)
    doc_weight = (
        _parse_doc_weight(args.doc_weight) if args.doc_weight is not None else 0.0
    )
    ingest_cache_dir = _ingest_cache_root() if args.cache_ingestion else None
    output = Path(args.output)
    output_preexistant = output.exists()
    idx = Index.build(
        corpus,
        output,
        grid=_GRIDS[args.grid],
        lexicon=lex,
        embeddings_path=embeddings_path,
        weights=weights,
        profile_weighting=profile_weighting,
        smoothing_rank=smoothing_rank,
        abtt=abtt,
        ingest_cache_dir=ingest_cache_dir,
        doc_weight=doc_weight,
        rerank_vectors=args.rerank_vectors,
        hybride=args.hybride,
        atlas=args.atlas,
        grilles_typees=args.grilles_typees,
        grammatical=args.grammatical,
        index_paths=not args.no_path_tokens,
        ocr=args.ocr,
        relations=args.relations,
        type_doc=args.type_doc,
        profil=(profil_module.charger(Path(args.profil)) if args.profil else None),
    )
    if not idx.ids:
        # Le pire échec silencieux du produit : un index VIDE livré rc 0 rendrait []
        # à chaque recherche sans jamais dire pourquoi. Refus loud, et l'artefact
        # n'est pas conservé (sauf si le dossier préexistait à l'appel).
        if not output_preexistant:
            shutil.rmtree(output, ignore_errors=True)
        raise ValueError(
            f"aucun document indexable trouvé dans {corpus} — index non conservé"
        )
    _out(idx.stats())
    return 0


def _cmd_embed_prepare(args) -> int:
    _parse_int_positive(args.keep, "keep")
    abtt = _parse_abtt(args.abtt)
    lex_pivots = load_lexicon()
    extra_words: set[str] = set()
    for pivot in lex_pivots.values():
        extra_words.update(pivot.split("_"))
    stats = prepare(
        Path(args.vec_gz),
        Path(args.output),
        keep=args.keep,
        extra_words=extra_words,
        abtt=abtt,
    )
    _out(stats)
    return 0


def _cmd_add(args) -> int:
    ingest_cache_dir = _ingest_cache_root() if args.cache_ingestion else None
    idx = _ouvrir_index(Path(args.index))
    idx.add(Path(args.file), ingest_cache_dir=ingest_cache_dir)
    _out(idx.stats())
    return 0


def _cmd_search(args) -> int:
    _parse_int_positive(args.top, "top")
    rerank_lambda = _parse_rerank_lambda(args.rerank_lambda)
    rerank_depth = _parse_rerank_depth(args.rerank_depth)
    if args.conn_lambda is not None and not args.connecteurs:
        raise ValueError("--conn-lambda est refusé sans --connecteurs (sans effet)")
    conn_lambda = (
        _parse_cos_01(args.conn_lambda, "conn-lambda")
        if args.conn_lambda is not None
        else 0.7
    )
    if args.fusion and args.rerank:
        raise ValueError(
            "fusion et rerank sont exclusifs : la fusion intègre déjà le canal "
            "embeddings comme canal plein"
        )
    if args.connecteurs and (args.batch or args.rerank):
        raise ValueError("--connecteurs est incompatible avec --batch et --rerank")
    if args.fusion and args.connecteurs:
        raise ValueError("--fusion est incompatible avec --connecteurs")
    if args.connecteurs and (args.type_filtre or args.recence):
        raise ValueError("--connecteurs est incompatible avec --type/--recence")
    if args.grammatical and args.connecteurs:
        raise ValueError("--grammatical est incompatible avec --connecteurs")
    if args.batch and args.query is not None:
        raise ValueError(
            "--batch : pas de `query` positionnelle, les requêtes viennent de "
            "stdin (une par ligne)"
        )
    if not args.batch and args.query is None:
        raise ValueError("query requise (sauf en mode --batch)")

    # verify_embeddings=False : chemin recherche uniquement (jamais build/add) —
    # cf. Index.open() et Embeddings.load(verify=...).
    idx = _ouvrir_index(Path(args.index), verify_embeddings=False)
    # Options construites UNE fois pour les deux branches (batch/simple) : une option
    # ajoutée dans l'une ne peut plus être silencieusement absente de l'autre.
    opts: dict[str, Any] = dict(
        k=args.top,
        rerank=args.rerank,
        rerank_lambda=rerank_lambda,
        rerank_depth=rerank_depth,
        type_filtre=args.type_filtre,
        recence=args.recence,
        fusion=args.fusion,
        grammatical=args.grammatical,
        nettoyer_requete=args.nettoyer_requete,
    )
    if args.batch:
        echecs = 0
        for line in sys.stdin:
            query = line.rstrip("\r\n")
            if not query:
                continue
            # Résilience par LIGNE : une requête en échec n'avale pas le reste du
            # flux, et l'erreur est corrélée à SA requête (audit finding 25).
            try:
                _out(idx.search(query, **opts))
            except (OSError, ValueError) as exc:
                _out({"error": str(exc), "requete": query})
                echecs += 1
        return 1 if echecs else 0
    if args.connecteurs:
        _out(idx.search_connecteurs(args.query, k=args.top, lam=conn_lambda))
        return 0
    _out(idx.search(args.query, **opts))
    return 0


def _cmd_like(args) -> int:
    if len(args.docs) < 2:
        raise ValueError(
            "mosaic like requiert au moins un document (id ou chemin) et un index "
            "(dernier positionnel)"
        )
    _parse_int_positive(args.top, "top")
    rerank_lambda = _parse_rerank_lambda(args.rerank_lambda)
    rerank_depth = _parse_rerank_depth(args.rerank_depth)
    *doc_specs, index_path = args.docs
    ingest_cache_dir = _ingest_cache_root() if args.cache_ingestion else None
    idx = _ouvrir_index(Path(index_path), verify_embeddings=False)
    _out(
        idx.search_like(
            doc_specs,
            k=args.top,
            rerank=args.rerank,
            rerank_lambda=rerank_lambda,
            rerank_depth=rerank_depth,
            ingest_cache_dir=ingest_cache_dir,
        )
    )
    return 0


def _cmd_chemin(args) -> int:
    _parse_int_positive(args.top, "top")
    idx = _ouvrir_index(Path(args.index), verify_embeddings=False)
    _out(idx.chemin(args.doc_id, k=args.top, role=args.role))
    return 0


def _cmd_related(args) -> int:
    _parse_int_positive(args.top, "top")
    idx = _ouvrir_index(Path(args.index), verify_embeddings=False)
    _out(idx.related(args.entite, k=args.top, role=args.role))
    return 0


def _cmd_render(args) -> int:
    # verify_embeddings=False : commande de LECTURE (l'audit a montré que render/
    # stats/explain payaient la re-vérification sha de la table sans jamais écrire).
    render_doc(
        _ouvrir_index(Path(args.index), verify_embeddings=False),
        args.doc_id,
        Path(args.output),
    )
    _out({"ok": True, "out": args.output})
    return 0


def _cmd_stats(args) -> int:
    _out(_ouvrir_index(Path(args.index), verify_embeddings=False).stats())
    return 0


def _magasin(chemins: list[str]):
    """Dérive un magasin structurel des index donnés. Le nom d'un index est celui
    de son dossier — c'est ce que l'appelant reconnaîtra dans la réponse."""
    from mosaic.structurel import Magasin

    m = Magasin()
    noms = []
    for c in chemins:
        nom = Path(c).name
        m.charger(nom, Path(c))
        noms.append(nom)
    return m, noms


def _cmd_compter(args) -> int:
    m, (nom,) = _magasin([args.index])
    sortie = {
        "total": m.compter(
            nom,
            chemin_contient=args.chemin,
            type_doc=args.type_doc,
            date_prefixe=args.date,
        ),
        # La couverture accompagne TOUJOURS le compte : un total daté qui tait le
        # nombre de documents sans date laisse croire qu'il couvre tout.
        "sans_date": m.sans_date(nom),
    }
    if args.par_mois:
        sortie["par_mois"] = m.repartition_par_mois(
            nom, chemin_contient=args.chemin, type_doc=args.type_doc
        )
    _out(sortie)
    return 0


def _cmd_recents(args) -> int:
    _parse_int_positive(args.k, "k")
    m, (nom,) = _magasin([args.index])
    _out(
        [
            {"id": doc, "date": date}
            for doc, date in m.plus_recents(
                nom, k=args.k, chemin_contient=args.chemin, type_doc=args.type_doc
            )
        ]
    )
    return 0


def _cmd_refs(args) -> int:
    m, _ = _magasin(args.index)
    _out(
        [
            {"index": idx, "id": doc, "date": date}
            for idx, doc, date in m.documents_portant_ref(args.ref)
        ]
    )
    return 0


def _cmd_explain(args) -> int:
    _parse_int_positive(args.top, "top")
    idx = _ouvrir_index(Path(args.index), verify_embeddings=False)
    if args.query:
        _out(idx.explain_match(args.query, args.doc_id, k=args.top))
    else:
        _out(idx.explain(args.doc_id, k=args.top))
    return 0


def _cmd_carte(args) -> int:
    _parse_int_positive(args.top_concepts, "top-concepts")
    dossier = Path(args.dossier)
    ingest_cache_dir = _ingest_cache_root() if args.cache_ingestion else None
    profile_weighting = _parse_profile_weighting(args.profile_weighting)
    smoothing_rank = _parse_int_nonnegative(args.smoothing_rank, "smoothing-rank")
    out, docs = carte.generer(
        dossier,
        k_concepts=args.top_concepts,
        grid=_GRIDS[args.grid],
        ingest_cache_dir=ingest_cache_dir,
        profile_weighting=profile_weighting,
        smoothing_rank=smoothing_rank,
        embeddings_path=Path(args.embeddings) if args.embeddings else None,
        ocr=args.ocr,
        type_doc=args.type_doc,
        profil=(profil_module.charger(Path(args.profil)) if args.profil else None),
    )
    _out({"ok": True, "out": str(out), "docs": docs})
    return 0


def _cmd_proches(args) -> int:
    _parse_int_positive(args.top, "top")
    source = Path(args.table)
    if source.is_dir():  # index de corpus -> voisins sémantiques métier
        if args.abtt is not None:
            raise ValueError(
                "--abtt est refusé sur un dossier d'index (sans effet — il ne "
                "s'applique qu'au chargement d'une table .msee)"
            )
        idx = _ouvrir_index(source, verify_embeddings=False)
        if args.dico and idx.embeddings is None:
            raise ValueError(
                "--dico nécessite un index construit avec --embeddings (le filtre "
                "lexical lit le vocabulaire de la table) — sans elle il serait "
                "silencieusement sans effet"
            )
        voisins = idx.proches(args.mot, k=args.top, dico=args.dico)
        if voisins is None:
            raise ValueError(
                f"mot absent du vocabulaire (ou sans cooccurrence) : {args.mot}"
            )
    else:  # table .msee -> voisins génériques
        abtt = _parse_abtt(args.abtt) if args.abtt is not None else 0
        emb = Embeddings.load(source, abtt=abtt)
        voisins = emb.proches(args.mot, k=args.top)
        if voisins is None:
            raise ValueError(f"mot absent de la table : {args.mot}")
    _out(
        [
            {"mot": mot, "score": round(float(score), 6), "rang": rang}
            for rang, (mot, score) in enumerate(voisins, start=1)
        ]
    )
    return 0


def _cmd_actuel(args) -> int:
    _parse_int_positive(args.top, "top")
    seuil = _parse_cos_01(args.seuil_version, "seuil-version")
    rerank_lambda = _parse_rerank_lambda(args.rerank_lambda)
    rerank_depth = _parse_rerank_depth(args.rerank_depth)
    idx = _ouvrir_index(Path(args.index), verify_embeddings=False)
    resultats = versions_actuelles(
        idx,
        args.query,
        k=args.top,
        seuil_version=seuil,
        rerank=args.rerank,
        rerank_lambda=rerank_lambda,
        rerank_depth=rerank_depth,
        type_filtre=args.type_filtre,
        recence=args.recence,
    )
    # Socle de sortie : tout résultat classé porte `id` — `canonique` reste en alias
    # (compat) mais les agents peuvent parser uniformément (audit finding 27).
    for r in resultats:
        if "canonique" in r:
            r.setdefault("id", r["canonique"])
    _out(resultats)
    return 0


def _cmd_diff(args) -> int:
    _parse_int_positive(args.top, "top")
    avant, apres = Path(args.avant), Path(args.apres)
    est_index_a = (avant / "docs.msei").is_file()
    est_index_b = (apres / "docs.msei").is_file()
    if est_index_a != est_index_b:
        raise ValueError(
            "diff : comparer un corpus avec un index n'a pas de sens — donner deux "
            "dossiers de corpus (build éphémère) ou deux dossiers d'index"
        )
    if est_index_a:
        ia = _ouvrir_index(avant, verify_embeddings=False)
        ib = _ouvrir_index(apres, verify_embeddings=False)
        _out(diff_indexes(ia, ib, top=args.top))
        return 0
    if not avant.is_dir() or not apres.is_dir():
        raise ValueError(
            f"dossier introuvable : {avant if not avant.is_dir() else apres}"
        )
    _out(diff_corpus(avant, apres, top=args.top))
    return 0


def _cmd_meta(args) -> int:
    _parse_int_positive(args.top, "top")
    _parse_int_positive(args.profondeur, "profondeur")
    rerank_lambda = _parse_rerank_lambda(args.rerank_lambda)
    rerank_depth = _parse_rerank_depth(args.rerank_depth)
    if len(args.indexes) < 2:
        raise ValueError("meta : donner au moins 2 index à fusionner")
    listes = []
    sans_rerank: list[str] = []
    for chemin_index in args.indexes:
        p = Path(chemin_index)
        idx = _ouvrir_index(p, verify_embeddings=False)
        if args.rerank and idx.rerank_vecs is None:
            # Dégradation PAR INDEX : un index sans rerank.msrv ne fait plus échouer
            # la fusion entière — il est fusionné sans repêcheur, et listé.
            sans_rerank.append(p.name)
            res = idx.search(args.query, k=args.profondeur)
        else:
            res = idx.search(
                args.query,
                k=args.profondeur,
                rerank=args.rerank,
                rerank_lambda=rerank_lambda,
                rerank_depth=rerank_depth,
            )
        listes.append((p.name, res))
    fusion = (
        znorm_fuse(listes, k=args.top)
        if args.fusion == "znorm"
        else rrf_fuse(listes, k=args.top)
    )
    # Racine TOUJOURS objet (audit finding 26) : la forme ne dépend plus des drapeaux.
    sortie: dict = {"resultats": fusion}
    if args.resume:
        sortie["resume"] = resume_par_index(listes)
    if sans_rerank:
        sortie["sans_rerank"] = sans_rerank
    _out(sortie)
    return 0


def _cmd_calibrer(args) -> int:
    from mosaic.calibration import calibrer, expliquer_calibration

    if args.langue is not None and not args.explique:
        raise ValueError("--langue est refusé sans --explique (sans effet)")
    langue = args.langue or "fr"
    if args.verite_auto and args.requetes:
        raise ValueError(
            "--requetes et --verite-auto sont exclusifs : la calibration doit savoir "
            "quelle vérité fait foi (l'étalon-or rédigé OU le held-out déterministe)"
        )
    if args.verite_auto:
        requetes = None
    elif args.requetes:
        requetes = [
            json.loads(ligne)
            for ligne in Path(args.requetes).read_text(encoding="utf-8").splitlines()
            if ligne.strip()
        ]
    else:
        raise ValueError("donner --requetes <verite.jsonl> ou --verite-auto")
    rapport = calibrer(
        Path(args.corpus),
        requetes,
        embeddings_path=Path(args.embeddings) if args.embeddings else None,
        abtt=_parse_abtt(args.abtt),
        verite_auto=args.verite_auto,
        index_paths=not args.no_path_tokens,
    )
    rapport["verite"] = "auto" if args.verite_auto else "fichier"
    if args.explique:
        _out({"explication": expliquer_calibration(rapport, langue=langue)})
    else:
        _out(rapport)
    return 0


def _cmd_profil(args) -> int:
    if args.langue is not None and not args.explique:
        raise ValueError("--langue est refusé sans --explique (sans effet)")
    langue = args.langue or "fr"
    cible = Path(args.cible)
    if args.suggere:
        if not cible.is_dir():
            raise ValueError(f"corpus introuvable : {cible}")
        suggestion = profil_module.suggerer(cible)
        if args.explique:
            _out({"explication": profil_module.expliquer(suggestion, langue=langue)})
        else:
            _out(suggestion)
        return 0
    # cible = un index : montrer son profil actif (meta), None = défauts
    idx = _ouvrir_index(cible, verify_embeddings=False)
    if args.explique:
        _out({"explication": profil_module.expliquer(idx.profil, langue=langue)})
    else:
        _out(idx.profil or {"profil": None, "note": "défauts historiques"})
    return 0


def _cmd_croyance(args) -> int:
    # Refus des drapeaux hors-sujet selon l'action (audit finding 11) : `courant
    # --valeur X` faisait croire à une écriture qui n'avait pas lieu.
    if args.action != "assert" and (args.valeur is not None or args.t is not None):
        raise ValueError(f"--valeur/--t sont refusés pour `{args.action}` (sans effet)")
    if args.action != "calibrer" and args.alpha is not None:
        raise ValueError(f"--alpha est refusé pour `{args.action}` (sans effet)")
    if args.action == "calibrer" and (args.entite or args.attribut):
        raise ValueError(
            "--entite/--attribut sont refusés pour `calibrer` (sans effet)"
        )
    store = Path(args.store)
    if args.action != "assert" and not store.exists():
        # Un chemin fauté fabriquait une mémoire vide indiscernable d'un historique
        # vide (audit finding 23) ; seule l'écriture crée le store.
        raise ValueError(f"store de croyance introuvable : {store}")
    mem = MemoireCroyance.charger(store) if store.exists() else MemoireCroyance()
    if args.action == "calibrer":
        _out(
            mem.calibrer_conteste(alpha=args.alpha if args.alpha is not None else 0.05)
        )
        return 0
    if args.entite is None or args.attribut is None:
        raise ValueError("--entite et --attribut requis pour cette action")
    if args.action == "assert":
        if args.valeur is None:
            raise ValueError("--valeur requise pour assert")
        mem.asserter(args.entite, args.attribut, args.valeur, t=args.t)
        mem.sauver(store)
        _out(
            {
                "ok": True,
                "entite": args.entite,
                "attribut": args.attribut,
                "valeur": args.valeur,
            }
        )
        return 0
    if args.action == "courant":
        res = mem.courant(args.entite, args.attribut)
        if res is None:
            raise ValueError(f"emplacement inconnu : {args.entite}/{args.attribut}")
        _out(res)
        return 0
    _out(mem.historique(args.entite, args.attribut))  # historique
    return 0


_COMMANDES = {
    "build": _cmd_build,
    "embed-prepare": _cmd_embed_prepare,
    "add": _cmd_add,
    "search": _cmd_search,
    "like": _cmd_like,
    "chemin": _cmd_chemin,
    "related": _cmd_related,
    "render": _cmd_render,
    "stats": _cmd_stats,
    "compter": _cmd_compter,
    "recents": _cmd_recents,
    "refs": _cmd_refs,
    "explain": _cmd_explain,
    "carte": _cmd_carte,
    "proches": _cmd_proches,
    "actuel": _cmd_actuel,
    "diff": _cmd_diff,
    "meta": _cmd_meta,
    "calibrer": _cmd_calibrer,
    "profil": _cmd_profil,
    "croyance": _cmd_croyance,
}


def main(argv: list[str] | None = None) -> int:
    # Force UTF-8 on stdin/stdout/stderr (nécessaire pour les noms de fichiers accentués et,
    # v1.5, les requêtes lues sur stdin en mode `search --batch`).
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # type: ignore

    args = _construire_parser().parse_args(argv)
    try:
        return _COMMANDES[args.cmd](args)
    except (OSError, ValueError, KeyError, struct.error) as exc:
        # KeyError/struct.error : un meta JSON amputé ou un binaire tronqué sortait en
        # traceback non-JSON (audit finding 24) — même contrat {"error": ...} pour tout.
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
