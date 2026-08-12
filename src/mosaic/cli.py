"""CLI Mosaic — toutes les sorties en JSON UTF-8, consommables par les agents."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from mosaic import carte
from mosaic.croyance import MemoireCroyance
from mosaic.embeddings import Embeddings, prepare
from mosaic.index import PROFILE_WEIGHTING_DEFAULT, SMOOTHING_RANK_DEFAULT, Index
from mosaic.lexicon import load_lexicon
from mosaic import profil as profil_module
from mosaic.meta import resume_par_index, rrf_fuse
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


def main(argv: list[str] | None = None) -> int:
    # Force UTF-8 on stdin/stdout/stderr (nécessaire pour les noms de fichiers accentués et,
    # v1.5, les requêtes lues sur stdin en mode `search --batch`).
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # type: ignore

    parser = argparse.ArgumentParser(prog="mosaic")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("corpus")
    p_build.add_argument("-o", "--output", required=True)
    p_build.add_argument("--grid", choices=sorted(_GRIDS), default="64x64")
    p_build.add_argument("--lexicon", default=None)
    p_build.add_argument("--lexicon-extra", default=None)
    p_build.add_argument("--embeddings", default=None)
    p_build.add_argument(
        "--weights",
        default=None,
        help="a,b,g (ex. 0.25,0.15,0.60), défaut WEIGHTS_DEFAULT",
    )
    p_build.add_argument(
        "--profile-weighting",
        default=PROFILE_WEIGHTING_DEFAULT,
        help=f"brut ou ppmi, défaut {PROFILE_WEIGHTING_DEFAULT}",
    )
    p_build.add_argument(
        "--smoothing-rank",
        type=int,
        default=SMOOTHING_RANK_DEFAULT,
        help=f"entier >= 0, défaut {SMOOTHING_RANK_DEFAULT}",
    )
    p_build.add_argument("--abtt", type=int, default=0, help="entier >= 0, défaut 0")
    p_build.add_argument(
        "--doc-weight",
        type=float,
        default=None,
        help="δ, poids du canal document (niveau sujet) dans [0, 1), défaut 0 (v1.2 exact)",
    )
    p_build.add_argument(
        "--cache-ingestion",
        action="store_true",
        help="cache opt-in des conversions markitdown sous tempfile.gettempdir()/mosaic_ingest/ "
        "(jamais à côté des documents, jamais activé par défaut, "
        "surchargeable par MOSAIC_INGEST_CACHE)",
    )
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

    p_embed = sub.add_parser("embed-prepare")
    p_embed.add_argument("vec_gz")
    p_embed.add_argument("-o", "--output", required=True)
    p_embed.add_argument("--keep", type=int, default=200_000)
    p_embed.add_argument(
        "--abtt",
        type=int,
        default=0,
        help="applique all-but-the-top À LA PRÉPARATION (table déjà nettoyée à l'écriture, "
        "cf. mosaic build --abtt) — entier dans [0, 255], défaut 0",
    )

    p_add = sub.add_parser("add")
    p_add.add_argument("file")
    p_add.add_argument("index")

    p_search = sub.add_parser("search")
    p_search.add_argument("query", nargs="?", default=None)
    p_search.add_argument("index")
    p_search.add_argument("--top", type=int, default=10)
    p_search.add_argument(
        "--fusion",
        action="store_true",
        help="fusion RRF à trois canaux (grille + BM25 + embeddings) — nécessite un "
        "index construit avec --hybride ; exclusif avec --rerank",
    )
    p_search.add_argument(
        "--rerank",
        action="store_true",
        help="repêcheur : reclasse le top --rerank-depth par mélange avec model2vec "
        "(nécessite un index construit avec --rerank-vectors)",
    )
    p_search.add_argument(
        "--rerank-lambda",
        type=float,
        default=0.70,
        help="λ du mélange λ·cos_mosaic + (1-λ)·cos_m2v, dans [0, 1], défaut 0.70",
    )
    p_search.add_argument(
        "--rerank-depth",
        type=int,
        default=50,
        help="profondeur du reclassement (top-N mosaic re-classé), entier >= 10, défaut 50",
    )
    p_search.add_argument(
        "--batch",
        action="store_true",
        help="lit les requêtes depuis stdin (une par ligne, UTF-8), affiche un tableau JSON "
        "par ligne — un seul process, un seul chargement pour N requêtes (agents "
        "faisant plusieurs recherches : amortit le coût de démarrage). Pas de `query` "
        "positionnelle dans ce mode.",
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
        default=0.7,
        help="λ de l'algèbre de connecteurs (poids de la soustraction), défaut 0.7",
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
        help="facette : poids de la fraîcheur dans [0,1] (0 = ordre sémantique pur, défaut). "
        "Fusion de rangs sens/date — les documents récents remontent, utile sur les dossiers "
        "qui évoluent (la date est lue dans le nom de fichier AAAA-MM-JJ).",
    )

    p_like = sub.add_parser("like")
    p_like.add_argument(
        "docs",
        nargs="+",
        help="1+ id(s) déjà indexé(s) ou chemin(s) de fichier externe, suivi(s) de l'index "
        "(dernier positionnel) ; 2+ documents = mélange (moyenne normalisée)",
    )
    p_like.add_argument("--top", type=int, default=10)
    p_like.add_argument(
        "--rerank",
        action="store_true",
        help="repêcheur : id interne réutilise son empreinte rerank stockée (aucun modèle "
        "requis) ; fichier externe encodé via model2vec (nécessite un index construit "
        "avec --rerank-vectors)",
    )
    p_like.add_argument(
        "--rerank-lambda",
        type=float,
        default=0.70,
        help="λ du mélange λ·cos_mosaic + (1-λ)·cos_m2v, dans [0, 1], défaut 0.70",
    )
    p_like.add_argument(
        "--rerank-depth",
        type=int,
        default=50,
        help="profondeur du reclassement (top-N mosaic re-classé), entier >= 10, défaut 50",
    )
    p_like.add_argument(
        "--cache-ingestion",
        action="store_true",
        help="cache opt-in des conversions markitdown d'un document-requête externe sous "
        "tempfile.gettempdir()/mosaic_ingest/ (même mécanique que `mosaic build`, "
        "surchargeable par MOSAIC_INGEST_CACHE)",
    )

    p_related = sub.add_parser("related")
    p_related.add_argument("entite")
    p_related.add_argument("index")
    p_related.add_argument("--role", choices=("dossier", "annee", "mois"), default=None)
    p_related.add_argument("--top", type=int, default=10)

    p_chemin = sub.add_parser("chemin")
    p_chemin.add_argument("doc_id", help="document de départ (id indexé)")
    p_chemin.add_argument("index")
    p_chemin.add_argument(
        "--role",
        choices=("dossier", "annee", "mois"),
        default=None,
        help="restreindre la traversée à un rôle (défaut : toutes les entités du document)",
    )
    p_chemin.add_argument("--top", type=int, default=10)

    p_render = sub.add_parser("render")
    p_render.add_argument("doc_id")
    p_render.add_argument("index")
    p_render.add_argument("-o", "--output", required=True)

    p_stats = sub.add_parser("stats")
    p_stats.add_argument("index")

    p_explain = sub.add_parser("explain")
    p_explain.add_argument("doc_id")
    p_explain.add_argument("index")
    p_explain.add_argument("--top", type=int, default=20)
    p_explain.add_argument("--query", default=None)

    p_carte = sub.add_parser("carte")
    p_carte.add_argument("dossier")
    p_carte.add_argument("--top-concepts", type=int, default=8)
    p_carte.add_argument(
        "--cache-ingestion",
        action="store_true",
        help="cache opt-in des conversions markitdown sous tempfile.gettempdir()/mosaic_ingest/ "
        "(jamais à côté des documents, jamais activé par défaut, "
        "surchargeable par MOSAIC_INGEST_CACHE)",
    )
    p_carte.add_argument(
        "--profile-weighting",
        default=PROFILE_WEIGHTING_DEFAULT,
        help=f"brut ou ppmi, défaut {PROFILE_WEIGHTING_DEFAULT}",
    )
    p_carte.add_argument(
        "--smoothing-rank",
        type=int,
        default=SMOOTHING_RANK_DEFAULT,
        help=f"entier >= 0, défaut {SMOOTHING_RANK_DEFAULT}",
    )

    p_proches = sub.add_parser("proches")
    p_proches.add_argument("mot")
    p_proches.add_argument(
        "table",
        help="table .msee (voisins génériques, ex. data_externes/potion_fr.msee) OU un "
        "dossier d'index (voisins SÉMANTIQUES appris de ce corpus, ex. ./index_devis)",
    )
    p_proches.add_argument("--top", type=int, default=10)
    p_proches.add_argument(
        "--abtt",
        type=int,
        default=0,
        help="all-but-the-top au chargement (0 = table déjà nettoyée)",
    )
    p_proches.add_argument(
        "--dico",
        action="store_true",
        help="voisinage strictement LEXICAL : ne garde que les vrais mots français (vocab de la "
        "table potion embarquée), écarte le boilerplate de contenu (collocations, codes, dates) "
        "au prix des codes-métier hors dictionnaire. Sans effet sur une table .msee.",
    )

    p_actuel = sub.add_parser("actuel")
    p_actuel.add_argument("query")
    p_actuel.add_argument("index")
    p_actuel.add_argument("--top", type=int, default=10)
    p_actuel.add_argument(
        "--seuil-version",
        type=float,
        default=SEUIL_VERSION_DEFAUT,
        help="cosinus au-dessus duquel deux documents sont des versions du même aspect "
        f"(défaut {SEUIL_VERSION_DEFAUT} ; date lue dans le nom de fichier AAAA-MM-JJ)",
    )

    p_meta = sub.add_parser("meta")
    p_meta.add_argument("query")
    p_meta.add_argument(
        "indexes",
        nargs="+",
        help="2+ dossiers d'index à interroger ensemble (ex. ./index_devis "
        "./index_courriels). Le nom affiché = nom du dossier.",
    )
    p_meta.add_argument("--top", type=int, default=10)
    p_meta.add_argument(
        "--profondeur",
        type=int,
        default=20,
        help="nombre de résultats tirés de CHAQUE index avant fusion (défaut 20)",
    )
    p_meta.add_argument(
        "--rerank",
        action="store_true",
        help="applique le repêcheur sur chaque index avant fusion (si rerank.msrv présent)",
    )
    p_meta.add_argument(
        "--resume",
        action="store_true",
        help="ajoute un diagnostic de rappel par index (candidats, meilleur score local)",
    )

    p_profil = sub.add_parser("profil")
    p_profil.add_argument(
        "cible",
        help="dossier d'INDEX (voir son profil) ou de CORPUS (avec --suggere : proposer un "
        "profil calibré sur cet environnement)",
    )
    p_profil.add_argument(
        "--explique",
        action="store_true",
        help="mode humain : raconte en français ce que le profil fait faire au moteur",
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
        default="fr",
        help="langue du mode --explique (défaut fr ; en = English explanation)",
    )

    p_calib = sub.add_parser("calibrer")
    p_calib.add_argument("corpus")
    p_calib.add_argument(
        "--requetes",
        default=None,
        help='jeu de requêtes-vérité (JSONL : {"query": ..., "relevant": [doc_ids]}) — '
        "c'est LUI qui choisit les poids, jamais un curseur manuel",
    )
    p_calib.add_argument(
        "--verite-auto",
        action="store_true",
        help="génère la vérité DÉTERMINISTIQUEMENT (held-out : moitié indexée, requête = "
        "termes distinctifs de l'autre moitié) — zéro LLM, zéro humain. Plancher utile "
        "sans requêtes rédigées ; celles-ci restent l'étalon-or.",
    )
    p_calib.add_argument("--embeddings", default=None)
    p_calib.add_argument("--abtt", type=int, default=0)
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
        help="verdict en clair (mode humain) au lieu du rapport JSON",
    )
    p_calib.add_argument("--langue", choices=("fr", "en"), default="fr")

    p_croy = sub.add_parser("croyance")
    p_croy.add_argument(
        "action", choices=["assert", "courant", "historique", "calibrer"]
    )
    p_croy.add_argument("store", help="fichier de mémoire de croyance (.jsonl)")
    p_croy.add_argument("--entite", default=None)
    p_croy.add_argument("--attribut", default=None)
    p_croy.add_argument(
        "--valeur", default=None, help="valeur du fait (requis pour assert)"
    )
    p_croy.add_argument(
        "--t", type=float, default=None, help="ordre temporel (défaut : à la suite)"
    )
    p_croy.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="calibrer : taux d'erreur cible (le seuil « contesté » est CALIBRÉ sur les "
        "faits du store — prédiction conforme, sans donnée externe ; défaut 0.05)",
    )

    args = parser.parse_args(argv)
    try:
        if args.cmd == "build":
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
            smoothing_rank = _parse_int_nonnegative(
                args.smoothing_rank, "smoothing-rank"
            )
            abtt = _parse_int_nonnegative(args.abtt, "abtt")
            doc_weight = (
                _parse_doc_weight(args.doc_weight)
                if args.doc_weight is not None
                else 0.0
            )
            ingest_cache_dir = _ingest_cache_root() if args.cache_ingestion else None
            idx = Index.build(
                Path(args.corpus),
                Path(args.output),
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
                grilles_typees=args.grilles_typees,
                index_paths=not args.no_path_tokens,
                ocr=args.ocr,
                relations=args.relations,
                type_doc=args.type_doc,
                profil=(
                    profil_module.charger(Path(args.profil)) if args.profil else None
                ),
            )
            _out(idx.stats())
        elif args.cmd == "embed-prepare":
            lex_pivots = load_lexicon()
            extra_words: set[str] = set()
            for pivot in lex_pivots.values():
                extra_words.update(pivot.split("_"))
            abtt = _parse_int_nonnegative(args.abtt, "abtt")
            stats = prepare(
                Path(args.vec_gz),
                Path(args.output),
                keep=args.keep,
                extra_words=extra_words,
                abtt=abtt,
            )
            _out(stats)
        elif args.cmd == "add":
            idx = Index.open(Path(args.index))
            idx.add(Path(args.file))
            _out(idx.stats())
        elif args.cmd == "search":
            if args.top < 1:
                raise ValueError("--top doit être >= 1")
            rerank_lambda = _parse_rerank_lambda(args.rerank_lambda)
            rerank_depth = _parse_rerank_depth(args.rerank_depth)
            # verify_embeddings=False : chemin recherche uniquement (jamais build/add) —
            # cf. Index.open() et Embeddings.load(verify=...).
            idx = Index.open(Path(args.index), verify_embeddings=False)
            if args.connecteurs and (args.batch or args.rerank):
                raise ValueError(
                    "--connecteurs est incompatible avec --batch et --rerank"
                )
            if args.fusion and args.connecteurs:
                raise ValueError("--fusion est incompatible avec --connecteurs")
            if args.connecteurs and (args.type_filtre or args.recence):
                raise ValueError("--connecteurs est incompatible avec --type/--recence")
            if args.batch:
                if args.query is not None:
                    raise ValueError(
                        "--batch : pas de `query` positionnelle, les requêtes viennent de "
                        "stdin (une par ligne)"
                    )
                for line in sys.stdin:
                    query = line.rstrip("\r\n")
                    if not query:
                        continue
                    _out(
                        idx.search(
                            query,
                            k=args.top,
                            rerank=args.rerank,
                            rerank_lambda=rerank_lambda,
                            rerank_depth=rerank_depth,
                            type_filtre=args.type_filtre,
                            recence=args.recence,
                            fusion=args.fusion,
                        )
                    )
            elif args.connecteurs:
                if args.query is None:
                    raise ValueError("query requise")
                _out(
                    idx.search_connecteurs(args.query, k=args.top, lam=args.conn_lambda)
                )
            else:
                if args.query is None:
                    raise ValueError("query requise (sauf en mode --batch)")
                _out(
                    idx.search(
                        args.query,
                        k=args.top,
                        rerank=args.rerank,
                        rerank_lambda=rerank_lambda,
                        rerank_depth=rerank_depth,
                        type_filtre=args.type_filtre,
                        recence=args.recence,
                        fusion=args.fusion,
                    )
                )
        elif args.cmd == "like":
            if len(args.docs) < 2:
                raise ValueError(
                    "mosaic like requiert au moins un document (id ou chemin) et un index "
                    "(dernier positionnel)"
                )
            if args.top < 1:
                raise ValueError("--top doit être >= 1")
            rerank_lambda = _parse_rerank_lambda(args.rerank_lambda)
            rerank_depth = _parse_rerank_depth(args.rerank_depth)
            *doc_specs, index_path = args.docs
            ingest_cache_dir = _ingest_cache_root() if args.cache_ingestion else None
            idx = Index.open(Path(index_path), verify_embeddings=False)
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
        elif args.cmd == "chemin":
            if args.top < 1:
                raise ValueError("--top doit être >= 1")
            idx = Index.open(Path(args.index), verify_embeddings=False)
            _out(idx.chemin(args.doc_id, k=args.top, role=args.role))
        elif args.cmd == "related":
            if args.top < 1:
                raise ValueError("--top doit être >= 1")
            idx = Index.open(Path(args.index), verify_embeddings=False)
            _out(idx.related(args.entite, k=args.top, role=args.role))
        elif args.cmd == "render":
            render_doc(Index.open(Path(args.index)), args.doc_id, Path(args.output))
            _out({"ok": True, "out": args.output})
        elif args.cmd == "stats":
            _out(Index.open(Path(args.index)).stats())
        elif args.cmd == "explain":
            if args.top < 1:
                raise ValueError("--top doit être >= 1")
            idx = Index.open(Path(args.index))
            if args.query:
                _out(idx.explain_match(args.query, args.doc_id, k=args.top))
            else:
                _out(idx.explain(args.doc_id, k=args.top))
        elif args.cmd == "carte":
            if args.top_concepts < 1:
                raise ValueError("--top-concepts doit être >= 1")
            dossier = Path(args.dossier)
            ingest_cache_dir = _ingest_cache_root() if args.cache_ingestion else None
            profile_weighting = _parse_profile_weighting(args.profile_weighting)
            smoothing_rank = _parse_int_nonnegative(
                args.smoothing_rank, "smoothing-rank"
            )
            out = carte.generer(
                dossier,
                k_concepts=args.top_concepts,
                ingest_cache_dir=ingest_cache_dir,
                profile_weighting=profile_weighting,
                smoothing_rank=smoothing_rank,
            )
            docs = Index.open(carte.index_dir(dossier)).stats()["docs"]
            _out({"docs": docs, "html": str(out)})
        elif args.cmd == "proches":
            if args.top < 1:
                raise ValueError("--top doit être >= 1")
            source = Path(args.table)
            if source.is_dir():  # index de corpus -> voisins sémantiques métier
                voisins = Index.open(source, verify_embeddings=False).proches(
                    args.mot, k=args.top, dico=args.dico
                )
                if voisins is None:
                    raise ValueError(
                        f"mot absent du vocabulaire (ou sans cooccurrence) : {args.mot}"
                    )
            else:  # table .msee -> voisins génériques
                emb = Embeddings.load(source, abtt=args.abtt)
                voisins = emb.proches(args.mot, k=args.top)
                if voisins is None:
                    raise ValueError(f"mot absent de la table : {args.mot}")
            _out(
                [
                    {"mot": mot, "score": score, "rang": rang}
                    for rang, (mot, score) in enumerate(voisins, start=1)
                ]
            )
        elif args.cmd == "actuel":
            if args.top < 1:
                raise ValueError("--top doit être >= 1")
            idx = Index.open(Path(args.index), verify_embeddings=False)
            _out(
                versions_actuelles(
                    idx, args.query, k=args.top, seuil_version=args.seuil_version
                )
            )
        elif args.cmd == "meta":
            if args.top < 1:
                raise ValueError("--top doit être >= 1")
            if args.profondeur < 1:
                raise ValueError("--profondeur doit être >= 1")
            if len(args.indexes) < 2:
                raise ValueError("meta : donner au moins 2 index à fusionner")
            listes = []
            for chemin in args.indexes:
                p = Path(chemin)
                nom = p.name
                idx = Index.open(p, verify_embeddings=False)
                res = idx.search(args.query, k=args.profondeur, rerank=args.rerank)
                listes.append((nom, res))
            fusion = rrf_fuse(listes, k=args.top)
            if args.resume:
                _out({"resultats": fusion, "resume": resume_par_index(listes)})
            else:
                _out(fusion)
        elif args.cmd == "calibrer":
            from mosaic.calibration import calibrer, expliquer_calibration

            if args.verite_auto:
                requetes = None
            elif args.requetes:
                requetes = [
                    json.loads(ligne)
                    for ligne in Path(args.requetes)
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if ligne.strip()
                ]
            else:
                raise ValueError("donner --requetes <verite.jsonl> ou --verite-auto")
            rapport = calibrer(
                Path(args.corpus),
                requetes,
                embeddings_path=Path(args.embeddings) if args.embeddings else None,
                abtt=_parse_int_nonnegative(args.abtt, "abtt"),
                verite_auto=args.verite_auto,
                index_paths=not args.no_path_tokens,
            )
            if args.explique:
                print(expliquer_calibration(rapport, langue=args.langue))
            else:
                _out(rapport)
        elif args.cmd == "profil":
            cible = Path(args.cible)
            if args.suggere:
                if not cible.is_dir():
                    raise ValueError(f"corpus introuvable : {cible}")
                suggestion = profil_module.suggerer(cible)
                if args.explique:
                    print(profil_module.expliquer(suggestion, langue=args.langue))
                else:
                    _out(suggestion)
            else:
                # cible = un index : montrer son profil actif (meta), None = défauts
                idx = Index.open(cible, verify_embeddings=False)
                if args.explique:
                    print(profil_module.expliquer(idx.profil, langue=args.langue))
                else:
                    _out(idx.profil or {"profil": None, "note": "défauts historiques"})
        elif args.cmd == "croyance":
            store = Path(args.store)
            mem = (
                MemoireCroyance.charger(store) if store.exists() else MemoireCroyance()
            )
            if args.action == "calibrer":
                _out(mem.calibrer_conteste(alpha=args.alpha))
            elif args.entite is None or args.attribut is None:
                raise ValueError("--entite et --attribut requis pour cette action")
            elif args.action == "assert":
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
            elif args.action == "courant":
                res = mem.courant(args.entite, args.attribut)
                if res is None:
                    raise ValueError(
                        f"emplacement inconnu : {args.entite}/{args.attribut}"
                    )
                _out(res)
            else:  # historique
                _out(mem.historique(args.entite, args.attribut))
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
