"""Wrapper MCP `mosaic` — serveur JSON-RPC 2.0 minimal sur stdio (spec v1.6 §E).

Zéro dépendance au SDK `mcp` officiel (préférence spec §E : léger, un seul fichier, rien à
installer côté serveur). Transport MCP stdio réel : messages newline-delimited JSON, un objet
JSON par ligne, aussi bien en lecture (stdin) qu'en écriture (stdout) — pas de framing
Content-Length (celui-ci concerne le transport HTTP/pipe historique, pas le stdio simple).
Tous les logs vont sur stderr : stdout ne porte jamais que des réponses JSON-RPC, un seul
octet de bruit dessus casserait le client.

Dispatch pur et testable sans I/O : `handle_request(request, state) -> response | None`. Les
tests (`tests/test_mcp.py`) le pilotent directement avec de faux payloads JSON-RPC contre un
petit index construit à la volée — `run_stdio()` (boucle d'I/O réelle) n'est couverte que par
un test de fumée en subprocess.

`state["cache"]` garde les `Index` ouverts, un par domaine, réutilisés entre les appels — la
raison d'être du serveur plutôt que la CLI : `mosaic search` réouvre l'index à chaque process
(~1-2 s), le serveur ne l'ouvre qu'une fois puis répond en ~50 ms. `state["cache_mtime"]`
retient la mtime de `vocab.msev` au moment de l'ouverture : un rebuild planifié
(`LOCAL_Mosaic_Rebuild`) la fait changer, et l'appel suivant réouvre l'index plutôt que de
servir indéfiniment le contenu chargé au premier `open()` du process (revue finale v1.6,
Critical — cf. `_get_index`).
"""

import json
import os
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mosaic.croyance import MemoireCroyance  # noqa: E402
from mosaic.index import Index  # noqa: E402

SERVER_NAME = "mosaic"
SERVER_VERSION = "1.6.0"
# Version de protocole MCP stable la plus largement supportée au moment d'écrire ce serveur —
# aucune négociation dynamique ici (serveur minimal, un seul comportement).
PROTOCOL_VERSION = "2024-11-05"

# domaine -> /opt/mosaic/mosaic/index_<domaine> (chemins de production, cf. mémo agents LOCAL
# reference_mosaic_usage_agents.md et scripts/reconstruire_index.py §D).
# Les domaines sont DÉCOUVERTS dynamiquement : tout dossier `index_<domaine>` sous la racine
# des index est un domaine interrogeable (plus de liste en dur — un nouvel index déposé par un
# script ou un rebuild devient accessible sans toucher au serveur). DEFAULT_DATA_DIR reste le
# défaut local LOCAL ; surchargé par MOSAIC_MCP_DATA_DIR (tests, autre machine).
DEFAULT_DATA_DIR = Path("/opt/mosaic/mosaic")


def domaines_disponibles(data_dir: Path) -> list[str]:
    """Les domaines réellement présents sur disque (dossiers index_*), triés."""
    try:
        return sorted(
            p.name[len("index_") :]
            for p in Path(data_dir).iterdir()
            if p.is_dir() and p.name.startswith("index_")
        )
    except OSError:
        return []


TOOLS = [
    {
        "name": "mosaic_search",
        "description": (
            "Recherche sémantique dans un index Mosaic de production LOCAL (question en "
            "français, la paraphrase est acceptée, pas besoin des mots exacts). "
            "FORMULER LA QUESTION AVEC LES MOTS DU DOCUMENT, PAS CEUX DU BESOIN — c'est "
            "le facteur le plus déterminant, mesuré : la même recherche en 3 mots "
            "discriminants sort au rang 2, décorée de 6 mots plausibles mais absents du "
            "document, au rang 231. Une pièce scannée ne contient ni son fournisseur "
            "(logo non OCRisé) ni son type développé ('BL', jamais 'bon de livraison'). "
            "Question COURTE et discriminante d'abord ; élargir seulement si vide. "
            "QUALIFIER quand vous savez : c'est vous qui connaissez l'intention de "
            "l'utilisateur, le moteur ne connaît que le corpus — un `type` ou une "
            "`recence` passés font gagner plus que n'importe quelle reformulation. Un "
            "filtre qui écarterait tout vous est SIGNALÉ (avec le vocabulaire réel du "
            "domaine), donc qualifier ne fait courir aucun risque. "
            "FORMULER LA QUESTION AVEC LES MOTS DU DOCUMENT, PAS CEUX DU BESOIN — c'est "
            "le facteur le plus déterminant, mesuré : la même recherche en 3 mots "
            "discriminants sort au rang 2, décorée de 6 mots plausibles mais absents du "
            "document, au rang 231. Une pièce scannée ne contient ni son fournisseur "
            "(logo non OCRisé) ni son type développé ('BL', jamais 'bon de livraison'). "
            "Question COURTE et discriminante d'abord ; élargir seulement si vide. "
            "QUALIFIER quand vous savez : c'est vous qui connaissez l'intention de "
            "l'utilisateur, le moteur ne connaît que le corpus — un `type` ou une "
            "`recence` passés font gagner plus que n'importe quelle reformulation. Un "
            "filtre qui écarterait tout vous est SIGNALÉ (avec le vocabulaire réel du "
            "domaine), donc qualifier ne fait courir aucun risque. "
            "MODE D'EMPLOI (choisir les bonnes options rend la réponse meilleure) : "
            "question datée ou dossier qui évolue -> recence=0.5 (les versions récentes "
            "remontent, ne pas prendre une donnée périmée pour canonique) ; besoin d'un type "
            "précis de document -> type ('tableur' pour une liste de prix, 'photo' pour un "
            "relevé chantier, 'pdf scanné' vs 'pdf numérique', 'document rédigé', 'page web', "
            "'note texte') ; une RÉFÉRENCE/code dans la question (réf produit, numéro de "
            "devis/commande) -> boost AUTOMATIQUE : les documents portant ce code exactement "
            "remontent en tête avec le champ ref_exacte (pour un produit, préférer le domaine "
            "'produits')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question en langage naturel.",
                },
                "domaine": {
                    "type": "string",
                    "description": "nom du domaine (= dossier index_<domaine> sous la racine des index).",
                },
                "top": {"type": "integer", "default": 5, "minimum": 1},
                "fusion": {
                    "type": "boolean",
                    "default": False,
                    "description": "Fusion RRF multi-canaux (grille+BM25+embeddings, +atlas si "
                    "présent) — nécessite un index construit avec --hybride ; gagnant mesuré "
                    "sur terrain LEXICAL (la question recopie le vocabulaire des documents). "
                    "Exclusif avec rerank.",
                },
                "grammatical": {
                    "type": "boolean",
                    "default": False,
                    "description": "Canal grammatical (nécessite un index --grammatical). "
                    "DÉCONSEILLÉ — MESURÉ SANS APPORT : sur 2 316 requêtes réelles, il "
                    "rattrape 7 requêtes que la grille rate et en fait perdre 19 ; il voit "
                    "le même monde que la grille, en plus flou. Laissé pour la recherche "
                    "et le portage vers d'autres langues, pas pour interroger la "
                    "production : préférer fusion. Exclusif avec rerank/fusion/type/recence.",
                },
                "rerank": {
                    "type": "boolean",
                    "default": True,
                    "description": "Repêcheur model2vec — nécessite un index construit avec "
                    "--rerank-vectors (c'est le cas des index de production).",
                },
                "type": {
                    "type": ["string", "null"],
                    "default": None,
                    "description": "Facette : ne garder que ce type de document (tableur, "
                    "pdf numérique, pdf scanné, photo, document rédigé, présentation, "
                    "page web, note texte). Nécessite un index avec facettes.json.",
                },
                "recence": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Facette : poids de la fraîcheur (0 = ordre sémantique "
                    "pur). 0.5 équilibre sens et date ; 1.0 = le plus récent d'abord.",
                },
            },
            "required": ["question", "domaine"],
        },
    },
    {
        "name": "mosaic_explain",
        "description": (
            "Décompose un document indexé en ses tokens dominants (démélange). Avec "
            "`question`, donne plutôt la contribution de chaque token de la question au "
            "score de ce document (justifie un match)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "id du document (chemin relatif au corpus).",
                },
                "domaine": {
                    "type": "string",
                    "description": "nom du domaine (= dossier index_<domaine> sous la racine des index).",
                },
                "question": {"type": ["string", "null"], "default": None},
            },
            "required": ["doc_id", "domaine"],
        },
    },
    {
        "name": "mosaic_like",
        "description": (
            "Chiffrage par similarité : la requête est un document ENTIER (id déjà indexé ou "
            "chemin de fichier accessible du serveur), pas une phrase — retourne les documents "
            "les plus proches, jamais le document source lui-même."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "document": {
                    "type": "string",
                    "description": "id indexé ou chemin de fichier.",
                },
                "domaine": {
                    "type": "string",
                    "description": "nom du domaine (= dossier index_<domaine> sous la racine des index).",
                },
                "top": {"type": "integer", "default": 5, "minimum": 1},
            },
            "required": ["document", "domaine"],
        },
    },
    {
        "name": "mosaic_croyance_assert",
        "description": (
            "Mémoire de croyance : asserte un FAIT (entité, attribut, valeur). Le fait le plus "
            "récent devient la vérité-courante ; l'historique est conservé. Déterministe, local."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entite": {"type": "string"},
                "attribut": {"type": "string"},
                "valeur": {"type": "string"},
                "t": {
                    "type": ["number", "null"],
                    "default": None,
                    "description": "ordre temporel (défaut : à la suite de l'historique).",
                },
            },
            "required": ["entite", "attribut", "valeur"],
        },
    },
    {
        "name": "mosaic_croyance_courant",
        "description": (
            "Mémoire de croyance : la vérité-COURANTE d'un (entité, attribut) — "
            "{valeur, confiance, conteste}. `conteste`=true si deux valeurs se disputent."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entite": {"type": "string"},
                "attribut": {"type": "string"},
            },
            "required": ["entite", "attribut"],
        },
    },
    {
        "name": "mosaic_croyance_historique",
        "description": (
            "Mémoire de croyance : toutes les valeurs PASSÉES d'un (entité, attribut), "
            "ordre chronologique."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entite": {"type": "string"},
                "attribut": {"type": "string"},
            },
            "required": ["entite", "attribut"],
        },
    },
    {
        "name": "mosaic_meta",
        "description": (
            "MÉTA-RECHERCHE : interroge PLUSIEURS domaines d'un coup et fusionne par rangs "
            "(RRF — les scores entre index ne sont pas comparables, les rangs si). À utiliser "
            "quand la question peut vivre dans plusieurs corpus (« le chantier X est-il "
            "payé ? » -> compta + comms + chantiers). Chaque résultat garde sa provenance "
            "(index), son rang et son score locaux."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "domaines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "2+ domaines à fusionner.",
                },
                "top": {"type": "integer", "default": 8, "minimum": 1},
            },
            "required": ["question", "domaines"],
        },
    },
    {
        "name": "mosaic_actuel",
        "description": (
            "VÉRITÉ TEMPORELLE : regroupe les VERSIONS successives d'un même aspect et rend "
            "la plus récente comme CANONIQUE, les autres marquées PÉRIMÉES — pour ne jamais "
            "prendre une donnée obsolète pour vérité sur un dossier qui évolue (la date est "
            "lue dans le nom de fichier AAAA-MM-JJ)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "domaine": {
                    "type": "string",
                    "description": "nom du domaine (= dossier index_<domaine> sous la racine des index).",
                },
                "top": {"type": "integer", "default": 8, "minimum": 1},
            },
            "required": ["question", "domaine"],
        },
    },
    {
        "name": "mosaic_chemin",
        "description": (
            "PARCOURS MULTI-SAUTS : depuis un document, retrouve « les autres documents du "
            "même dossier / de la même année » — deux sauts vectoriels (doc -> ses entités "
            "-> leurs documents frères). `role` restreint (dossier/annee/mois, ou les rôles "
            "du profil de l'index). Nécessite un index construit avec relations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"},
                "domaine": {
                    "type": "string",
                    "description": "nom du domaine (= dossier index_<domaine> sous la racine des index).",
                },
                "role": {"type": ["string", "null"], "default": None},
                "top": {"type": "integer", "default": 8, "minimum": 1},
            },
            "required": ["doc_id", "domaine"],
        },
    },
    {
        "name": "mosaic_stats",
        "description": (
            "DÉCOUVERTE d'un domaine : nombre de documents, canaux actifs (relations, "
            "rerank), et le PROFIL de l'index (rôles d'arborescence, types custom, critère "
            "de réfs) — à consulter pour formuler des questions adaptées à l'environnement."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "domaine": {
                    "type": "string",
                    "description": "nom du domaine (= dossier index_<domaine>).",
                }
            },
            "required": ["domaine"],
        },
    },
    {
        "name": "mosaic_compter",
        "description": (
            "COMPTE EXACT de documents — pas une recherche. À utiliser dès que la question "
            "est « combien » (« combien de BL Fournisseur en juin ? », « combien de photos sur ce "
            "chantier ? ») : une recherche sémantique rend des documents classés par "
            "ressemblance, elle ne sait pas COMPTER. Filtres cumulatifs et tous optionnels : "
            "chemin (fragment, cherché littéralement), type, date (préfixe 2026 / 2026-06 / "
            "2026-06-22). Rend aussi `sans_date` : le nombre de documents SANS date connue, "
            "donc exclus des comptes datés — le lire avant de conclure « il n'y en a que N »."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "domaine": {"type": "string", "description": "nom du domaine."},
                "chemin": {
                    "type": "string",
                    "description": "fragment de chemin (ex. 'Bons de Livraison/FOURNISSEUR').",
                },
                "type": {"type": "string", "description": "type de document."},
                "date": {
                    "type": "string",
                    "description": "préfixe de date : 2026, 2026-06, 2026-06-22.",
                },
                "par_mois": {
                    "type": "boolean",
                    "description": "ajouter la répartition par mois.",
                },
            },
            "required": ["domaine"],
        },
    },
    {
        "name": "mosaic_recents",
        "description": (
            "Les k documents les plus RÉCENTS, par ordre de date exact (pas de pertinence). "
            "Pour « le dernier bon de livraison », « les 3 dernières notes de ce chantier ». "
            "Les documents dont la date est inconnue sont EXCLUS — jamais classés au hasard. "
            "Différent de mosaic_search avec recence : ici l'ordre est la date, point."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "domaine": {"type": "string", "description": "nom du domaine."},
                "k": {
                    "type": "integer",
                    "description": "combien en rendre (défaut 5).",
                },
                "chemin": {"type": "string", "description": "fragment de chemin."},
                "type": {"type": "string", "description": "type de document."},
            },
            "required": ["domaine"],
        },
    },
    {
        "name": "mosaic_refs",
        "description": (
            "Quels documents portent CETTE référence exacte, à travers PLUSIEURS domaines ? "
            "C'est la jointure que la recherche ne sait pas faire : une même réf (n° de BL, "
            "code article, n° de devis) relie un devis, un bon de livraison et une facture "
            "sans qu'aucun mot ne se ressemble. Rend (domaine, document, date), du plus "
            "récent au plus ancien. Pour la traçabilité chantier -> pièce fournisseur."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "référence EXACTE (ex. '9990001', '9990004').",
                },
                "domaines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "domaines à joindre ; défaut : tous les disponibles.",
                },
            },
            "required": ["ref"],
        },
    },
    {
        "name": "mosaic_diff",
        "description": (
            "Diff SÉMANTIQUE entre deux index d'un même corpus à deux moments : ce qui a "
            "changé de SENS, pas seulement de contenu — vocabulaire apparu/disparu, mots "
            "dont le contexte a dérivé, déclins/croissances d'usage, et documents dont la "
            "grille a bougé. Garantie : deux index identiques rendent un diff strictement "
            "vide (déterminisme). Les deux index doivent partager la même configuration "
            "d'encodage (refus net sinon)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "domaine_avant": {
                    "type": "string",
                    "description": "domaine de l'état t1 (= dossier index_<domaine>).",
                },
                "domaine_apres": {
                    "type": "string",
                    "description": "domaine de l'état t2.",
                },
                "top": {"type": "integer", "default": 20, "minimum": 1},
            },
            "required": ["domaine_avant", "domaine_apres"],
        },
    },
]


def _default_data_dir() -> Path:
    """Racine des 4 index de production, surchargeable par MOSAIC_MCP_DATA_DIR (tests —
    jamais pointer les tests vers /opt/mosaic/mosaic, index de production en lecture seule)."""
    override = os.environ.get("MOSAIC_MCP_DATA_DIR")
    return Path(override) if override else DEFAULT_DATA_DIR


def new_state(data_dir: Path | str | None = None) -> dict:
    """État initial du serveur : cache d'Index vide + racine des index."""
    resolved = Path(data_dir) if data_dir is not None else _default_data_dir()
    return {"data_dir": resolved, "cache": {}, "cache_mtime": {}}


def _vocab_mtime(index_dir: Path) -> float | None:
    """mtime de vocab.msev, ou None si illisible (fenêtre de swap atomique d'un rebuild en
    cours — cf. mosaic.store._write — ou index disparu) : ne doit jamais faire planter une
    lecture, juste renoncer à la comparaison de fraîcheur pour cet appel."""
    try:
        return (index_dir / "vocab.msev").stat().st_mtime
    except OSError:
        return None


def _get_index(state: dict, domaine: str) -> Index:
    index_dir = Path(state["data_dir"]) / f"index_{domaine}"
    if not index_dir.is_dir():
        dispo = domaines_disponibles(Path(state["data_dir"]))
        raise ValueError(
            f"domaine inconnu : {domaine!r} (disponibles : {dispo or 'aucun'})"
        )
    cache = state["cache"]
    cache_mtime = state.setdefault("cache_mtime", {})
    current_mtime = _vocab_mtime(index_dir)
    # Réutilise l'Index en cache seulement si sa mtime d'ouverture correspond ENCORE à celle
    # sur disque — un rebuild planifié (LOCAL_Mosaic_Rebuild, dimanche 03h) remplace
    # atomiquement vocab.msev (cf. Fix atomique de _write) : sans cette comparaison, le
    # serveur MCP servirait indéfiniment le contenu du premier open() du process, y compris
    # des résultats vieux d'une semaine (revue finale v1.6, Critical). `current_mtime is
    # None` (stat ratée, fenêtre de swap) ne déclenche jamais de réouverture inutile — on
    # retombe sur le cache si présent.
    if domaine in cache and (
        current_mtime is None or cache_mtime.get(domaine) == current_mtime
    ):
        return cache[domaine]
    # verify_embeddings=False : chemin recherche/consultation uniquement, jamais de mutation
    # (aucun outil MCP n'appelle add()) — cf. Index.open() et la CLI `mosaic search`.
    idx = Index.open(index_dir, verify_embeddings=False)
    idx.chauffer_recherche()  # process long : RAM contre latence (x11 mesuré)
    cache[domaine] = idx
    cache_mtime[domaine] = current_mtime
    return idx


def _require(args: dict, *names: str) -> None:
    manquants = [n for n in names if not args.get(n)]
    if manquants:
        raise ValueError(f"paramètre(s) requis manquant(s) : {', '.join(manquants)}")


def _conseil_aiguillage(question: str) -> str | None:
    """Un mot si un AUTRE outil répondrait mieux — sinon rien.

    Ne conseille que les deux circuits qui ont un outil dédié rendant une réponse
    d'une autre NATURE (un compte, un ordre exact). Le circuit référence n'est pas
    conseillé : son boost est déjà automatique dans la recherche, le signaler
    serait du bruit."""
    from mosaic.aiguilleur import Circuit, aiguiller

    r = aiguiller(question)
    if r.circuit is Circuit.COMPTAGE:
        return (
            f"cette question attend un NOMBRE ({r.motif}) — `mosaic_compter` y répond "
            "exactement, là où une recherche ne peut que rendre des documents "
            "ressemblants ; ci-dessous les résultats de la recherche, au cas où"
        )
    if r.circuit is Circuit.ORDRE:
        return (
            f"cette question attend LE document en tête d'un ordre ({r.motif}) — "
            "`mosaic_recents` classe par date exacte, là où une recherche classe par "
            "ressemblance ; ci-dessous les résultats de la recherche, au cas où"
        )
    return None


def _diagnostic_filtre_vide(
    state: dict, args: dict, idx, top: int, rerank: bool, fusion: bool
) -> object:
    """Rejoue la recherche SANS le filtre de type et rend un diagnostic lisible.

    N'est appelé que dans le cas rare où le filtre a tout écarté : le coût d'une
    seconde recherche ne se paie donc jamais en régime normal."""
    sans = idx.search(args["question"], k=top, rerank=rerank, fusion=fusion)
    types = {}
    try:
        magasin, charges = _magasin_pour(state, [args["domaine"]])
        if charges:
            types = magasin.types_disponibles(args["domaine"])
    except (OSError, ValueError):
        pass  # le diagnostic est un service rendu, jamais une raison d'échouer
    return {
        "resultats": [],
        "filtre_ecarte_tout": {
            "type_demande": args.get("type"),
            "sans_ce_filtre": len(sans),
            "premier_sans_filtre": sans[0]["id"] if sans else None,
            "types_reels_du_domaine": types,
            "conseil": (
                "le type demandé n'existe pas dans ce domaine ou ne correspond à "
                "aucun résultat pertinent — relancer sans `type`, ou avec l'un des "
                "types réellement présents ci-dessus"
            ),
        },
    }


def _conseil_aiguillage(question: str) -> str | None:
    """Un mot si un AUTRE outil répondrait mieux — sinon rien.

    Ne conseille que les deux circuits qui ont un outil dédié rendant une réponse
    d'une autre NATURE (un compte, un ordre exact). Le circuit référence n'est pas
    conseillé : son boost est déjà automatique dans la recherche, le signaler
    serait du bruit."""
    from mosaic.aiguilleur import Circuit, aiguiller

    r = aiguiller(question)
    if r.circuit is Circuit.COMPTAGE:
        return (
            f"cette question attend un NOMBRE ({r.motif}) — `mosaic_compter` y répond "
            "exactement, là où une recherche ne peut que rendre des documents "
            "ressemblants ; ci-dessous les résultats de la recherche, au cas où"
        )
    if r.circuit is Circuit.ORDRE:
        return (
            f"cette question attend LE document en tête d'un ordre ({r.motif}) — "
            "`mosaic_recents` classe par date exacte, là où une recherche classe par "
            "ressemblance ; ci-dessous les résultats de la recherche, au cas où"
        )
    return None


def _diagnostic_filtre_vide(
    state: dict, args: dict, idx, top: int, rerank: bool, fusion: bool
) -> object:
    """Rejoue la recherche SANS le filtre de type et rend un diagnostic lisible.

    N'est appelé que dans le cas rare où le filtre a tout écarté : le coût d'une
    seconde recherche ne se paie donc jamais en régime normal."""
    sans = idx.search(args["question"], k=top, rerank=rerank, fusion=fusion)
    types = {}
    try:
        magasin, charges = _magasin_pour(state, [args["domaine"]])
        if charges:
            types = magasin.types_disponibles(args["domaine"])
    except (OSError, ValueError):
        pass  # le diagnostic est un service rendu, jamais une raison d'échouer
    return {
        "resultats": [],
        "filtre_ecarte_tout": {
            "type_demande": args.get("type"),
            "sans_ce_filtre": len(sans),
            "premier_sans_filtre": sans[0]["id"] if sans else None,
            "types_reels_du_domaine": types,
            "conseil": (
                "le type demandé n'existe pas dans ce domaine ou ne correspond à "
                "aucun résultat pertinent — relancer sans `type`, ou avec l'un des "
                "types réellement présents ci-dessus"
            ),
        },
    }


def _call_mosaic_search(state: dict, args: dict) -> object:
    _require(args, "question", "domaine")
    top = int(args.get("top", 5))
    if top < 1:
        raise ValueError(f"top doit être >= 1 : {top!r}")
    fusion = bool(args.get("fusion", False))
    grammatical = bool(args.get("grammatical", False))
    # rerank par défaut, SAUF si un mode exclusif est demandé (le moteur refuse les
    # combinaisons — ici on résout le défaut, jamais une exclusivité en silence).
    rerank = bool(args.get("rerank", not (fusion or grammatical)))
    type_filtre = args.get("type") or None
    recence = float(args.get("recence", 0.0))
    idx = _get_index(state, args["domaine"])
    # `rerank` vaut True par DÉFAUT (les index de production le portent tous), mais
    # un domaine découvert dynamiquement peut avoir été construit sans. Le moteur
    # refuse alors — c'est la doctrine, jamais de dégradation silencieuse : un agent
    # qui croit avoir un repêcheur qu'il n'a pas tirerait de fausses conclusions.
    # Ce qui manquait, c'est le geste de SORTIE : le message parlait de reconstruire
    # l'index, alors que l'appelant peut simplement réessayer sans le drapeau.
    try:
        hits = idx.search(
            args["question"],
            k=top,
            rerank=rerank,
            type_filtre=type_filtre,
            recence=recence,
            fusion=fusion,
            grammatical=grammatical,
        )
        conseil = _conseil_aiguillage(args["question"])
        if conseil and hits:
            # Les résultats sont rendus TELS QUELS : le conseil s'ajoute, il ne
            # remplace pas. L'aiguilleur est resté en diagnostic à son banc
            # (3,32 % de fausses alarmes) — à ce taux, conseiller est utile,
            # router serait dangereux.
            return {"resultats": hits, "conseil": conseil}
        if type_filtre and not hits:
            # Un filtre qui écarte TOUT est le pire mode de défaillance pour un
            # agent : il reçoit une liste vide et conclut « ce document n'existe
            # pas » alors qu'il l'a lui-même exclu. On lui rend donc ce que la
            # recherche aurait donné sans le filtre, et le vocabulaire RÉEL du
            # domaine — de quoi se corriger en un tour, sans deviner.
            return _diagnostic_filtre_vide(state, args, idx, top, rerank, fusion)
        return hits
    except ValueError as e:
        if rerank and "rerank.msrv" in str(e) and "rerank" not in args:
            raise ValueError(
                f"le domaine '{args['domaine']}' n'a pas de vecteurs de repêchage "
                "(index construit sans --rerank-vectors) alors que rerank est actif "
                "par défaut — relancer la même question avec rerank=false, ou "
                "reconstruire l'index avec --rerank-vectors pour un meilleur classement"
            ) from e
        raise


def _call_mosaic_diff(state: dict, args: dict) -> object:
    from mosaic.diff import diff_indexes

    _require(args, "domaine_avant", "domaine_apres")
    top = int(args.get("top", 20))
    if top < 1:
        raise ValueError(f"top doit être >= 1 : {top!r}")
    ia = _get_index(state, args["domaine_avant"])
    ib = _get_index(state, args["domaine_apres"])
    return diff_indexes(ia, ib, top=top)


def _call_mosaic_explain(state: dict, args: dict) -> object:
    _require(args, "doc_id", "domaine")
    idx = _get_index(state, args["domaine"])
    question = args.get("question")
    if question:
        return idx.explain_match(question, args["doc_id"])
    return idx.explain(args["doc_id"])


def _call_mosaic_like(state: dict, args: dict) -> object:
    _require(args, "document", "domaine")
    top = int(args.get("top", 5))
    if top < 1:
        raise ValueError(f"top doit être >= 1 : {top!r}")
    idx = _get_index(state, args["domaine"])
    return idx.search_like(args["document"], k=top)


def _croyance_store(state: dict) -> Path:
    return Path(state["data_dir"]) / "croyance.jsonl"


def _call_croyance_assert(state: dict, args: dict) -> object:
    _require(args, "entite", "attribut", "valeur")
    store = _croyance_store(state)
    mem = MemoireCroyance.charger(store) if store.exists() else MemoireCroyance()
    mem.asserter(args["entite"], args["attribut"], str(args["valeur"]), t=args.get("t"))
    mem.sauver(store)
    return {
        "ok": True,
        "entite": args["entite"],
        "attribut": args["attribut"],
        "valeur": args["valeur"],
    }


def _call_croyance_courant(state: dict, args: dict) -> object:
    _require(args, "entite", "attribut")
    store = _croyance_store(state)
    if not store.exists():
        raise ValueError("aucune croyance enregistrée")
    res = MemoireCroyance.charger(store).courant(args["entite"], args["attribut"])
    if res is None:
        raise ValueError(f"emplacement inconnu : {args['entite']}/{args['attribut']}")
    return res


def _call_croyance_historique(state: dict, args: dict) -> object:
    _require(args, "entite", "attribut")
    store = _croyance_store(state)
    if not store.exists():
        return []
    return MemoireCroyance.charger(store).historique(args["entite"], args["attribut"])


def _call_mosaic_meta(state: dict, args: dict) -> object:
    from mosaic.meta import resume_par_index, rrf_fuse

    _require(args, "question", "domaines")
    domaines = args["domaines"]
    if not isinstance(domaines, list) or len(domaines) < 2:
        raise ValueError("domaines : liste de 2+ domaines attendue")
    top = int(args.get("top", 8))
    listes = []
    for d in domaines:
        idx = _get_index(state, d)
        listes.append((d, idx.search(args["question"], k=max(top * 2, 20))))
    return {
        "resultats": rrf_fuse(listes, k=top),
        "resume": resume_par_index(listes),
    }


def _call_mosaic_actuel(state: dict, args: dict) -> object:
    from mosaic.temporel import versions_actuelles

    _require(args, "question", "domaine")
    idx = _get_index(state, args["domaine"])
    return versions_actuelles(idx, args["question"], k=int(args.get("top", 8)))


def _call_mosaic_chemin(state: dict, args: dict) -> object:
    _require(args, "doc_id", "domaine")
    idx = _get_index(state, args["domaine"])
    return idx.chemin(
        args["doc_id"], k=int(args.get("top", 8)), role=args.get("role") or None
    )


def _call_mosaic_stats(state: dict, args: dict) -> object:
    _require(args, "domaine")
    return _get_index(state, args["domaine"]).stats()


def _magasin_pour(state: dict, domaines: list[str]):
    """Magasin structurel dérivé des facettes, MIS EN CACHE par domaine.

    Le peuplement était payé à chaque appel : mesuré 58 ms pour un domaine mais
    **630 ms pour `mosaic_refs` sur six domaines** — douze fois la promesse de
    latence du serveur. Le magasin suit donc le même motif que les Index :
    conservé entre les appels, re-peuplé pour le seul domaine dont la mtime de
    `facettes.json` a changé (un rebuild nocturne la fait bouger). Un compte
    périmé reste impossible, sans repayer la lecture à chaque question.
    Mesuré après : 2,4 ms pour la jointure, 0,3 ms pour un comptage."""
    from mosaic.structurel import Magasin

    m = state.get("magasin")
    if m is None:
        m = state["magasin"] = Magasin()
    mtimes = state.setdefault("magasin_mtimes", {})
    charges = []
    for d in domaines:
        facettes = Path(state["data_dir"]) / f"index_{d}" / "facettes.json"
        try:
            mtime = facettes.stat().st_mtime
        except OSError:
            continue  # domaine sans facettes (index d'avant v1.5) : ignoré, pas fatal
        if mtimes.get(d) != mtime:
            m.charger_depuis(d, json.loads(facettes.read_text(encoding="utf-8")))
            mtimes[d] = mtime
        charges.append(d)
    return m, charges


def _call_mosaic_compter(state: dict, args: dict) -> object:
    _require(args, "domaine")
    d = args["domaine"]
    m, charges = _magasin_pour(state, [d])
    if not charges:
        raise ValueError(
            f"domaine {d!r} sans facettes.json — reconstruire l'index pour compter"
        )
    sortie = {
        "total": m.compter(
            d,
            chemin_contient=args.get("chemin", ""),
            type_doc=args.get("type", ""),
            date_prefixe=args.get("date", ""),
        ),
        "sans_date": m.sans_date(d),
    }
    if args.get("par_mois"):
        sortie["par_mois"] = m.repartition_par_mois(
            d, chemin_contient=args.get("chemin", ""), type_doc=args.get("type", "")
        )
    return sortie


def _call_mosaic_recents(state: dict, args: dict) -> object:
    _require(args, "domaine")
    d = args["domaine"]
    m, charges = _magasin_pour(state, [d])
    if not charges:
        raise ValueError(f"domaine {d!r} sans facettes.json — reconstruire l'index")
    return [
        {"id": doc, "date": date}
        for doc, date in m.plus_recents(
            d,
            k=int(args.get("k", 5)),
            chemin_contient=args.get("chemin", ""),
            type_doc=args.get("type", ""),
        )
    ]


def _call_mosaic_refs(state: dict, args: dict) -> object:
    _require(args, "ref")
    domaines = args.get("domaines") or domaines_disponibles(Path(state["data_dir"]))
    m, charges = _magasin_pour(state, domaines)
    return {
        "domaines_interroges": charges,
        "documents": [
            {"domaine": idx, "id": doc, "date": date}
            for idx, doc, date in m.documents_portant_ref(args["ref"])
        ],
    }


_TOOL_HANDLERS = {
    "mosaic_search": _call_mosaic_search,
    "mosaic_compter": _call_mosaic_compter,
    "mosaic_recents": _call_mosaic_recents,
    "mosaic_refs": _call_mosaic_refs,
    "mosaic_explain": _call_mosaic_explain,
    "mosaic_like": _call_mosaic_like,
    "mosaic_croyance_assert": _call_croyance_assert,
    "mosaic_croyance_courant": _call_croyance_courant,
    "mosaic_croyance_historique": _call_croyance_historique,
    "mosaic_meta": _call_mosaic_meta,
    "mosaic_actuel": _call_mosaic_actuel,
    "mosaic_chemin": _call_mosaic_chemin,
    "mosaic_stats": _call_mosaic_stats,
    "mosaic_diff": _call_mosaic_diff,
}


def _rpc_error(req_id: object, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _rpc_result(req_id: object, result: object) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _tool_error_result(message: str) -> dict:
    """Erreur d'EXÉCUTION d'un outil connu (domaine inconnu, index absent sur disque,
    paramètre manquant, doc_id introuvable...) : convention MCP — reportée DANS le résultat
    (`isError: true`, texte lisible par le modèle appelant), jamais comme erreur JSON-RPC
    protocole. Une erreur JSON-RPC protocole (-32601/-32602) reste réservée à une méthode ou
    un NOM d'outil qui n'existe pas — ça, le modèle appelant ne peut pas le corriger en
    ajustant ses arguments."""
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _tool_ok_result(data: object) -> dict:
    return {
        "content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}],
        "isError": False,
    }


def handle_request(request: dict, state: dict) -> dict | None:
    """Dispatch pur JSON-RPC 2.0 pour les 4 méthodes MCP nécessaires au handshake minimal +
    aux outils. Ni lecture stdin ni écriture stdout ici — testable directement. `None` en
    retour = aucune réponse à émettre (notification JSON-RPC : pas de membre `id` dans la
    requête, ex. `notifications/initialized` envoyé par le client après `initialize`)."""
    method = request.get("method")
    is_notification = "id" not in request
    req_id = request.get("id")

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "capabilities": {"tools": {}},
        }
        return None if is_notification else _rpc_result(req_id, result)

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return None if is_notification else _rpc_result(req_id, {"tools": TOOLS})

    if method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        handler = _TOOL_HANDLERS.get(name) if isinstance(name, str) else None
        if handler is None:
            return (
                None
                if is_notification
                else _rpc_error(req_id, -32602, f"outil inconnu : {name!r}")
            )
        try:
            data = handler(state, args)
        except Exception as exc:  # erreur d'exécution outil, jamais un crash du serveur
            return (
                None
                if is_notification
                else _rpc_result(req_id, _tool_error_result(str(exc)))
            )
        return None if is_notification else _rpc_result(req_id, _tool_ok_result(data))

    return (
        None
        if is_notification
        else _rpc_error(req_id, -32601, f"méthode inconnue : {method!r}")
    )


def run_stdio(state: dict | None = None) -> None:
    """Boucle d'I/O réelle : une requête JSON par ligne sur stdin, une réponse JSON par ligne
    sur stdout (transport MCP stdio). Logs exclusivement sur stderr — jamais de print() nu
    hors ce module vers stdout."""
    if state is None:
        state = new_state()
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # type: ignore
    print(
        f"{SERVER_NAME} v{SERVER_VERSION} démarré — data_dir={state['data_dir']} "
        "(index ouverts à la demande, un seul chargement par domaine)",
        file=sys.stderr,
        flush=True,
    )
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"JSON invalide ignoré : {exc}", file=sys.stderr, flush=True)
            print(
                json.dumps(_rpc_error(None, -32700, "parse error"), ensure_ascii=False),
                flush=True,
            )
            continue
        try:
            response = handle_request(request, state)
        except Exception as exc:  # jamais crasher le serveur sur une requête inattendue
            print(
                f"erreur interne sur la requête {request!r} : {exc}",
                file=sys.stderr,
                flush=True,
            )
            response = _rpc_error(request.get("id"), -32603, f"erreur interne : {exc}")
        if response is not None:
            print(json.dumps(response, ensure_ascii=False), flush=True)
    print(f"{SERVER_NAME} : stdin fermé, arrêt.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    run_stdio()
