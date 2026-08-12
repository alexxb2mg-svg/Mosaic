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
(`une tâche planifiée`) la fait changer, et l'appel suivant réouvre l'index plutôt que de
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

# domaine -> la racine des index/index_<domaine>
# Les domaines sont DÉCOUVERTS dynamiquement : tout dossier `index_<domaine>` sous la racine
# des index est un domaine interrogeable (plus de liste en dur — un nouvel index déposé par un
# script ou un rebuild devient accessible sans toucher au serveur). DEFAULT_DATA_DIR reste le
# défaut local ; surchargé par MOSAIC_MCP_DATA_DIR (tests, autre machine).
DEFAULT_DATA_DIR = Path.home() / ".mosaic"


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
            "Recherche sémantique dans un index Mosaic (question en "
            "français, la paraphrase est acceptée, pas besoin des mots exacts). "
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
                    "description": "Canal grammatical (nécessite un index --grammatical) : "
                    "sépare les clauses à mots identiques et sens opposé (négation à portée, "
                    "amont/aval). À activer quand la STRUCTURE de la phrase porte le sens ; "
                    "exclusif avec rerank/fusion/type/recence.",
                },
                "rerank": {
                    "type": "boolean",
                    "default": True,
                    "description": "Repêcheur model2vec — nécessite un index construit avec "
                    "--rerank-vectors (si l'index a été construit avec).",
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
    jamais pointer les tests vers la racine des index, index de production en lecture seule)."""
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
    # sur disque — un rebuild planifié (une tâche planifiée, dimanche 03h) remplace
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
    return idx.search(
        args["question"],
        k=top,
        rerank=rerank,
        type_filtre=type_filtre,
        recence=recence,
        fusion=fusion,
        grammatical=grammatical,
    )


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


_TOOL_HANDLERS = {
    "mosaic_search": _call_mosaic_search,
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
