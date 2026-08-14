"""Tests volet E v1.6 : wrapper MCP `mosaic-local` (infra_mcp/mosaic_mcp.py).

Dispatch JSON-RPC 2.0 écrit à la main (zéro dépendance `mcp`), testé sans stdio via
`handle_request(request, state) -> response | None`. `conftest.py` ajoute `infra_mcp/` au
sys.path (comme `scripts/` pour reconstruire_index.py) — pas de package installé.
"""

import json
import subprocess
import sys
from pathlib import Path

import mosaic_mcp as mcp

from mosaic.index import Index

GRID = (32, 32, 3)
SERVER_SCRIPT = Path(__file__).resolve().parent.parent / "infra_mcp" / "mosaic_mcp.py"


def _corpus_devis(tmp_path: Path) -> Path:
    c = tmp_path / "corpus"
    c.mkdir(parents=True, exist_ok=True)
    docs = {
        "relamping_a.md": "relamping éclairage led remplacement tubes fluorescents entrepôt",
        "relamping_b.md": "relamping éclairage led entrepôt luminaires basse consommation",
        "tableau.md": "mise aux normes tableau électrique disjoncteur différentiel",
    }
    for name, text in docs.items():
        (c / name).write_text(text, encoding="utf-8")
    return c


def _built_state(tmp_path: Path) -> dict:
    """Construit un petit index réel sous tmp_path/index_devis (sans rerank-vectors — les
    tests passent rerank=False) et retourne un state prêt à l'emploi."""
    corpus = _corpus_devis(tmp_path)
    Index.build(corpus, tmp_path / "index_devis", grid=GRID)
    return mcp.new_state(tmp_path)


# -- initialize / notifications / méthode inconnue --------------------------------------


def test_initialize_returns_server_info_and_capabilities():
    state = mcp.new_state(Path("."))
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    resp = mcp.handle_request(req, state)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    result = resp["result"]
    assert result["serverInfo"]["name"] == "mosaic"
    assert result["capabilities"] == {"tools": {}}
    assert "protocolVersion" in result


def test_notification_initialized_returns_none():
    state = mcp.new_state(Path("."))
    req = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
    assert mcp.handle_request(req, state) is None


def test_unknown_method_returns_jsonrpc_error():
    state = mcp.new_state(Path("."))
    req = {"jsonrpc": "2.0", "id": 7, "method": "bogus/method"}
    resp = mcp.handle_request(req, state)
    assert resp["id"] == 7
    assert "error" in resp
    assert resp["error"]["code"] == -32601


def test_tools_list_returns_tool_schemas():
    state = mcp.new_state(Path("."))
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp = mcp.handle_request(req, state)
    tools = resp["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {
        "mosaic_search",
        "mosaic_explain",
        "mosaic_like",
        "mosaic_croyance_assert",
        "mosaic_croyance_courant",
        "mosaic_croyance_historique",
        "mosaic_meta",
        "mosaic_actuel",
        "mosaic_chemin",
        "mosaic_stats",
        "mosaic_diff",
        # magasin structurel (14/08) : compter / ordonner / joindre par réf —
        # ce que la recherche par similarité ne sait structurellement pas faire
        "mosaic_compter",
        "mosaic_recents",
        "mosaic_refs",
    }
    for t in tools:
        assert "inputSchema" in t
        assert "description" in t


def test_tools_call_croyance_round_trip(tmp_path):
    state = mcp.new_state(tmp_path)  # store croyance.jsonl sous tmp_path

    def call(name, args):
        req = {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
        resp = mcp.handle_request(req, state)
        import json as _json

        return _json.loads(resp["result"]["content"][0]["text"])

    base = {"entite": "disjoncteur_roger", "attribut": "etat"}
    call("mosaic_croyance_assert", {**base, "valeur": "defectueux", "t": 0})
    call("mosaic_croyance_assert", {**base, "valeur": "repare", "t": 1})
    assert call("mosaic_croyance_courant", base)["valeur"] == "repare"
    hist = call("mosaic_croyance_historique", base)
    assert [x["valeur"] for x in hist] == ["defectueux", "repare"]


# -- tools/call : round-trips réels sur un petit index -----------------------------------


def test_tools_call_mosaic_search_round_trip(tmp_path):
    state = _built_state(tmp_path)
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "mosaic_search",
            "arguments": {
                "question": "remplacement tubes led entrepôt",
                "domaine": "devis",
                "top": 3,
                "rerank": False,
            },
        },
    }
    resp = mcp.handle_request(req, state)
    result = resp["result"]
    assert result["isError"] is False
    hits = json.loads(result["content"][0]["text"])
    assert hits[0]["id"] == "relamping_a.md"


def test_tools_call_mosaic_explain_default_and_with_question(tmp_path):
    state = _built_state(tmp_path)
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "mosaic_explain",
            "arguments": {"doc_id": "relamping_a.md", "domaine": "devis"},
        },
    }
    resp = mcp.handle_request(req, state)
    tokens = json.loads(resp["result"]["content"][0]["text"])
    assert isinstance(tokens, list) and tokens
    assert {"token", "poids"} <= tokens[0].keys()

    req["id"] = 5
    req["params"]["arguments"]["question"] = "tableau électrique"
    resp2 = mcp.handle_request(req, state)
    tokens2 = json.loads(resp2["result"]["content"][0]["text"])
    assert isinstance(tokens2, list) and tokens2


def test_tools_call_mosaic_like_excludes_source(tmp_path):
    state = _built_state(tmp_path)
    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "mosaic_like",
            "arguments": {"document": "relamping_a.md", "domaine": "devis", "top": 5},
        },
    }
    resp = mcp.handle_request(req, state)
    hits = json.loads(resp["result"]["content"][0]["text"])
    ids = [h["id"] for h in hits]
    assert "relamping_a.md" not in ids
    assert "relamping_b.md" in ids


# -- erreurs métier (isError, PAS une erreur JSON-RPC protocole) -------------------------


def test_tools_call_unknown_domaine_returns_error_result(tmp_path):
    state = _built_state(tmp_path)
    req = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {
            "name": "mosaic_search",
            "arguments": {"question": "x", "domaine": "bogus"},
        },
    }
    resp = mcp.handle_request(req, state)
    assert "error" not in resp
    result = resp["result"]
    assert result["isError"] is True
    assert "domaine" in result["content"][0]["text"]


def test_tools_call_missing_index_on_disk_returns_error_result(tmp_path):
    state = mcp.new_state(tmp_path)  # aucun index construit sous tmp_path
    req = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {
            "name": "mosaic_search",
            "arguments": {"question": "x", "domaine": "compta"},
        },
    }
    resp = mcp.handle_request(req, state)
    result = resp["result"]
    assert result["isError"] is True
    # découverte dynamique : le message nomme le domaine ET liste ce qui existe (rien ici)
    assert "compta" in result["content"][0]["text"]
    assert "disponibles" in result["content"][0]["text"]


# -- erreurs protocole (JSON-RPC error, PAS un résultat isError) -------------------------


def test_tools_call_unknown_tool_returns_jsonrpc_error(tmp_path):
    state = _built_state(tmp_path)
    req = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {"name": "mosaic_frobnicate", "arguments": {}},
    }
    resp = mcp.handle_request(req, state)
    assert "result" not in resp
    assert resp["error"]["code"] == -32602


def test_tools_call_missing_required_param_returns_jsonrpc_error(tmp_path):
    state = _built_state(tmp_path)
    req = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "mosaic_search",
            "arguments": {"question": "x"},
        },  # pas de domaine
    }
    resp = mcp.handle_request(req, state)
    assert (
        "error" not in resp
    )  # rentre dans le handler -> isError, cf. politique ci-dessus
    assert resp["result"]["isError"] is True


# -- nouveaux outils (meta / actuel / chemin / stats) ------------------------------------


def _call(state, name, arguments, req_id=50):
    resp = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        state,
    )
    result = resp["result"]
    data = json.loads(result["content"][0]["text"]) if not result["isError"] else None
    return result, data


def test_tools_call_meta_fusionne_deux_domaines(tmp_path):
    """mosaic_meta fusionne 2 domaines par RRF avec provenance."""
    corpus = _corpus_devis(tmp_path)
    Index.build(corpus, tmp_path / "index_devis", grid=GRID)
    Index.build(corpus, tmp_path / "index_compta", grid=GRID)
    state = mcp.new_state(tmp_path)
    result, data = _call(
        state,
        "mosaic_meta",
        {"question": "relamping led", "domaines": ["devis", "compta"], "top": 4},
    )
    assert not result["isError"]
    assert {r["index"] for r in data["resultats"]} == {"devis", "compta"}
    assert all("score_rrf" in r for r in data["resultats"])
    assert len(data["resume"]) == 2


def test_tools_call_actuel_marque_les_perimees(tmp_path):
    """mosaic_actuel : la version la plus récente est canonique, l'ancienne périmée."""
    c = tmp_path / "corpus"
    c.mkdir(parents=True, exist_ok=True)
    sujet = "relamping éclairage led entrepôt luminaires"
    (c / "2025-01-10_spec.md").write_text(f"{sujet} version initiale", encoding="utf-8")
    (c / "2025-06-20_spec.md").write_text(f"{sujet} version validée", encoding="utf-8")
    Index.build(c, tmp_path / "index_devis", grid=GRID)
    state = mcp.new_state(tmp_path)
    result, data = _call(
        state, "mosaic_actuel", {"question": "relamping led", "domaine": "devis"}
    )
    assert not result["isError"]
    groupe = next(g for g in data if "spec" in g["canonique"])
    assert groupe["canonique"] == "2025-06-20_spec.md"
    assert [p["id"] for p in groupe["perimees"]] == ["2025-01-10_spec.md"]


def test_tools_call_chemin_et_erreur_sans_relations(tmp_path):
    """mosaic_chemin traverse quand l'index a le canal relations ; erreur claire sinon."""
    c = tmp_path / "corpus"
    for chemin, texte in [
        ("RIVIERA/2026/note_a.md", "eclairage led riviera"),
        ("RIVIERA/2026/note_b.md", "reception travaux riviera"),
    ]:
        f = c / chemin
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(texte, encoding="utf-8")
    Index.build(c, tmp_path / "index_devis", grid=GRID, relations=True)
    Index.build(c, tmp_path / "index_compta", grid=GRID)  # SANS relations
    state = mcp.new_state(tmp_path)
    result, data = _call(
        state, "mosaic_chemin", {"doc_id": "RIVIERA/2026/note_a.md", "domaine": "devis"}
    )
    assert not result["isError"]
    dossier = next(g for g in data if g["role"] == "dossier")
    assert {d["id"] for d in dossier["documents"]} == {"RIVIERA/2026/note_b.md"}
    # index sans relations : erreur d'exécution lisible, pas un crash protocole
    result2, _ = _call(
        state,
        "mosaic_chemin",
        {"doc_id": "RIVIERA/2026/note_a.md", "domaine": "compta"},
    )
    assert result2["isError"]
    assert "relations" in result2["content"][0]["text"]


def test_tools_call_stats_expose_le_profil(tmp_path):
    """mosaic_stats rend la carte d'identité du domaine, profil inclus (découverte agent)."""
    corpus = _corpus_devis(tmp_path)
    Index.build(
        corpus,
        tmp_path / "index_devis",
        grid=GRID,
        profil={"nom": "test", "refs": {"min_mixte": 4, "min_chiffres": 4}},
    )
    state = mcp.new_state(tmp_path)
    result, data = _call(state, "mosaic_stats", {"domaine": "devis"})
    assert not result["isError"]
    assert data["docs"] == 3
    assert data["profil"]["nom"] == "test"


# -- cache d'Index ouverts (le point de la spec : réutiliser, pas rouvrir) ---------------


def test_index_cache_reused_across_calls(tmp_path):
    state = _built_state(tmp_path)
    req = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {
            "name": "mosaic_search",
            "arguments": {"question": "led", "domaine": "devis", "rerank": False},
        },
    }
    mcp.handle_request(req, state)
    assert "devis" in state["cache"]
    first = state["cache"]["devis"]
    req["id"] = 13
    mcp.handle_request(req, state)
    assert state["cache"]["devis"] is first


def test_index_cache_reopens_when_vocab_mtime_changes(tmp_path):
    """Revue finale v1.6 (Critical) : un rebuild complet de l'index pendant que le serveur
    tourne (tâche planifiée `LOCAL_Mosaic_Rebuild`) doit être visible à l'appel suivant —
    sans redémarrer le process. Sans reprise sur mtime, le cache servirait indéfiniment
    l'ancien contenu (résultats vieux d'une semaine, cf. revue).

    `Index.open()` (`lazy=True` par défaut) garde `vocab.msev` ouvert en `np.memmap` —
    sur Windows, tant que CE memmap est actif, `os.replace()` du rebuild échoue proprement
    (Fix 1a : jamais de corruption, cf. `mosaic.store._write`) plutôt que de réussir en
    concurrence. Ce test libère explicitement ce memmap avant le rebuild (équivalent d'un
    process MCP relancé entre deux runs planifiés, ou d'un remplacement de fichier mappé qui
    n'est jamais bloquant sous Linux/prod) — il porte sur la reprise sur mtime UNE FOIS le
    remplacement effectif, pas sur la résolution du verrou Windows lui-même (hors scope)."""
    state = _built_state(tmp_path)
    req = {
        "jsonrpc": "2.0",
        "id": 20,
        "method": "tools/call",
        "params": {
            "name": "mosaic_search",
            "arguments": {
                "question": "led entrepot",
                "domaine": "devis",
                "top": 5,
                "rerank": False,
            },
        },
    }
    resp1 = mcp.handle_request(req, state)
    hits1 = json.loads(resp1["result"]["content"][0]["text"])
    ids1 = {h["id"] for h in hits1}
    assert "led_nouveau.md" not in ids1
    cached_before = state["cache"]["devis"]

    # Libère le memmap tenu sur vocab.msev (cf. docstring) avant le rebuild ci-dessous —
    # sans ça, os.replace() lèverait PermissionError (verrou Windows), pas le sujet de ce test.
    import gc

    cached_before.profiles.acc = None
    gc.collect()

    # Rebuild complet de l'index (comme reconstruire_index.py) avec un document en plus —
    # mtime de vocab.msev explicitement bumpée pour ne jamais dépendre de la granularité
    # de l'horloge du filesystem (flake).
    corpus = tmp_path / "corpus"
    (corpus / "led_nouveau.md").write_text(
        "relamping led entrepot nouveau produit remplacement",
        encoding="utf-8",
    )
    Index.build(corpus, tmp_path / "index_devis", grid=GRID)
    vocab_path = tmp_path / "index_devis" / "vocab.msev"
    import os
    import time

    futur = time.time() + 5
    os.utime(vocab_path, (futur, futur))

    req["id"] = 21
    resp2 = mcp.handle_request(req, state)
    hits2 = json.loads(resp2["result"]["content"][0]["text"])
    ids2 = {h["id"] for h in hits2}
    assert "led_nouveau.md" in ids2
    assert state["cache"]["devis"] is not cached_before


# -- config snippet -----------------------------------------------------------------------


def test_config_snippet_is_valid_json_pointing_at_server_script():
    snippet_path = (
        Path(__file__).resolve().parent.parent
        / "infra_mcp"
        / "claude_desktop_config_snippet.json"
    )
    data = json.loads(snippet_path.read_text(encoding="utf-8"))
    entry = data["mcpServers"]["mosaic"]
    assert entry["args"][-1].replace("\\\\", "\\").endswith("mosaic_mcp.py")


def test_config_snippet_env_pointe_le_modele_potion_local():
    """Revue finale v1.6 (Important, Fix 4) : sans MOSAIC_POTION_MODEL_DIR (chemin absolu —
    le serveur MCP n'est pas nécessairement lancé depuis la racine du dépôt Mosaic, contrairement
    à l'usage documenté de `mosaic search`), chaque process serveur retombe sur le nom
    HuggingFace et retape le hub (~4.4 s, vérifs réseau) au lieu du répertoire local préparé
    par `scripts/prepare_potion.py --save-model` — cf. mosaic.rerank._model_source."""
    snippet_path = (
        Path(__file__).resolve().parent.parent
        / "infra_mcp"
        / "claude_desktop_config_snippet.json"
    )
    data = json.loads(snippet_path.read_text(encoding="utf-8"))
    env = data["mcpServers"]["mosaic"]["env"]
    assert "MOSAIC_POTION_MODEL_DIR" in env
    normalized = env["MOSAIC_POTION_MODEL_DIR"].replace("\\\\", "\\").replace("\\", "/")
    assert normalized.endswith("data_externes/potion_model")
    # Absolu Windows ("C:/…") OU POSIX ("/…"). PurePosixPath.is_absolute() renvoie False
    # pour un chemin à lettre de lecteur sur un runner Linux : on teste sans dépendre de l'OS.
    assert normalized.startswith("/") or normalized[1:3] == ":/", normalized


# -- smoke end-to-end réel sur stdio (subprocess, newline-delimited JSON) ----------------


def test_stdio_smoke_initialize_and_tools_list(tmp_path):
    env = {"MOSAIC_MCP_DATA_DIR": str(tmp_path)}
    import os

    full_env = dict(os.environ)
    full_env.update(env)
    input_lines = (
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        + "\n"
    )
    r = subprocess.run(
        [sys.executable, str(SERVER_SCRIPT)],
        input=input_lines,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=full_env,
        timeout=30,
    )
    lines = [line for line in r.stdout.splitlines() if line.strip()]
    assert len(lines) == 2, r.stderr
    resp1 = json.loads(lines[0])
    resp2 = json.loads(lines[1])
    assert resp1["result"]["serverInfo"]["name"] == "mosaic"
    assert {t["name"] for t in resp2["result"]["tools"]} == {
        "mosaic_search",
        "mosaic_explain",
        "mosaic_like",
        "mosaic_croyance_assert",
        "mosaic_croyance_courant",
        "mosaic_croyance_historique",
        "mosaic_meta",
        "mosaic_actuel",
        "mosaic_chemin",
        "mosaic_stats",
        "mosaic_diff",
        # magasin structurel (14/08) : compter / ordonner / joindre par réf —
        # ce que la recherche par similarité ne sait structurellement pas faire
        "mosaic_compter",
        "mosaic_recents",
        "mosaic_refs",
    }


def test_tools_list_expose_diff_et_search_les_nouveaux_drapeaux(tmp_path):
    """Le serveur suit le moteur : mosaic_diff présent, fusion/grammatical déclarés
    sur mosaic_search (l'audit avait montré un README MCP en retard — le schéma est
    désormais testé)."""
    state = mcp.new_state(tmp_path)
    rep = mcp.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, state)
    outils = {t["name"]: t for t in rep["result"]["tools"]}
    assert "mosaic_diff" in outils
    props = outils["mosaic_search"]["inputSchema"]["properties"]
    assert "fusion" in props and "grammatical" in props


def test_call_diff_domaines_identiques_diff_vide(tmp_path, monkeypatch):
    """Deux domaines pointant le MÊME index : diff strictement vide (la garantie
    contractuelle traverse le serveur)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("interrupteur différentiel tableau", encoding="utf-8")
    (corpus / "b.md").write_text("carrelage colle joint", encoding="utf-8")
    from mosaic.index import Index

    Index.build(corpus, tmp_path / "index_alpha", grid=(32, 32, 3))
    Index.build(corpus, tmp_path / "index_beta", grid=(32, 32, 3))
    state = mcp.new_state(tmp_path)
    rep = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "mosaic_diff",
                "arguments": {"domaine_avant": "alpha", "domaine_apres": "beta"},
            },
        },
        state,
    )
    assert not rep["result"].get("isError"), rep
    donnees = json.loads(rep["result"]["content"][0]["text"])
    assert donnees["docs_ajoutes"] == [] and donnees["derive_mots"] == []


# --- magasin structurel : les outils de comptage répondent VRAIMENT --------------


def test_les_outils_du_magasin_repondent_sur_un_index_reel(tmp_path, monkeypatch):
    """Test de bout en bout des trois outils de comptage, avec le cache.

    Écrit après un incident du 14/08 : une méthode du magasin (`charger_depuis`,
    appelée uniquement par le chemin de cache du serveur) a disparu lors d'une
    copie entre dépôts, et AUCUN test ne l'a vu — la suite entière passait au vert
    pendant que les trois outils rendaient une erreur. Un test qui ne traverse pas
    l'appel réel ne protège pas l'appel réel."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "2026-06-01_note.md").write_text("tableau électrique", encoding="utf-8")
    (corpus / "2026-07-02_note.md").write_text("disjoncteur courbe C", encoding="utf-8")
    racine = tmp_path / "indexes"
    racine.mkdir()
    Index.build(corpus, racine / "index_test", grid=(32, 32, 3), smoothing_rank=0)
    state = mcp.new_state(racine)

    def appeler(nom, args):
        rep = mcp.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": nom, "arguments": args},
            },
            state,
        )
        return rep["result"]

    r = appeler("mosaic_compter", {"domaine": "test"})
    assert not r.get("isError"), r["content"][0]["text"]
    assert json.loads(r["content"][0]["text"])["total"] == 2

    # deuxième appel : c'est le CHEMIN DE CACHE qui était cassé, pas le premier
    r2 = appeler("mosaic_compter", {"domaine": "test", "date": "2026-07"})
    assert not r2.get("isError"), r2["content"][0]["text"]
    assert json.loads(r2["content"][0]["text"])["total"] == 1

    r3 = appeler("mosaic_recents", {"domaine": "test", "k": 1})
    assert not r3.get("isError"), r3["content"][0]["text"]
    assert json.loads(r3["content"][0]["text"])[0]["date"] == "2026-07-02"

    r4 = appeler("mosaic_refs", {"ref": "inexistante", "domaines": ["test"]})
    assert not r4.get("isError"), r4["content"][0]["text"]
    assert json.loads(r4["content"][0]["text"])["documents"] == []


def test_un_filtre_de_type_qui_ecarte_tout_le_dit(tmp_path):
    """Jamais de dégradation silencieuse : si le type demandé vide le résultat,
    l'agent reçoit ce qu'aurait donné la recherche sans lui, et le vocabulaire
    réel du domaine — de quoi se corriger seul plutôt que conclure « ça n'existe
    pas » sur un document qu'il a lui-même exclu."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text(
        "tableau électrique disjoncteur courbe C", encoding="utf-8"
    )
    racine = tmp_path / "indexes"
    racine.mkdir()
    Index.build(corpus, racine / "index_test", grid=(32, 32, 3), smoothing_rank=0)
    state = mcp.new_state(racine)

    rep = mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mosaic_search",
                "arguments": {
                    "question": "tableau électrique",
                    "domaine": "test",
                    "type": "présentation",
                    "rerank": False,
                },
            },
        },
        state,
    )
    assert not rep["result"].get("isError")
    d = json.loads(rep["result"]["content"][0]["text"])["filtre_ecarte_tout"]
    assert d["type_demande"] == "présentation"
    assert d["sans_ce_filtre"] >= 1, "la recherche sans filtre trouvait pourtant"
    assert d["premier_sans_filtre"] == "note.md"
    assert "note texte" in d["types_reels_du_domaine"], d["types_reels_du_domaine"]


def test_une_question_de_comptage_recoit_un_conseil_sans_perdre_ses_resultats(tmp_path):
    """L'aiguilleur CONSEILLE, il ne route pas — son banc l'a laissé en diagnostic
    (3,32 % de fausses alarmes). Un conseil erroné coûte une ligne ignorée ; un
    routage erroné rendrait un nombre là où l'agent attend des documents. L'agent
    apprend donc au moment où il se trompe, pas dans un mode d'emploi."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "bl.md").write_text("bon de livraison fournisseur", encoding="utf-8")
    racine = tmp_path / "indexes"
    racine.mkdir()
    Index.build(corpus, racine / "index_test", grid=(32, 32, 3), smoothing_rank=0)
    state = mcp.new_state(racine)

    def chercher(question):
        rep = mcp.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "mosaic_search",
                    "arguments": {
                        "question": question,
                        "domaine": "test",
                        "rerank": False,
                    },
                },
            },
            state,
        )
        assert not rep["result"].get("isError"), rep["result"]["content"][0]["text"]
        return json.loads(rep["result"]["content"][0]["text"])

    d = chercher("combien de bons de livraison en juin")
    assert d["resultats"], "les résultats sont rendus MALGRÉ le conseil"
    assert "mosaic_compter" in d["conseil"], d["conseil"]

    d2 = chercher("le dernier bon de livraison reçu")
    assert "mosaic_recents" in d2["conseil"], d2["conseil"]

    # une question de sens ne reçoit AUCUN conseil : pas de bruit dans le cas courant
    d3 = chercher("bon de livraison fournisseur")
    assert isinstance(d3, list), "sans conseil, la réponse reste la liste habituelle"
