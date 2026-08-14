import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from bm25 import BM25


def test_bm25_prefere_le_terme_rare():
    docs = [
        ["cable", "cuivre", "section"],
        ["cable", "gaine", "icta"],
        ["carrelage", "colle", "joint"],
    ]
    top = BM25(docs).search(["cuivre"], k=1)
    assert top == [0]


def test_bm25_terme_absent():
    docs = [["a", "b"], ["c", "d"]]
    assert BM25(docs).search(["zzz"], k=2) == []


def test_run_bench_avec_profile_weighting_et_smoothing_rank(tmp_path):
    """Test run_bench.py with --profile-weighting ppmi and --smoothing-rank 2."""
    # Create minimal corpus
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "un.md").write_text(
        "interrupteur différentiel protection tableau", encoding="utf-8"
    )
    (c / "deux.md").write_text("carrelage colle joint sol cuisine", encoding="utf-8")

    # Create minimal queries
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query": "interrupteur", "relevant": ["un.md"]})
        + "\n"
        + json.dumps({"query": "carrelage", "relevant": ["deux.md"]})
        + "\n",
        encoding="utf-8",
    )

    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "bench" / "run_bench.py"),
            str(c),
            str(queries),
            "--profile-weighting",
            "ppmi",
            "--smoothing-rank",
            "2",
            "--config",
            "test_ppmi",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert "mosaic" in report and "bm25" in report
    assert report["config"] == "test_ppmi"
    assert "recall@10" in report["mosaic"]


def test_run_bench_corpus_mixte_refuse(tmp_path):
    """Mosaic ingère pdf/docx/xlsx/html mais BM25 n'est nourri que de .md/.txt : comparer
    les deux sur un corpus mixte revient à comparer des univers différents. run_bench.py
    doit refuser (exit 2 + erreur JSON sur stderr) plutôt que produire un verdict biaisé."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "un.md").write_text(
        "interrupteur différentiel protection tableau", encoding="utf-8"
    )
    (c / "devis.pdf").write_bytes(
        b"%PDF-1.4\npeu importe, jamais parse : le scan s'arrete a l'extension\n%%EOF"
    )

    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query": "interrupteur", "relevant": ["un.md"]}) + "\n",
        encoding="utf-8",
    )

    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "bench" / "run_bench.py"),
            str(c),
            str(queries),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 2
    err = json.loads(r.stderr)
    assert "error" in err
    assert "corpus mixte" in err["error"]
    assert "1 fichiers convertibles" in err["error"]


def test_run_bench_corpus_mixte_refuse_avec_image(tmp_path):
    """v1.6 §B : IMAGE_EXTS n'est jamais fusionné dans CONVERTIBLE_EXTS, mais la garde
    corpus mixte de run_bench doit quand même les détecter — une photo dans le corpus
    fausse la comparaison Mosaic/BM25 exactement comme un PDF."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "un.md").write_text(
        "interrupteur différentiel protection tableau", encoding="utf-8"
    )
    (c / "plaque.jpg").write_bytes(
        b"\xff\xd8\xff\xe0 peu importe le contenu binaire jpeg"
    )

    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query": "interrupteur", "relevant": ["un.md"]}) + "\n",
        encoding="utf-8",
    )

    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "bench" / "run_bench.py"),
            str(c),
            str(queries),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 2
    err = json.loads(r.stderr)
    assert "error" in err
    assert "corpus mixte" in err["error"]
    assert "1 fichiers convertibles" in err["error"]


def test_run_bench_corpus_mixte_ignore_les_repertoires_exclus(tmp_path):
    """Revue finale v1.6 (Important, #8) : la garde corpus mixte de run_bench scannait TOUT
    le corpus, sans honorer EXCLUDED_DIRS (_MOSAIC/_backups/_corbeille/_cimetiere/
    poubelleClaude) — Index.build() les exclut pourtant déjà à la construction. Un
    convertible/une image posé dans un de ces dossiers (backup, corbeille…) ne fait donc
    jamais partie du corpus réellement indexé par Mosaic : le refuser serait un faux
    positif, pas une vraie comparaison biaisée."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "un.md").write_text(
        "interrupteur différentiel protection tableau", encoding="utf-8"
    )
    piege = c / "_backups"
    piege.mkdir()
    (piege / "vieux_devis.pdf").write_bytes(
        b"%PDF-1.4\njamais indexe, jamais compare\n%%EOF"
    )

    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query": "interrupteur", "relevant": ["un.md"]}) + "\n",
        encoding="utf-8",
    )

    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "bench" / "run_bench.py"),
            str(c),
            str(queries),
            "--config",
            "test_excluded_dirs",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert "mosaic" in report and "bm25" in report


def test_run_bench_rerank_lambda_hors_bornes_refuse_avant_le_build(tmp_path):
    """--rerank-lambda hors [0, 1] doit être rejeté (même contrat de validation que
    cli.py._parse_rerank_lambda), et AVANT le build (échec rapide, pas de coût gaspillé) —
    aucun _bench_index_* ne doit apparaître."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "un.md").write_text(
        "interrupteur différentiel protection tableau", encoding="utf-8"
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query": "interrupteur", "relevant": ["un.md"]}) + "\n",
        encoding="utf-8",
    )

    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "bench" / "run_bench.py"),
            str(c),
            str(queries),
            "--rerank",
            "--rerank-lambda",
            "1.5",
            "--config",
            "test_rerank_bounds",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode != 0
    assert "--rerank-lambda" in r.stderr
    assert not list(tmp_path.glob("_bench_index_test_rerank_bounds*"))


def test_run_bench_rerank_depth_hors_bornes_refuse(tmp_path):
    """--rerank-depth < 10 doit être rejeté (même contrat que cli.py._parse_rerank_depth)."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "un.md").write_text(
        "interrupteur différentiel protection tableau", encoding="utf-8"
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query": "interrupteur", "relevant": ["un.md"]}) + "\n",
        encoding="utf-8",
    )

    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "bench" / "run_bench.py"),
            str(c),
            str(queries),
            "--rerank",
            "--rerank-depth",
            "5",
            "--config",
            "test_rerank_depth_bounds",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode != 0
    assert "--rerank-depth" in r.stderr


def test_run_bench_avec_embeddings_et_abtt(tmp_path):
    """Test run_bench.py with --embeddings and --abtt 1."""
    # Create minimal corpus
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "un.md").write_text(
        "interrupteur différentiel protection tableau", encoding="utf-8"
    )
    (c / "deux.md").write_text("carrelage colle joint sol cuisine", encoding="utf-8")

    # Create minimal queries
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query": "interrupteur", "relevant": ["un.md"]}) + "\n",
        encoding="utf-8",
    )

    # Create minimal embeddings table
    lines = ["1 300"]
    rng = np.random.default_rng(1)
    lines.append("interrupteur " + " ".join(f"{v:.4f}" for v in rng.normal(size=300)))
    vec_gz = tmp_path / "mini.vec.gz"
    with gzip.open(vec_gz, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Prepare embeddings table
    table = tmp_path / "table.msee"
    r_prep = subprocess.run(
        [
            sys.executable,
            "-m",
            "mosaic.cli",
            "embed-prepare",
            str(vec_gz),
            "-o",
            str(table),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r_prep.returncode == 0, r_prep.stderr

    # Run bench with embeddings and abtt
    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "bench" / "run_bench.py"),
            str(c),
            str(queries),
            "--embeddings",
            str(table),
            "--abtt",
            "1",
            "--config",
            "test_embeddings_abtt",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert "mosaic" in report and "bm25" in report
    assert report["config"] == "test_embeddings_abtt"
    assert "recall@10" in report["mosaic"]


def _corpus_et_queries_minimal(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "un.md").write_text(
        "interrupteur différentiel protection tableau", encoding="utf-8"
    )
    (c / "deux.md").write_text("carrelage colle joint sol cuisine", encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query": "interrupteur", "relevant": ["un.md"], "type": "lexical"})
        + "\n",
        encoding="utf-8",
    )
    return c, queries


def test_run_bench_avec_echecs_ajoute_les_requetes_reelles(tmp_path):
    """v1.6 §F : --avec-echecs charge echecs_reels.jsonl (chemin alternatif ici via
    --echecs-path) et les fait apparaître comme un type "reel" à part du rapport."""
    c, queries = _corpus_et_queries_minimal(tmp_path)

    echecs = tmp_path / "echecs.jsonl"
    echecs.write_text(
        json.dumps(
            {
                "query": "carrelage cuisine",
                "attendu": ["deux.md"],
                "constate": "rien",
                "date": "2026-08-10",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "bench" / "run_bench.py"),
            str(c),
            str(queries),
            "--avec-echecs",
            "--echecs-path",
            str(echecs),
            "--config",
            "test_avec_echecs",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert "reel" in report["mosaic"]
    assert "reel" in report["bm25"]


def test_run_bench_avec_echecs_absent_ne_change_rien(tmp_path):
    """Fichier d'échecs inexistant : --avec-echecs ne doit rien ajouter, rien casser."""
    c, queries = _corpus_et_queries_minimal(tmp_path)

    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "bench" / "run_bench.py"),
            str(c),
            str(queries),
            "--avec-echecs",
            "--echecs-path",
            str(tmp_path / "n_existe_pas.jsonl"),
            "--config",
            "test_avec_echecs_absent",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert "reel" not in report["mosaic"]


def test_run_bench_avec_echecs_vide_ne_change_rien(tmp_path):
    """Fichier d'échecs présent mais vide : silencieux, comme absent (contrat de la spec :
    "si le fichier existe et est non vide")."""
    c, queries = _corpus_et_queries_minimal(tmp_path)
    echecs = tmp_path / "vide.jsonl"
    echecs.write_text("", encoding="utf-8")

    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "bench" / "run_bench.py"),
            str(c),
            str(queries),
            "--avec-echecs",
            "--echecs-path",
            str(echecs),
            "--config",
            "test_avec_echecs_vide",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert "reel" not in report["mosaic"]


def test_run_bench_sans_avec_echecs_ignore_le_fichier(tmp_path):
    """Sans --avec-echecs, même un echecs_reels.jsonl valide et non vide n'est jamais lu."""
    c, queries = _corpus_et_queries_minimal(tmp_path)
    echecs = tmp_path / "echecs.jsonl"
    echecs.write_text(
        json.dumps({"query": "carrelage cuisine", "attendu": ["deux.md"]}) + "\n",
        encoding="utf-8",
    )

    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "bench" / "run_bench.py"),
            str(c),
            str(queries),
            "--config",
            "test_sans_avec_echecs",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert "reel" not in report["mosaic"]


def test_run_bench_avec_echecs_ligne_malformee_ignoree_avec_avertissement(tmp_path):
    """v1.6 §F : une ligne malformée (JSON invalide, ou "attendu" absent/vide) est
    ignorée + avertissement stderr, sans faire planter le banc — la ligne valide du
    même fichier est quand même prise en compte."""
    c, queries = _corpus_et_queries_minimal(tmp_path)

    echecs = tmp_path / "echecs.jsonl"
    echecs.write_text(
        "{ceci n'est pas du JSON valide\n"
        + json.dumps({"query": "sans attendu"})
        + "\n"
        + json.dumps({"query": "attendu vide", "attendu": []})
        + "\n"
        + json.dumps({"query": "carrelage cuisine", "attendu": ["deux.md"]})
        + "\n",
        encoding="utf-8",
    )

    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "bench" / "run_bench.py"),
            str(c),
            str(queries),
            "--avec-echecs",
            "--echecs-path",
            str(echecs),
            "--config",
            "test_avec_echecs_malformees",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    assert "ignorée" in r.stderr
    report = json.loads(r.stdout)
    # 1 seule requête réelle valide sur les 4 lignes -> recall@10 défini (pas de division
    # par zéro, la ligne valide a bien été chargée)
    assert "reel" in report["mosaic"]
    assert report["mosaic"]["reel"]["recall@10"] is not None


def test_charger_echecs_reels_attendu_chaine_promu_en_liste(tmp_path):
    """ "attendu" en chaîne unique (pas liste) est accepté et promu en liste à 1 élément —
    tolérance d'usage : un utilisateur ou un agent qui note un seul document ne doit pas se soucier de
    la syntaxe liste JSON."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    import importlib

    run_bench = importlib.import_module("run_bench")

    echecs = tmp_path / "echecs.jsonl"
    echecs.write_text(
        json.dumps({"query": "q", "attendu": "un_seul.md"}) + "\n", encoding="utf-8"
    )

    resultat = run_bench._charger_echecs_reels(echecs)
    assert resultat == [{"query": "q", "relevant": ["un_seul.md"], "type": "reel"}]
