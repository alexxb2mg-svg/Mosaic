import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

PY = [sys.executable, "-m", "mosaic.cli"]


def _minimal_pdf(text: str) -> bytes:
    """Construit un PDF 1 page minimal et valide (xref calculé), avec `text` en contenu."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 200 100] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream_content = f"BT /F1 24 Tf 10 50 Td ({text}) Tj ET".encode("latin-1")
    objects.append(
        b"<< /Length %d >>\nstream\n" % len(stream_content)
        + stream_content
        + b"\nendstream"
    )

    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(buf))
        buf += f"{i} 0 obj\n".encode()
        buf += body
        buf += b"\nendobj\n"
    xref_offset = len(buf)
    n = len(objects) + 1
    buf += f"xref\n0 {n}\n".encode()
    buf += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        buf += f"{off:010d} 00000 n \n".encode()
    buf += f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    return bytes(buf)


def _run(*args: str, **kwargs):
    return subprocess.run(
        [*PY, *args], capture_output=True, text=True, encoding="utf-8", **kwargs
    )


def _run_bytes(*args: str, env=None):
    """Run CLI with raw byte capture (for UTF-8 tests)."""
    return subprocess.run([*PY, *args], capture_output=True, env=env)


def _erreur_json(stderr: str) -> dict:
    """Décode l'erreur JSON du CLI (dernière ligne de stderr). Robuste au bruit qu'une
    dépendance peut cracher sur stderr AVANT — cas réel : sous Linux, markitdown tire
    magika→onnxruntime, qui émet un avertissement `device_discovery` au chargement. Le
    JSON d'erreur du CLI reste la dernière ligne (json.dumps mono-ligne)."""
    return json.loads(stderr.strip().splitlines()[-1])


def _corpus(tmp_path: Path) -> Path:
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "un.md").write_text(
        "interrupteur différentiel protection tableau", encoding="utf-8"
    )
    (c / "deux.md").write_text("carrelage colle joint sol cuisine", encoding="utf-8")
    return c


def test_build_puis_search(tmp_path):
    idx = str(tmp_path / "idx")
    r = _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2
    r = _run("search", "interrupteur différentiel", idx, "--top", "1")
    hits = json.loads(r.stdout)
    assert hits[0]["id"] == "un.md" and "score" in hits[0]


def test_stats_et_render(tmp_path):
    idx = str(tmp_path / "idx")
    _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")
    assert json.loads(_run("stats", idx).stdout)["docs"] == 2
    png = tmp_path / "m.png"
    r = _run("render", "un.md", idx, "-o", str(png))
    assert r.returncode == 0 and png.read_bytes()[:4] == b"\x89PNG"


def test_erreur_index_absent(tmp_path):
    r = _run("search", "x", str(tmp_path / "nulle_part"))
    assert r.returncode == 1
    assert "error" in json.loads(r.stderr)


def test_utf8_accented_filenames(tmp_path):
    """Verify UTF-8 output even when PYTHONIOENCODING/PYTHONUTF8 are stripped."""
    c = tmp_path / "corpus"
    c.mkdir()
    # Create file with accented name
    (c / "réunion.md").write_text("contenu réunion", encoding="utf-8")
    (c / "deux.md").write_text("autre document", encoding="utf-8")

    idx = str(tmp_path / "idx")

    # Build with env stripped of UTF-8 settings
    env = os.environ.copy()
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)

    r = _run_bytes("build", str(c), "-o", idx, env=env)
    assert r.returncode == 0, r.stderr

    # Search to get doc IDs in output (with env stripped)
    r = _run_bytes("search", "réunion", idx, env=env)
    assert r.returncode == 0, r.stderr
    # Verify UTF-8 bytes are in raw stdout (not mojibake)
    assert "réunion.md".encode("utf-8") in r.stdout


def test_embed_prepare(tmp_path):
    import gzip

    import numpy as np

    lines = ["2 300"]
    for w, seed in (("disjoncteur", 1), ("chat", 2)):
        rng = np.random.default_rng(seed)
        lines.append(w + " " + " ".join(f"{v:.4f}" for v in rng.normal(size=300)))
    src = tmp_path / "mini.vec.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    out = tmp_path / "table.msee"
    r = _run("embed-prepare", str(src), "-o", str(out), "--keep", "200000")
    assert r.returncode == 0, r.stderr
    stats = json.loads(r.stdout)
    assert stats["kept"] >= 1
    assert out.exists() and out.stat().st_size > 0


def test_embed_prepare_abtt(tmp_path):
    """v1.6 §C : --abtt N à embed-prepare écrit une table déjà nettoyée, chargeable au même
    abtt sans re-transformation (cf. tests unitaires mosaic.embeddings pour le détail bas
    niveau — ce test couvre seulement le passage CLI --abtt -> prepare(..., abtt=N))."""
    import gzip

    import numpy as np

    rng = np.random.default_rng(42)
    base = rng.normal(size=300) * 5.0
    words = [f"mot{i}" for i in range(30)]
    lines = ["30 300"]
    for w in words:
        vec = base + rng.normal(size=300) * 0.3
        lines.append(w + " " + " ".join(f"{v:.4f}" for v in vec))
    src = tmp_path / "mini.vec.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    out = tmp_path / "table.msee"
    r = _run("embed-prepare", str(src), "-o", str(out), "--abtt", "2")
    assert r.returncode == 0, r.stderr
    stats = json.loads(r.stdout)
    assert stats["kept"] == 30

    from mosaic.embeddings import Embeddings

    emb = Embeddings.load(out, abtt=2)
    assert emb.abtt == 2
    with pytest.raises(ValueError):
        Embeddings.load(out, abtt=0)


def test_build_avec_embeddings(tmp_path):
    import gzip

    import numpy as np

    lines = ["1 300"]
    rng = np.random.default_rng(1)
    lines.append("interrupteur " + " ".join(f"{v:.4f}" for v in rng.normal(size=300)))
    src = tmp_path / "mini.vec.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    table = tmp_path / "table.msee"
    r = _run("embed-prepare", str(src), "-o", str(table))
    assert r.returncode == 0, r.stderr

    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--embeddings",
        str(table),
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2
    r = _run("search", "interrupteur différentiel", idx, "--top", "1")
    hits = json.loads(r.stdout)
    assert hits[0]["id"] == "un.md" and "score" in hits[0]


def test_explain(tmp_path):
    idx = str(tmp_path / "idx")
    _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")
    r = _run("explain", "un.md", idx, "--top", "5")
    assert r.returncode == 0, r.stderr
    top = json.loads(r.stdout)
    assert isinstance(top, list) and len(top) <= 5
    assert "token" in top[0] and "poids" in top[0]


def test_explain_avec_query(tmp_path):
    idx = str(tmp_path / "idx")
    _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")
    r = _run("explain", "un.md", idx, "--top", "3", "--query", "interrupteur")
    assert r.returncode == 0, r.stderr
    top = json.loads(r.stdout)
    assert isinstance(top, list) and len(top) <= 3
    assert top[0]["token"] == "interrupteur"


def test_explain_doc_inconnu(tmp_path):
    idx = str(tmp_path / "idx")
    _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")
    r = _run("explain", "inconnu.md", idx)
    assert r.returncode == 1
    assert "error" in json.loads(r.stderr)


def test_build_avec_lexicon_extra_chemin(tmp_path):
    extra_path = tmp_path / "extra.json"
    extra_path.write_text(json.dumps({"interrupteur": "toggle"}), encoding="utf-8")
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--lexicon-extra",
        str(extra_path),
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2


def test_build_avec_lexicon_extra_wikdict_shortcut(tmp_path):
    """Le raccourci "wikdict" résout vers le lexique embarqué dans le package."""
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--lexicon-extra",
        "wikdict",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2


def test_build_avec_lexicon_extra_chemin_introuvable(tmp_path):
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--lexicon-extra",
        str(tmp_path / "n_existe_pas.json"),
    )
    assert r.returncode == 1
    assert "error" in json.loads(r.stderr)


def test_build_avec_weights_personnalises(tmp_path):
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--weights",
        "0.35,0.25,0.40",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2


def test_build_avec_weights_invalides_erreur_json(tmp_path):
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--weights",
        "0.1,0.2",  # seulement 2 valeurs au lieu de 3
    )
    assert r.returncode == 1
    err = json.loads(r.stderr)
    assert "error" in err
    assert "--weights" in err["error"]


def test_top_validation_negative(tmp_path):
    """Verify --top < 1 raises error."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "un.md").write_text("test", encoding="utf-8")

    idx = str(tmp_path / "idx")
    _run("build", str(c), "-o", idx, "--grid", "32x32")

    r = _run("search", "test", idx, "--top", "-1")
    assert r.returncode == 1
    assert "error" in json.loads(r.stderr)
    assert "--top" in json.loads(r.stderr)["error"]


def test_build_avec_profile_weighting_brut(tmp_path):
    """Verify --profile-weighting brut works."""
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--profile-weighting",
        "brut",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2
    # Verify search still works
    r = _run("search", "interrupteur", idx, "--top", "1")
    assert r.returncode == 0 and json.loads(r.stdout)[0]["id"] == "un.md"


def test_build_avec_profile_weighting_ppmi(tmp_path):
    """Verify --profile-weighting ppmi works."""
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--profile-weighting",
        "ppmi",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2


def test_build_avec_profile_weighting_invalide(tmp_path):
    """Verify invalid --profile-weighting raises error."""
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--profile-weighting",
        "invalid",
    )
    assert r.returncode == 1
    err = json.loads(r.stderr)
    assert "error" in err


def test_build_avec_smoothing_rank(tmp_path):
    """Verify --smoothing-rank works with non-negative value."""
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--smoothing-rank",
        "5",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2


def test_build_avec_smoothing_rank_zero(tmp_path):
    """Verify --smoothing-rank 0 works."""
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--smoothing-rank",
        "0",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2


def test_build_avec_smoothing_rank_negatif_erreur(tmp_path):
    """Verify negative --smoothing-rank raises error."""
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--smoothing-rank",
        "-1",
    )
    assert r.returncode == 1
    err = json.loads(r.stderr)
    assert "error" in err


def test_build_avec_abtt(tmp_path):
    """Verify --abtt works with non-negative value."""
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--abtt",
        "10",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2


def test_build_avec_abtt_zero(tmp_path):
    """Verify --abtt 0 works."""
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--abtt",
        "0",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2


def test_build_avec_abtt_negatif_erreur(tmp_path):
    """Verify negative --abtt raises error."""
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--abtt",
        "-1",
    )
    assert r.returncode == 1
    err = json.loads(r.stderr)
    assert "error" in err


def test_build_avec_doc_weight(tmp_path):
    """Verify --doc-weight works with a valid value in [0, 1)."""
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--doc-weight",
        "0.25",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2


def test_cli_doc_weight_invalide(tmp_path):
    """--doc-weight en dehors de [0, 1) refuse avec rc 1 + erreur JSON."""
    corpus = str(_corpus(tmp_path))
    idx = str(tmp_path / "idx")
    for val in ("1.5", "-0.1"):
        r = _run(
            "build",
            corpus,
            "-o",
            idx,
            "--grid",
            "32x32",
            "--doc-weight",
            val,
        )
        assert r.returncode == 1
        err = json.loads(r.stderr)
        assert "error" in err
        assert "--doc-weight" in err["error"]


def test_build_avec_rerank_vectors(tmp_path):
    """Intégration réelle (model2vec en cache local dans cet env de dev)."""
    pytest.importorskip("model2vec")
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--rerank-vectors",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2
    assert (Path(idx) / "rerank.msrv").is_file()


def test_search_avec_rerank(tmp_path):
    pytest.importorskip("model2vec")
    idx = str(tmp_path / "idx")
    _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--rerank-vectors",
    )
    r = _run("search", "interrupteur différentiel", idx, "--top", "1", "--rerank")
    assert r.returncode == 0, r.stderr
    hits = json.loads(r.stdout)
    assert "score_rerank" in hits[0] and "score" in hits[0]


def test_search_avec_rerank_lambda_et_depth_personnalises(tmp_path):
    pytest.importorskip("model2vec")
    idx = str(tmp_path / "idx")
    _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--rerank-vectors",
    )
    r = _run(
        "search",
        "interrupteur différentiel",
        idx,
        "--top",
        "1",
        "--rerank",
        "--rerank-lambda",
        "0.5",
        "--rerank-depth",
        "10",
    )
    assert r.returncode == 0, r.stderr


def test_search_rerank_sans_msrv_erreur_json(tmp_path):
    idx = str(tmp_path / "idx")
    _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")
    r = _run("search", "interrupteur", idx, "--rerank")
    assert r.returncode == 1
    assert "error" in json.loads(r.stderr)


def test_search_rerank_lambda_hors_bornes_erreur_json(tmp_path):
    idx = str(tmp_path / "idx")
    _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")
    for val in ("1.5", "-0.1"):
        r = _run("search", "interrupteur", idx, "--rerank-lambda", val)
        assert r.returncode == 1
        err = json.loads(r.stderr)
        assert "error" in err and "--rerank-lambda" in err["error"]


def test_search_rerank_depth_hors_bornes_erreur_json(tmp_path):
    idx = str(tmp_path / "idx")
    _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")
    r = _run("search", "interrupteur", idx, "--rerank-depth", "5")
    assert r.returncode == 1
    err = json.loads(r.stderr)
    assert "error" in err and "--rerank-depth" in err["error"]


def _corpus_nomme(tmp_path: Path) -> Path:
    c = tmp_path / "devis"
    (c / "2026" / "08.2026").mkdir(parents=True)
    (c / "2026" / "08.2026" / "D26089903_KUMQUAT_devis.md").write_text(
        "fourniture pose tableau électrique cablage protection", encoding="utf-8"
    )
    (c / "2026" / "08.2026" / "D26080004_client_devis.md").write_text(
        "fourniture pose interrupteur disjoncteur cablage protection", encoding="utf-8"
    )
    return c


def test_build_index_paths_defaut_on_trouve_par_nom_fichier(tmp_path):
    idx = str(tmp_path / "idx")
    r = _run("build", str(_corpus_nomme(tmp_path)), "-o", idx, "--grid", "32x32")
    assert r.returncode == 0, r.stderr
    r = _run("search", "kumquat", idx, "--top", "1")
    hits = json.loads(r.stdout)
    assert "KUMQUAT" in hits[0]["id"]
    assert hits[0]["score"] > 0.3  # match net (contenu ne mentionne jamais « kumquat »)


def test_build_no_path_tokens_score_bien_plus_faible(tmp_path):
    """--no-path-tokens : « kumquat » reste un token jamais appris (OOV) — seul un
    résidu de bruit de signature de hachage subsiste (voir test_index.py pour la
    même mesure au niveau API : présence dans le top-k n'est pas discriminante sur
    un corpus minuscule, l'écart de score l'est)."""
    corpus = _corpus_nomme(tmp_path)
    idx_on, idx_off = str(tmp_path / "idx_on"), str(tmp_path / "idx_off")
    assert _run("build", str(corpus), "-o", idx_on, "--grid", "32x32").returncode == 0
    r = _run("build", str(corpus), "-o", idx_off, "--grid", "32x32", "--no-path-tokens")
    assert r.returncode == 0, r.stderr

    score_on = next(
        h["score"]
        for h in json.loads(_run("search", "kumquat", idx_on, "--top", "2").stdout)
        if "KUMQUAT" in h["id"]
    )
    score_off = next(
        h["score"]
        for h in json.loads(_run("search", "kumquat", idx_off, "--top", "2").stdout)
        if "KUMQUAT" in h["id"]
    )
    assert score_on > 5 * score_off


def test_build_avec_ocr_texte_deja_suffisant(tmp_path):
    """--ocr accepté par le CLI ; sans document muet, aucun OCR requis, build normal."""
    pytest.importorskip("markitdown")

    idx = str(tmp_path / "idx")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    texte = "pose interrupteur différentiel tableau protection cablage disjoncteur " * 4
    (corpus / "devis.pdf").write_bytes(_minimal_pdf(texte))
    r = _run("build", str(corpus), "-o", idx, "--grid", "32x32", "--ocr")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 1


def test_build_avec_ocr_sans_provider_erreur_json(tmp_path):
    """Sans moteur OCR (simulé par stub d'import dans le sous-processus) : --ocr
    sur un document muet doit échouer avec une erreur JSON claire, rc 1."""
    pytest.importorskip("markitdown")

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for nom in ("rapidocr", "rapidocr_onnxruntime"):
        (stubs / f"{nom}.py").write_text(
            'raise ImportError("stub test")', encoding="utf-8"
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stubs) + os.pathsep + env.get("PYTHONPATH", "")

    idx = str(tmp_path / "idx")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "scan.pdf").write_bytes(_minimal_pdf("bref"))
    r = subprocess.run(
        [*PY, "build", str(corpus), "-o", idx, "--grid", "32x32", "--ocr"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert r.returncode == 1
    assert "error" in _erreur_json(r.stderr)


def test_search_batch_lit_plusieurs_requetes_depuis_stdin(tmp_path):
    idx = str(tmp_path / "idx")
    _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")
    r = subprocess.run(
        [*PY, "search", "--batch", idx],
        input="interrupteur différentiel\ncarrelage cuisine\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    lines = [line for line in r.stdout.splitlines() if line.strip()]
    assert len(lines) == 2
    hits1, hits2 = json.loads(lines[0]), json.loads(lines[1])
    assert hits1[0]["id"] == "un.md"
    assert hits2[0]["id"] == "deux.md"


def test_search_batch_ignore_les_lignes_vides(tmp_path):
    idx = str(tmp_path / "idx")
    _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")
    r = subprocess.run(
        [*PY, "search", "--batch", idx],
        input="interrupteur\n\ncarrelage\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    lines = [line for line in r.stdout.splitlines() if line.strip()]
    assert len(lines) == 2


def test_search_batch_avec_query_positionnelle_erreur_json(tmp_path):
    idx = str(tmp_path / "idx")
    _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")
    r = _run("search", "requete", idx, "--batch")
    assert r.returncode == 1
    assert "error" in json.loads(r.stderr)


def test_search_sans_batch_ni_query_erreur_json(tmp_path):
    idx = str(tmp_path / "idx")
    _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")
    r = _run(
        "search", idx
    )  # un seul positionnel -> consommé par `index`, query manquante
    assert r.returncode == 1
    assert "error" in json.loads(r.stderr)


def test_search_batch_avec_rerank(tmp_path):
    pytest.importorskip("model2vec")
    idx = str(tmp_path / "idx")
    _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--rerank-vectors",
    )
    r = subprocess.run(
        [*PY, "search", "--batch", idx, "--rerank"],
        input="interrupteur différentiel\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    hits = json.loads(r.stdout.splitlines()[0])
    assert "score_rerank" in hits[0]


def test_search_batch_amorti_le_cout_de_demarrage(tmp_path):
    """5 requêtes en 1 process (--batch) doivent être plus rapides que 5 process séparés
    (chacun repaye l'import Python + Index.open) — démontre l'amortissement."""
    idx = str(tmp_path / "idx")
    _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")

    t0 = time.perf_counter()
    for _ in range(5):
        _run("search", "interrupteur", idx, "--top", "1")
    separate = time.perf_counter() - t0

    queries = "\n".join(["interrupteur"] * 5) + "\n"
    t0 = time.perf_counter()
    r = subprocess.run(
        [*PY, "search", "--batch", idx],
        input=queries,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    batch = time.perf_counter() - t0
    assert r.returncode == 0, r.stderr
    assert len([line for line in r.stdout.splitlines() if line.strip()]) == 5
    assert batch < separate


def test_search_batch_utf8(tmp_path):
    """Requêtes accentuées lues depuis stdin en UTF-8 même env dépouillé."""
    idx = str(tmp_path / "idx")
    _run("build", str(_corpus(tmp_path)), "-o", idx, "--grid", "32x32")
    env = os.environ.copy()
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)
    r = subprocess.run(
        [*PY, "search", "--batch", idx],
        input="interrupteur différentiel\n".encode("utf-8"),
        capture_output=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    hits = json.loads(r.stdout.decode("utf-8").splitlines()[0])
    assert hits[0]["id"] == "un.md"


def test_build_avec_tous_les_drapeaux_nouveaux(tmp_path):
    """Verify all three new parameters work together."""
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--profile-weighting",
        "ppmi",
        "--smoothing-rank",
        "3",
        "--abtt",
        "5",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["docs"] == 2
    # Verify search works and index persists settings
    r = _run("search", "interrupteur", idx, "--top", "1")
    assert r.returncode == 0 and json.loads(r.stdout)[0]["id"] == "un.md"


# -- revue finale v1.6 (Important, #3/#6) : aide CLI ------------------------------------------


def test_build_ocr_help_mentionne_les_images():
    """`mosaic build --ocr` s'applique aussi aux photos/images (v1.6 §B, IMAGE_EXTS) — le
    texte d'aide se limitait aux convertibles muets, laissant croire que --ocr est sans
    effet sur .jpg/.jpeg/.png/.tiff (alors qu'il en est la SEULE voie d'ingestion)."""
    r = _run("build", "--help")
    assert r.returncode == 0, r.stderr
    assert "image" in r.stdout.lower()


def test_build_cache_ingestion_help_mentionne_la_variable_env():
    r = _run("build", "--help")
    assert r.returncode == 0, r.stderr
    assert "MOSAIC_INGEST_CACHE" in r.stdout


def test_carte_cache_ingestion_help_mentionne_la_variable_env():
    r = _run("carte", "--help")
    assert r.returncode == 0, r.stderr
    assert "MOSAIC_INGEST_CACHE" in r.stdout


# -- Gardes de la refonte (audit CLI 12/08) — chaque piège corrigé a son test ---------------


def test_build_corpus_inexistant_refuse(tmp_path):
    r = _run("build", str(tmp_path / "nulle_part"), "-o", str(tmp_path / "idx"))
    assert r.returncode == 1
    assert "corpus introuvable" in _erreur_json(r.stderr)["error"]


def test_build_corpus_vide_refuse_et_ne_laisse_rien(tmp_path):
    """Le pire échec silencieux : un index VIDE livré rc 0 rendrait [] pour toujours.
    Refus loud, et l'artefact n'est pas conservé."""
    vide = tmp_path / "vide"
    vide.mkdir()
    sortie = tmp_path / "idx"
    r = _run("build", str(vide), "-o", str(sortie))
    assert r.returncode == 1
    assert "aucun document indexable" in _erreur_json(r.stderr)["error"]
    assert not sortie.exists(), "l'index vide ne doit pas être conservé"


def test_search_cible_non_index_message_actionnable(tmp_path):
    """Une typo d'ordre positionnel (requête prise pour un index) doit se lire
    immédiatement — plus jamais un errno brut sur docs.msei."""
    r = _run("search", "interrupteur différentiel", str(tmp_path / "absent"))
    assert r.returncode == 1
    err = _erreur_json(r.stderr)["error"]
    assert "n'est pas un index Mosaic" in err and "mosaic build" in err


def test_like_index_oublie_message_actionnable(tmp_path):
    """`like a.md b.md` (index oublié) prenait silencieusement b.md pour l'index."""
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("alpha", encoding="utf-8")
    b.write_text("beta", encoding="utf-8")
    r = _run("like", str(a), str(b))
    assert r.returncode == 1
    assert "n'est pas un index Mosaic" in _erreur_json(r.stderr)["error"]


def test_conn_lambda_sans_connecteurs_refuse(tmp_path):
    r = _run("search", "q", str(tmp_path / "idx"), "--conn-lambda", "0.5")
    assert r.returncode == 1
    assert "--conn-lambda" in _erreur_json(r.stderr)["error"]


def test_fusion_et_rerank_refuses_avant_toute_io(tmp_path):
    """L'exclusivité doit tomber AVANT l'ouverture : l'index n'existe pas, et c'est
    bien l'exclusivité (pas l'index absent) qui doit répondre."""
    r = _run("search", "q", str(tmp_path / "absent"), "--fusion", "--rerank")
    assert r.returncode == 1
    assert "exclusifs" in _erreur_json(r.stderr)["error"]


def test_calibrer_requetes_et_verite_auto_exclusifs(tmp_path):
    r = _run("calibrer", str(tmp_path), "--requetes", "x.jsonl", "--verite-auto")
    assert r.returncode == 1
    assert "exclusifs" in _erreur_json(r.stderr)["error"]


def test_profil_langue_sans_explique_refuse(tmp_path):
    r = _run("profil", str(tmp_path), "--langue", "en")
    assert r.returncode == 1
    assert "--langue" in _erreur_json(r.stderr)["error"]


def test_croyance_store_inexistant_refuse_en_lecture(tmp_path):
    """Un chemin fauté fabriquait une mémoire vide indiscernable d'un historique
    vide ; seule l'écriture (assert) crée le store."""
    r = _run(
        "croyance",
        "courant",
        str(tmp_path / "absent.jsonl"),
        "--entite",
        "a",
        "--attribut",
        "b",
    )
    assert r.returncode == 1
    assert "store de croyance introuvable" in _erreur_json(r.stderr)["error"]


def test_croyance_drapeaux_hors_action_refuses(tmp_path):
    store = tmp_path / "store.jsonl"
    r = _run(
        "croyance",
        "assert",
        str(store),
        "--entite",
        "e",
        "--attribut",
        "a",
        "--valeur",
        "v",
    )
    assert r.returncode == 0
    r = _run(
        "croyance",
        "courant",
        str(store),
        "--entite",
        "e",
        "--attribut",
        "a",
        "--valeur",
        "X",
    )
    assert r.returncode == 1
    assert "--valeur" in _erreur_json(r.stderr)["error"]
    r = _run("croyance", "calibrer", str(store), "--entite", "e")
    assert r.returncode == 1
    assert "--entite" in _erreur_json(r.stderr)["error"]


def test_actuel_seuil_version_borne(tmp_path):
    r = _run("actuel", "q", str(tmp_path / "idx"), "--seuil-version", "5")
    assert r.returncode == 1
    assert "--seuil-version" in _erreur_json(r.stderr)["error"]


def test_embed_prepare_keep_positif(tmp_path):
    r = _run(
        "embed-prepare",
        str(tmp_path / "x.vec.gz"),
        "-o",
        str(tmp_path / "t"),
        "--keep",
        "0",
    )
    assert r.returncode == 1
    assert "--keep" in _erreur_json(r.stderr)["error"]


def test_abtt_borne_255(tmp_path):
    r = _run("build", str(tmp_path), "-o", str(tmp_path / "i"), "--abtt", "300")
    assert r.returncode == 1
    assert "--abtt" in _erreur_json(r.stderr)["error"]


def test_proches_abtt_refuse_sur_index(tmp_path):
    corpus = _corpus(tmp_path)
    idx = tmp_path / "idx"
    assert _run("build", str(corpus), "-o", str(idx)).returncode == 0
    r = _run("proches", "interrupteur", str(idx), "--abtt", "2")
    assert r.returncode == 1
    assert "--abtt" in _erreur_json(r.stderr)["error"]


def test_proches_dico_refuse_sans_embeddings(tmp_path):
    corpus = _corpus(tmp_path)
    idx = tmp_path / "idx"
    assert _run("build", str(corpus), "-o", str(idx)).returncode == 0
    r = _run("proches", "interrupteur", str(idx), "--dico")
    assert r.returncode == 1
    assert "--dico" in _erreur_json(r.stderr)["error"]


def test_batch_resilient_une_ligne_en_echec_ne_tue_pas_le_flux(tmp_path):
    """Chaque requête en échec émet {'error', 'requete'} sur SA ligne, le flux
    continue, rc final 1 — l'agent corrèle ligne à ligne."""
    corpus = _corpus(tmp_path)
    idx = tmp_path / "idx"
    assert _run("build", str(corpus), "-o", str(idx)).returncode == 0
    r = subprocess.run(
        [*PY, "search", str(idx), "--batch", "--rerank"],
        input="interrupteur\ncarrelage\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    lignes = [json.loads(li) for li in r.stdout.strip().splitlines()]
    assert len(lignes) == 2, "une ligne de sortie par requête, échec compris"
    assert all("error" in li and "requete" in li for li in lignes)
    assert lignes[0]["requete"] == "interrupteur"
    assert r.returncode == 1


def test_meta_racine_toujours_objet_et_degrade_sans_rerank(tmp_path):
    """La forme racine ne dépend plus des drapeaux ({'resultats': ...}), et --rerank
    sur des index sans rerank.msrv DÉGRADE par index au lieu d'échouer en bloc."""
    c1, c2 = _corpus(tmp_path), tmp_path / "corpus2"
    c2.mkdir()
    (c2 / "trois.md").write_text("peinture couloir escalier", encoding="utf-8")
    i1, i2 = tmp_path / "i1", tmp_path / "i2"
    assert _run("build", str(c1), "-o", str(i1)).returncode == 0
    assert _run("build", str(c2), "-o", str(i2)).returncode == 0
    r = _run("meta", "interrupteur", str(i1), str(i2), "--rerank")
    assert r.returncode == 0, r.stderr
    sortie = json.loads(r.stdout)
    assert isinstance(sortie, dict) and "resultats" in sortie
    assert set(sortie["sans_rerank"]) == {"i1", "i2"}


def test_actuel_socle_id(tmp_path):
    """Socle de sortie : tout résultat classé porte `id` (canonique reste en alias)."""
    corpus = _corpus(tmp_path)
    idx = tmp_path / "idx"
    assert _run("build", str(corpus), "-o", str(idx)).returncode == 0
    r = _run("actuel", "interrupteur", str(idx))
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)
    assert res and res[0]["id"] == res[0]["canonique"]


def test_carte_sortie_ok_out(tmp_path):
    corpus = _corpus(tmp_path)
    r = _run("carte", str(corpus))
    assert r.returncode == 0, r.stderr
    sortie = json.loads(r.stdout)
    assert sortie["ok"] is True and "out" in sortie and sortie["docs"] == 2
