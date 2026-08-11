"""Tests volet A v1.6 : le chiffrage par similarité (document-as-query).

Index.search_like(doc_or_liste, k, rerank, ...) et CLI `mosaic like`.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mosaic import ingest as ingest_module
from mosaic import rerank
from mosaic.index import Index

GRID = (32, 32, 3)
PY = [sys.executable, "-m", "mosaic.cli"]


def _minimal_pdf(text: str) -> bytes:
    """Construit un PDF 1 page minimal et valide (xref calculé), avec `text` en contenu
    (dupliqué de tests/test_cli.py — pas de package `tests` importable, cf. absence
    de __init__.py)."""
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


def _corpus_devis(tmp_path: Path) -> Path:
    c = tmp_path / "corpus"
    c.mkdir(parents=True, exist_ok=True)
    docs = {
        "relamping_a.md": (
            "relamping éclairage led remplacement tubes fluorescents entrepôt luminaires"
        ),
        "relamping_b.md": (
            "remplacement luminaires led éclairage entrepôt tubes néon relamping"
        ),
        "relamping_c.md": (
            "éclairage led relamping usine remplacement réglettes fluorescentes"
        ),
        "plomberie_a.md": "chauffe-eau ballon eau chaude raccordement cuivre plomberie",
        "plomberie_b.md": "remplacement chauffe-eau ballon raccordement sanitaire",
        "carrelage_a.md": "achat carrelage gris cuisine colle joint pose sol",
    }
    for name, text in docs.items():
        (c / name).write_text(text, encoding="utf-8")
    return c


def _corpus_devis_imbrique(tmp_path: Path) -> Path:
    """Comme _corpus_devis, mais avec des ids à plusieurs niveaux (self.ids en POSIX, ex.
    "2025/11.2025/relamping_a.md") — nécessaire pour tester la normalisation antislash."""
    c = tmp_path / "corpus"
    sub = c / "2025" / "11.2025"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "relamping_a.md").write_text(
        "relamping éclairage led remplacement tubes fluorescents entrepôt luminaires",
        encoding="utf-8",
    )
    (sub / "relamping_b.md").write_text(
        "remplacement luminaires led éclairage entrepôt tubes néon relamping",
        encoding="utf-8",
    )
    (c / "plomberie_a.md").write_text(
        "chauffe-eau ballon eau chaude raccordement cuivre plomberie", encoding="utf-8"
    )
    return c


def _fake_model(monkeypatch) -> None:
    """Même doublure que test_index_rerank.py : vecteurs déterministes seedés par hash(texte)."""

    class _FakeModel:
        def encode(self, texts):
            out = np.zeros((len(texts), rerank.DIM), dtype=np.float32)
            for i, t in enumerate(texts):
                out[i] = np.random.default_rng(hash(t) & 0xFFFFFFFF).normal(
                    size=rerank.DIM
                )
            return out

    # available() (donc le garde de build rerank_vectors) doit passer sans le vrai
    # model2vec installé : on rend StaticModel non-None. Les tests qui veulent l'ABSENCE
    # au runtime reposent StaticModel=None eux-mêmes après le build.
    monkeypatch.setattr(rerank, "StaticModel", _FakeModel)
    monkeypatch.setattr(rerank, "_get_model", lambda: _FakeModel())


# -- Index.search_like : id interne ----------------------------------------------------------


def test_search_like_id_interne_exclut_soi_meme_et_trouve_similaires(tmp_path):
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    hits = idx.search_like("relamping_a.md", k=5)
    ids = [h["id"] for h in hits]
    assert "relamping_a.md" not in ids
    assert set(ids[:2]) == {"relamping_b.md", "relamping_c.md"}


def test_search_like_id_interne_utilise_directement_la_ligne_stockee(tmp_path):
    """La requête d'un id interne est bit-identique à un dot product avec la ligne
    int8 stockée (pas de ré-encodage flottant) — cf. contrat spec §A."""
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    row = idx.ids.index("relamping_a.md")
    q_attendu = idx.mat[row]
    scores_attendus = idx.mat @ q_attendu.astype(np.int32)
    hits = idx.search_like("relamping_a.md", k=len(idx.ids))
    for h in hits:
        i = idx.ids.index(h["id"])
        denom = float(idx.norms[i]) * float(idx.norms[row])
        attendu = round(float(scores_attendus[i]) / denom, 6) if denom else 0.0
        assert h["score"] == pytest.approx(attendu, abs=1e-5)


def test_search_like_id_interne_imbrique_avec_antislashs_windows_est_normalise(
    tmp_path,
):
    """Revue (Critical, reproduit) : self.ids est en POSIX ('2025/11.2025/...') ; une saisie
    Windows naturelle avec des antislashs ('2025\\11.2025\\...') doit matcher le même id
    interne (normalisation avant comparaison) — sans quoi la requête retombait à tort sur la
    branche fichier externe et le document réapparaissait dans ses propres résultats."""
    idx = Index.build(_corpus_devis_imbrique(tmp_path), tmp_path / "idx", grid=GRID)
    id_posix = "2025/11.2025/relamping_a.md"
    assert id_posix in idx.ids
    id_windows = id_posix.replace("/", "\\")

    hits = idx.search_like(id_windows, k=5)

    ids = [h["id"] for h in hits]
    assert id_posix not in ids  # exclu de ses propres résultats malgré la saisie \
    assert "2025/11.2025/relamping_b.md" in ids


def test_search_like_id_connu_prioritaire_sur_chemin_de_fichier(tmp_path):
    """Si la même chaîne correspond à la fois à un id connu ET à un fichier existant sur le
    filesystem, l'id l'emporte toujours (spec CLI : « ids... internal ; existing filesystem
    paths... external », dans cet ordre de priorité)."""
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    # un fichier filesystem du même nom existe aussi (contenu totalement différent) : si le
    # chemin l'emportait à tort, le pipeline "requête externe" (sans injection de chemin)
    # encoderait CE contenu plutôt que de réutiliser la ligne stockée de "relamping_a.md".
    piege = tmp_path / "relamping_a.md"
    piege.write_text("carrelage gris cuisine colle joint pose sol", encoding="utf-8")

    row = idx.ids.index("relamping_a.md")
    hits_id = idx.search_like("relamping_a.md", k=len(idx.ids))
    scores_attendus = idx.mat @ idx.mat[row].astype(np.int32)
    for h in hits_id:
        i = idx.ids.index(h["id"])
        denom = float(idx.norms[i]) * float(idx.norms[row])
        attendu = round(float(scores_attendus[i]) / denom, 6) if denom else 0.0
        assert h["score"] == pytest.approx(attendu, abs=1e-5)


def test_search_like_doc_introuvable_leve_valueerror(tmp_path):
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    with pytest.raises(ValueError):
        idx.search_like("n_existe_pas_du_tout.md")


def test_search_like_liste_vide_leve_valueerror(tmp_path):
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    with pytest.raises(ValueError):
        idx.search_like([])


# -- Index.search_like : fichier externe -------------------------------------------------------


def test_search_like_fichier_externe_pipeline_complet(tmp_path):
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    external = tmp_path / "nouveau_chantier.md"
    external.write_text(
        "relamping éclairage led entrepôt remplacement tubes", encoding="utf-8"
    )
    hits = idx.search_like(str(external), k=3)
    assert hits[0]["id"] in {"relamping_a.md", "relamping_b.md", "relamping_c.md"}


def test_search_like_fichier_externe_ignore_les_tokens_du_nom(tmp_path):
    """Contrat spec §A : « path tokens NOT injected for a query-document — it's une
    requête ». Un doc interne nommé ...KUMQUAT... est électrique ; la requête externe
    nommée KUMQUAT_... est en fait de la plomberie. Si les tokens de chemin de la requête
    étaient (à tort) injectés, "pomme" tirerait le score vers le doc électrique malgré un
    contenu totalement différent."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "D26089903_KUMQUAT_devis.md").write_text(
        "fourniture pose tableau électrique cablage protection", encoding="utf-8"
    )
    (c / "plomberie_ballon.md").write_text(
        "chauffe-eau ballon eau chaude raccordement cuivre", encoding="utf-8"
    )
    idx = Index.build(c, tmp_path / "idx", grid=GRID)
    assert "pomme" in idx.profiles.rows  # confirmé injecté au vocabulaire via le BUILD

    external = tmp_path / "KUMQUAT_nouveau_chantier.md"
    external.write_text("chauffe-eau ballon eau chaude raccordement", encoding="utf-8")
    hits = idx.search_like(str(external), k=1)
    assert hits[0]["id"] == "plomberie_ballon.md"


def test_search_like_extension_non_supportee_leve_valueerror(tmp_path):
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    bad = tmp_path / "photo.jpg"
    # ni .md/.txt ni un convertible markitdown (ingest.CONVERTIBLE_EXTS) — les images
    # (ingest.IMAGE_EXTS, volet B) restent hors périmètre de search_like (spec §A).
    bad.write_bytes(b"\xff\xd8\xff\xe0")
    with pytest.raises(ValueError):
        idx.search_like(str(bad))


def test_search_like_fichier_externe_convertible_sans_markitdown_erreur_actionnable(
    tmp_path, monkeypatch
):
    """Revue (Important, reproduit) : markitdown absent (extra `ingest` non installé) sur un
    convertible levait un « document illisible ou vide » trompeur — indistinguable d'une
    vraie conversion échouée. Doit lever un message actionnable, PEU IMPORTE que markitdown
    soit réellement installé dans cet environnement de dev (pas d'importorskip ici : le test
    force `ingest.available()` à False)."""
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    monkeypatch.setattr(ingest_module, "available", lambda: False)
    pdf = tmp_path / "devis_client.pdf"
    pdf.write_bytes(
        b"%PDF-1.4\n%%EOF"
    )  # jamais lu : available() coupe avant toute conversion

    with pytest.raises(ValueError, match="markitdown"):
        idx.search_like(str(pdf), k=3)


def test_search_like_fichier_externe_convertible_pdf(tmp_path):
    pytest.importorskip("markitdown")

    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    pdf = tmp_path / "devis_client.pdf"
    texte = "relamping eclairage led entrepot remplacement tubes fluorescents " * 3
    pdf.write_bytes(_minimal_pdf(texte))
    hits = idx.search_like(str(pdf), k=3)
    assert hits[0]["id"] in {"relamping_a.md", "relamping_b.md", "relamping_c.md"}


def test_search_like_ingest_cache_dir_reutilise_la_conversion(tmp_path, monkeypatch):
    """Même mécanique que `mosaic build --cache-ingestion` (test_ingest.py) : un 2e appel
    avec le même `ingest_cache_dir` doit relire le cache disque, pas reconvertir."""
    pytest.importorskip("markitdown")
    from markitdown import MarkItDown

    calls = []
    real_converter = MarkItDown()

    class _CountingConverter:
        def convert(self, path, **kwargs):
            calls.append(path)
            return real_converter.convert(path, **kwargs)

    monkeypatch.setattr(ingest_module, "_get_converter", lambda: _CountingConverter())

    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    pdf = tmp_path / "devis_client.pdf"
    texte = "relamping eclairage led entrepot remplacement tubes fluorescents " * 3
    pdf.write_bytes(_minimal_pdf(texte))
    cache_dir = tmp_path / "cache"

    idx.search_like(str(pdf), k=3, ingest_cache_dir=cache_dir)
    idx.search_like(str(pdf), k=3, ingest_cache_dir=cache_dir)

    assert len(calls) == 1  # le 2e appel a réutilisé le cache disque


# -- Index.search_like : mélange (mix) ---------------------------------------------------------


def test_search_like_mix_de_deux_docs_similaires_retrouve_le_troisieme(tmp_path):
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    hits = idx.search_like(["relamping_a.md", "relamping_b.md"], k=1)
    assert hits[0]["id"] == "relamping_c.md"


def test_search_like_mix_exclut_les_deux_docs_sources(tmp_path):
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    hits = idx.search_like(["relamping_a.md", "plomberie_a.md"], k=5)
    ids = [h["id"] for h in hits]
    assert "relamping_a.md" not in ids
    assert "plomberie_a.md" not in ids
    assert len(ids) > 0


def test_search_like_mix_interne_et_externe(tmp_path):
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    external = tmp_path / "nouveau.md"
    external.write_text("relamping éclairage led entrepôt", encoding="utf-8")
    hits = idx.search_like(["relamping_a.md", str(external)], k=1)
    assert hits[0]["id"] in {"relamping_b.md", "relamping_c.md"}


def test_search_like_mix_document_duplique_leve_valueerror(tmp_path):
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    with pytest.raises(ValueError):
        idx.search_like(["relamping_a.md", "relamping_a.md"], k=3)


def test_search_like_mix_id_duplique_apres_normalisation_antislash_leve_valueerror(
    tmp_path,
):
    """Même id donné une fois en POSIX et une fois avec des antislashs Windows : c'est le
    MÊME document (normalisation), donc un doublon — refusé comme les autres doublons."""
    idx = Index.build(_corpus_devis_imbrique(tmp_path), tmp_path / "idx", grid=GRID)
    id_posix = "2025/11.2025/relamping_a.md"
    id_windows = id_posix.replace("/", "\\")
    with pytest.raises(ValueError):
        idx.search_like([id_posix, id_windows], k=3)


def test_search_like_mix_deux_documents_distincts_fonctionne(tmp_path):
    """Non-régression : le garde-fou de doublon ne doit jamais gêner un mélange de deux
    documents réellement différents."""
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    hits = idx.search_like(["relamping_a.md", "relamping_b.md"], k=1)
    assert hits[0]["id"] == "relamping_c.md"


# -- Index.search_like : rerank -----------------------------------------------------------------


def test_search_like_rerank_id_interne_sans_model2vec_fonctionne(tmp_path, monkeypatch):
    """Le contrat clé de la spec : id interne + --rerank ne nécessite AUCUN modèle
    (réutilise l'empreinte rerank déjà stockée)."""
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus_devis(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    monkeypatch.setattr(
        rerank, "StaticModel", None
    )  # simule model2vec absent au runtime
    hits = idx.search_like("relamping_a.md", k=3, rerank=True)
    assert len(hits) == 3
    assert all("score_rerank" in h for h in hits)


def test_search_like_rerank_fichier_externe_sans_model2vec_leve_valueerror(
    tmp_path, monkeypatch
):
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus_devis(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    external = tmp_path / "nouveau.md"
    external.write_text("relamping éclairage led entrepôt", encoding="utf-8")
    monkeypatch.setattr(rerank, "StaticModel", None)
    with pytest.raises(ValueError):
        idx.search_like(str(external), k=3, rerank=True)


def test_search_like_rerank_fichier_externe_fonctionne_avec_modele(
    tmp_path, monkeypatch
):
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus_devis(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    external = tmp_path / "nouveau.md"
    external.write_text("relamping éclairage led entrepôt", encoding="utf-8")
    hits = idx.search_like(str(external), k=3, rerank=True)
    assert all("score_rerank" in h for h in hits)


def test_search_like_rerank_sans_msrv_leve_valueerror(tmp_path):
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    with pytest.raises(ValueError):
        idx.search_like("relamping_a.md", k=3, rerank=True)


def test_search_like_rerank_mix_deux_ids_internes(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus_devis(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    hits = idx.search_like(["relamping_a.md", "relamping_b.md"], k=1, rerank=True)
    assert hits[0]["id"] == "relamping_c.md"
    assert "score_rerank" in hits[0]


# -- non-régression : search() classique inchangé -----------------------------------------------


def test_search_classique_inchange_apres_refactor(tmp_path):
    idx = Index.build(_corpus_devis(tmp_path), tmp_path / "idx", grid=GRID)
    hits = idx.search("relamping éclairage led", k=3)
    assert hits[0]["id"] in {"relamping_a.md", "relamping_b.md", "relamping_c.md"}
    assert all(set(h.keys()) == {"id", "score"} for h in hits)


def test_search_rerank_classique_inchange_apres_refactor(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus_devis(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    hits = idx.search("relamping éclairage led", k=3, rerank=True)
    assert all("score_rerank" in h and "score" in h for h in hits)


# -- CLI `mosaic like` --------------------------------------------------------------------------


def _run(*args: str):
    return subprocess.run(
        [*PY, *args], capture_output=True, text=True, encoding="utf-8"
    )


def _build_cli_corpus(tmp_path: Path) -> str:
    idx = str(tmp_path / "idx")
    r = _run("build", str(_corpus_devis(tmp_path)), "-o", idx, "--grid", "32x32")
    assert r.returncode == 0, r.stderr
    return idx


def test_cli_like_id_interne(tmp_path):
    idx = _build_cli_corpus(tmp_path)
    r = _run("like", "relamping_a.md", idx, "--top", "5")
    assert r.returncode == 0, r.stderr
    hits = json.loads(r.stdout)
    ids = [h["id"] for h in hits]
    assert "relamping_a.md" not in ids
    assert set(ids[:2]) == {"relamping_b.md", "relamping_c.md"}


def test_cli_like_mix_deux_positionnels(tmp_path):
    idx = _build_cli_corpus(tmp_path)
    r = _run("like", "relamping_a.md", "relamping_b.md", idx, "--top", "1")
    assert r.returncode == 0, r.stderr
    hits = json.loads(r.stdout)
    assert hits[0]["id"] == "relamping_c.md"


def test_cli_like_fichier_externe(tmp_path):
    idx = _build_cli_corpus(tmp_path)
    external = Path(idx).parent / "nouveau_chantier.md"
    external.write_text(
        "relamping éclairage led entrepôt remplacement tubes", encoding="utf-8"
    )
    r = _run("like", str(external), idx, "--top", "3")
    assert r.returncode == 0, r.stderr
    hits = json.loads(r.stdout)
    assert hits[0]["id"] in {"relamping_a.md", "relamping_b.md", "relamping_c.md"}


def test_cli_like_cache_ingestion_cree_le_cache(tmp_path):
    """--cache-ingestion sur `mosaic like` : même mécanique que `mosaic build`
    (cf. test_ingest.py::test_cli_build_cache_ingestion_va_sous_le_temp_systeme)."""
    pytest.importorskip("markitdown")
    idx = _build_cli_corpus(tmp_path)
    pdf = Path(idx).parent / "nouveau_chantier.pdf"
    texte = "relamping eclairage led entrepot remplacement tubes fluorescents " * 3
    pdf.write_bytes(_minimal_pdf(texte))

    cache_root = Path(tempfile.gettempdir()) / "mosaic_ingest"
    before = set(cache_root.glob("*.txt")) if cache_root.is_dir() else set()

    r = _run("like", str(pdf), idx, "--top", "3", "--cache-ingestion")
    assert r.returncode == 0, r.stderr

    after = set(cache_root.glob("*.txt")) if cache_root.is_dir() else set()
    assert after - before  # au moins un fichier de cache nouvellement créé


def test_cli_like_mix_document_duplique_erreur_json(tmp_path):
    idx = _build_cli_corpus(tmp_path)
    r = _run("like", "relamping_a.md", "relamping_a.md", idx, "--top", "3")
    assert r.returncode == 1
    assert "error" in json.loads(r.stderr)


def test_cli_like_doc_introuvable_erreur_json(tmp_path):
    idx = _build_cli_corpus(tmp_path)
    r = _run("like", "n_existe_pas.md", idx)
    assert r.returncode == 1
    assert "error" in json.loads(r.stderr)


def test_cli_like_un_seul_positionnel_erreur_json(tmp_path):
    """Un seul positionnel = pas de doc distinct de l'index -> erreur claire."""
    idx = _build_cli_corpus(tmp_path)
    r = _run("like", idx)
    assert r.returncode == 1
    assert "error" in json.loads(r.stderr)


def test_cli_like_top_invalide_erreur_json(tmp_path):
    idx = _build_cli_corpus(tmp_path)
    r = _run("like", "relamping_a.md", idx, "--top", "0")
    assert r.returncode == 1
    err = json.loads(r.stderr)
    assert "error" in err and "--top" in err["error"]


def test_cli_like_avec_rerank_id_interne(tmp_path):
    pytest.importorskip("model2vec")
    idx = str(tmp_path / "idx")
    r = _run(
        "build",
        str(_corpus_devis(tmp_path)),
        "-o",
        idx,
        "--grid",
        "32x32",
        "--rerank-vectors",
    )
    assert r.returncode == 0, r.stderr
    r = _run("like", "relamping_a.md", idx, "--top", "3", "--rerank")
    assert r.returncode == 0, r.stderr
    hits = json.loads(r.stdout)
    assert all("score_rerank" in h for h in hits)


def test_cli_like_index_absent_erreur_json(tmp_path):
    r = _run("like", "x.md", str(tmp_path / "nulle_part"))
    assert r.returncode == 1
    assert "error" in json.loads(r.stderr)
