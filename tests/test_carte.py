import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from mosaic import carte, ingest
from mosaic.index import Index

GRID = (32, 32, 3)
_PY = [sys.executable, "-m", "mosaic.cli"]


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


def _dossier_mixte(tmp_path: Path) -> Path:
    d = tmp_path / "chantier"
    d.mkdir()
    (d / "note.md").write_text(
        "interrupteur différentiel tableau protection disjoncteur cablage",
        encoding="utf-8",
    )
    (d / "devis.pdf").write_bytes(
        _minimal_pdf("pose interrupteur differentiel protection")
    )
    return d


def _dossier_md_seul(tmp_path: Path) -> Path:
    d = tmp_path / "chantier"
    d.mkdir()
    (d / "un.md").write_text(
        "interrupteur différentiel protection tableau électrique", encoding="utf-8"
    )
    (d / "deux.md").write_text(
        "carrelage colle joint sol cuisine douche", encoding="utf-8"
    )
    return d


# -- carte.generer --------------------------------------------------------------------------


def test_generer_retourne_le_chemin_html(tmp_path):
    d = _dossier_md_seul(tmp_path)
    out, _ = carte.generer(d, grid=GRID)
    assert out == d / "_MOSAIC" / "cartes.html"
    assert out.is_file()


def test_html_contient_une_carte_par_doc_avec_png_et_concepts(tmp_path):
    d = _dossier_md_seul(tmp_path)
    # index_paths=False : ce test porte sur la curation d'affichage des concepts
    # (bruit numérique/boilerplate), pas sur les tokens de chemin (v1.5) — épinglé
    # comme les tests brut/smoothing_rank=0 juste plus bas dans ce fichier.
    out, _ = carte.generer(d, k_concepts=5, grid=GRID, index_paths=False)
    text = out.read_text(encoding="utf-8")
    assert "un.md" in text
    assert "deux.md" in text
    assert text.count("data:image/png;base64,") == 2
    # les concepts dominants doivent apparaître (tokens du vocabulaire)
    assert "interrupteur" in text
    assert "carrelage" in text


def test_html_contient_le_nombre_de_tokens(tmp_path):
    d = _dossier_md_seul(tmp_path)
    out, _ = carte.generer(d, grid=GRID)
    text = out.read_text(encoding="utf-8")
    assert "tokens" in text


def test_html_contient_lien_file_vers_original(tmp_path):
    d = _dossier_md_seul(tmp_path)
    out, _ = carte.generer(d, grid=GRID)
    text = out.read_text(encoding="utf-8")
    original = (d / "un.md").resolve().as_uri()
    assert original in text


def test_html_autonome_sans_ressource_externe(tmp_path):
    d = _dossier_md_seul(tmp_path)
    out, _ = carte.generer(d, grid=GRID)
    text = out.read_text(encoding="utf-8")
    assert "http://" not in text
    assert "https://" not in text
    assert "<link" not in text
    assert "<script src" not in text


def test_generer_tri_par_nom(tmp_path):
    d = tmp_path / "chantier"
    d.mkdir()
    (d / "zeta.md").write_text("mot", encoding="utf-8")
    (d / "alpha.md").write_text("mot", encoding="utf-8")
    out, _ = carte.generer(d, grid=GRID)
    text = out.read_text(encoding="utf-8")
    assert text.index("alpha.md") < text.index("zeta.md")


def test_generer_date_str_par_defaut_et_fixe(tmp_path):
    d = _dossier_md_seul(tmp_path)
    out, _ = carte.generer(d, grid=GRID, date_str="09/08/2026")
    text = out.read_text(encoding="utf-8")
    assert "09/08/2026" in text
    assert "généré par Mosaic" in text


def test_generer_top_k_concepts_limite_le_nombre_de_barres(tmp_path):
    d = _dossier_md_seul(tmp_path)
    out, _ = carte.generer(d, k_concepts=2, grid=GRID)
    text = out.read_text(encoding="utf-8")
    assert text.count('class="bar-row"') <= 2 * 2  # 2 docs x 2 concepts max


def test_generer_pdf_reel_inclus(tmp_path):
    pytest.importorskip("markitdown")
    d = _dossier_mixte(tmp_path)
    out, _ = carte.generer(d, grid=GRID)
    text = out.read_text(encoding="utf-8")
    assert "devis.pdf" in text
    assert "note.md" in text
    assert text.count("data:image/png;base64,") == 2


# -- _MOSAIC jamais indexé (même au rebuild) -------------------------------------------------


def test_mosaic_non_indexe_au_rebuild(tmp_path):
    d = _dossier_md_seul(tmp_path)
    out1, _ = carte.generer(d, grid=GRID)
    # cartes.html a une extension .html : CONVERTIBLE_EXTS le rendrait indexable
    # sans l'exclusion _MOSAIC/. Un 2e passage ne doit pas gonfler le corpus.
    out2, _ = carte.generer(d, grid=GRID)
    idx = Index.open(carte.index_dir(d))
    assert idx.stats()["docs"] == 2
    assert "_MOSAIC/cartes.html" not in idx.ids
    assert (
        out2 == out1
    )  # même chemin HTML régénéré au 2e passage, pas un nouveau fichier


def test_carte_exclut_backups_et_poubelles_via_index_build(tmp_path):
    """v1.5 : EXCLUDED_DIRS élargi (_backups/_corbeille/_cimetiere/poubelleClaude) doit
    bénéficier à `carte.generer` sans aucune duplication de logique — il passe déjà
    par `Index.build`, seul propriétaire d'EXCLUDED_DIRS."""
    d = tmp_path / "chantier"
    d.mkdir()
    (d / "un.md").write_text(
        "interrupteur différentiel protection tableau", encoding="utf-8"
    )
    for sous_dossier in ("_backups", "_corbeille", "_cimetiere", "poubelleClaude"):
        piege = d / sous_dossier
        piege.mkdir()
        (piege / "fantome.md").write_text("jamais indexé", encoding="utf-8")

    carte.generer(d, grid=GRID)
    idx = Index.open(carte.index_dir(d))

    assert idx.ids == ["un.md"]
    assert idx.stats()["docs"] == 1


def test_index_build_exclut_mosaic_directement(tmp_path):
    """_MOSAIC/ est exclu même hors du flux carte.generer, directement via Index.build."""
    d = tmp_path / "chantier"
    d.mkdir()
    (d / "note.md").write_text("contenu réel", encoding="utf-8")
    piege = d / "_MOSAIC"
    piege.mkdir()
    (piege / "cartes.html").write_text("<html>ne pas indexer</html>", encoding="utf-8")
    (piege / "index").mkdir()

    idx = Index.build(d, tmp_path / "idx", grid=GRID)

    assert idx.ids == ["note.md"]
    assert idx.stats()["docs"] == 1


# -- CLI --------------------------------------------------------------------------------------


def test_cli_carte(tmp_path):
    d = _dossier_md_seul(tmp_path)
    r = subprocess.run(
        [*_PY, "carte", str(d), "--top-concepts", "3"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["docs"] == 2
    # Contrat uniforme des commandes produisant un fichier (audit CLI 12/08) :
    # {"ok": true, "out": <chemin>} — "html" etait la seule exception.
    assert out["ok"] is True
    assert Path(out["out"]).is_file()
    assert Path(out["out"]) == d / "_MOSAIC" / "cartes.html"


def test_cli_carte_top_concepts_defaut(tmp_path):
    d = _dossier_md_seul(tmp_path)
    r = subprocess.run(
        [*_PY, "carte", str(d)], capture_output=True, text=True, encoding="utf-8"
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["docs"] == 2


def test_cli_carte_top_concepts_invalide(tmp_path):
    d = _dossier_md_seul(tmp_path)
    r = subprocess.run(
        [*_PY, "carte", str(d), "--top-concepts", "0"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 1
    assert "error" in json.loads(r.stderr)


def test_cli_carte_cache_ingestion_va_sous_le_temp_systeme(tmp_path):
    """--cache-ingestion sur `mosaic carte` doit utiliser le même mécanisme que
    `mosaic build` : cache opt-in sous tempfile.gettempdir()/mosaic_ingest/."""
    pytest.importorskip("markitdown")
    d = _dossier_mixte(tmp_path)

    r = subprocess.run(
        [*_PY, "carte", str(d), "--cache-ingestion"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr

    cache_root = Path(tempfile.gettempdir()) / "mosaic_ingest"
    assert cache_root.is_dir()
    assert list(cache_root.glob("*.txt"))
    assert not any(
        d.glob("*.txt")
    )  # jamais un fichier converti déposé à côté des documents


def test_cli_carte_profile_weighting_et_smoothing_rank_transmis(tmp_path):
    """--profile-weighting et --smoothing-rank de `mosaic carte` doivent être transmis
    jusqu'à Index.build (mêmes parsers que `mosaic build`)."""
    d = _dossier_md_seul(tmp_path)
    r = subprocess.run(
        [*_PY, "carte", str(d), "--profile-weighting", "ppmi", "--smoothing-rank", "2"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["docs"] == 2


def test_cli_carte_profile_weighting_invalide(tmp_path):
    d = _dossier_md_seul(tmp_path)
    r = subprocess.run(
        [*_PY, "carte", str(d), "--profile-weighting", "yolo"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 1
    assert "error" in json.loads(r.stderr)


# -- Curation d'affichage : bruit numérique + boilerplate partagé (revue) --------------------


def _labels(html_text: str) -> list[str]:
    return re.findall(r'class="bar-label" title="([^"]+)"', html_text)


def test_concepts_numeriques_exclus_de_laffichage(tmp_path):
    """Sur ce corpus, le top-5 BRUT d'explain() (vérifié manuellement) contient
    2 tokens purement numériques ("20", "00") — la carte ne doit en montrer aucun,
    et doit tout de même afficher k_concepts barres (assez de mots survivent)."""
    d = tmp_path / "chantier"
    d.mkdir()
    (d / "seul.md").write_text(
        "disjoncteur cablage tableau protection differentiel prise interrupteur "
        "20 20 00 00 10 10 5 5 2 2 1 1 0 0",
        encoding="utf-8",
    )
    # v1.2 a changé les défauts de Index.build (ppmi/300) : ce test porte sur le
    # comportement BRUT vérifié manuellement (docstring ci-dessus), donc épinglé
    # explicitement — indépendant du changement de défaut.
    out, _ = carte.generer(
        d, k_concepts=5, grid=GRID, profile_weighting="brut", smoothing_rank=0
    )
    labels = _labels(out.read_text(encoding="utf-8"))
    assert len(labels) == 5
    assert not any(re.fullmatch(r"[\d.,%€\-]+", lab) for lab in labels)
    assert "prise" in labels and "interrupteur" in labels


def test_concepts_boilerplate_partages_filtres(tmp_path):
    """ "tva"/"ht" apparaissent dans le top-6 (post-filtrage numérique) des DEUX
    documents (vérifié manuellement) — un concept partagé par 100% du dossier ne
    distingue rien, doit être retiré de l'affichage sans liste figée."""
    d = tmp_path / "chantier"
    d.mkdir()
    (d / "un.md").write_text(
        "disjoncteur cablage tableau protection differentiel prise interrupteur "
        "tva ht tva ht tva ht tva ht 20 00 10 5 2 1",
        encoding="utf-8",
    )
    (d / "deux.md").write_text(
        "carrelage colle joint sol cuisine douche faience robinet "
        "tva ht tva ht tva ht tva ht 30 00 15 6 3 2",
        encoding="utf-8",
    )
    # index_paths=False : ce test épingle un top-6 « vérifié manuellement » avant
    # l'injection des tokens de chemin (v1.5) — même raison que le test précédent.
    out, _ = carte.generer(d, k_concepts=6, grid=GRID, index_paths=False)
    labels = _labels(out.read_text(encoding="utf-8"))
    assert "tva" not in labels
    assert "ht" not in labels
    assert len(labels) == 12  # 2 docs x 6 concepts survivants (assez de mots distincts)
    assert "prise" in labels and "faience" in labels


def test_boilerplate_plancher_par_carte_evite_carte_vide(tmp_path):
    """2 documents au vocabulaire quasi-identique (5 mots partagés à 100%) : sans
    plancher, le filtre boilerplate viderait ENTIÈREMENT chaque carte (tous les
    concepts sont "partagés"). Le plancher par carte doit retomber sur le top-k
    filtré du bruit numérique SEULEMENT (sans stoplist boilerplate) pour garder
    des barres visibles."""
    d = tmp_path / "chantier"
    d.mkdir()
    (d / "un.md").write_text(
        "disjoncteur cablage tableau protection differentiel "
        "disjoncteur cablage tableau protection differentiel "
        "20 00 10 5 2",
        encoding="utf-8",
    )
    (d / "deux.md").write_text(
        "disjoncteur cablage tableau protection differentiel "
        "differentiel protection tableau cablage disjoncteur "
        "30 00 15 6 3",
        encoding="utf-8",
    )
    out, _ = carte.generer(d, k_concepts=5, grid=GRID)
    text = out.read_text(encoding="utf-8")
    labels = _labels(text)
    assert (
        len(labels) == 10
    )  # 2 docs x 5 concepts — aucune carte vide malgré 100% de partage
    assert '<div class="bar-fill"' in text


def test_boilerplate_non_applique_a_un_seul_document(tmp_path):
    """À 1 seul document, aucune notion de "partagé par le dossier" n'est
    calculable — la règle boilerplate ne doit pas vider la carte unique."""
    d = tmp_path / "chantier"
    d.mkdir()
    (d / "seul.md").write_text(
        "disjoncteur cablage tableau protection differentiel prise interrupteur tva ht",
        encoding="utf-8",
    )
    out, _ = carte.generer(d, k_concepts=5, grid=GRID)
    labels = _labels(out.read_text(encoding="utf-8"))
    assert len(labels) == 5


# -- Régression : réutilisation du cache d'ingestion pour le comptage de tokens --------------


def test_generer_reutilise_cache_ingestion_pour_comptage_tokens(tmp_path, monkeypatch):
    """Index.build convertit un PDF une fois pour l'encodage ; carte._token_count()
    ne doit pas le reconvertir quand ingest_cache_dir est fourni — sert le cache
    disque au lieu de relancer markitdown une 2e fois pour le même fichier."""
    pytest.importorskip("markitdown")
    from markitdown import MarkItDown

    calls = []
    real_converter = MarkItDown()  # instance indépendante, jamais le singleton partagé

    class _CountingConverter:
        def convert(self, path, **kwargs):
            calls.append(path)
            return real_converter.convert(path, **kwargs)

    monkeypatch.setattr(ingest, "_get_converter", lambda: _CountingConverter())

    d = tmp_path / "chantier"
    d.mkdir()
    (d / "devis.pdf").write_bytes(_minimal_pdf("cablage tableau protection"))
    cache_dir = tmp_path / "cache"

    carte.generer(d, grid=GRID, ingest_cache_dir=cache_dir)

    assert (
        len(calls) == 1
    )  # 1 seule conversion réelle : build l'a convertie, _token_count a servi le cache
