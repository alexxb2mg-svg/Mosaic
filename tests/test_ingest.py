import json
import subprocess
import sys

import pytest

from mosaic import ingest
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


# -- ingest.available() / to_text() --------------------------------------------------------


def test_available_reflects_markitdown_import(monkeypatch):
    monkeypatch.setattr(ingest, "MarkItDown", None)
    assert ingest.available() is False


def test_available_true_quand_markitdown_installe():
    pytest.importorskip("markitdown")
    assert ingest.available() is True


def test_to_text_sans_markitdown_retourne_none(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "MarkItDown", None)
    f = tmp_path / "doc.pdf"
    f.write_bytes(_minimal_pdf("peu importe"))
    assert ingest.to_text(f) is None


def test_to_text_fichier_corrompu_retourne_none(tmp_path):
    pytest.importorskip("markitdown")
    corrompu = tmp_path / "corrompu.pdf"
    corrompu.write_bytes(bytes([0xFF, 0x00, 0xDE, 0xAD, 0xBE, 0xEF]) * 50)
    assert ingest.to_text(corrompu) is None


def test_to_text_convertit_pdf_reel(tmp_path):
    pytest.importorskip("markitdown")
    f = tmp_path / "un_paragraphe.pdf"
    f.write_bytes(_minimal_pdf("Bonjour Mosaic"))
    text = ingest.to_text(f)
    assert text is not None
    assert "Bonjour Mosaic" in text


def test_to_text_convertit_pptx_reel(tmp_path):
    """L'extra `ingest` promet .pptx (CONVERTIBLE_EXTS + README) : vérifie que la conversion
    réelle fonctionne, pas seulement que l'extension est listée."""
    pytest.importorskip("markitdown")
    pptx = pytest.importorskip("pptx")
    f = tmp_path / "presentation.pptx"
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[0])
    slide.shapes.title.text = "Bonjour Mosaic"
    presentation.save(f)
    text = ingest.to_text(f)
    assert text is not None
    assert "Bonjour Mosaic" in text


def test_cache_evite_la_reconversion(tmp_path, monkeypatch):
    calls = []

    class _FakeConverter:
        def convert(self, path):
            calls.append(path)

            class _R:
                text_content = "contenu converti une seule fois"

            return _R()

    monkeypatch.setattr(ingest, "available", lambda: True)
    monkeypatch.setattr(ingest, "_get_converter", lambda: _FakeConverter())

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"peu importe le contenu binaire ici")
    cache_dir = tmp_path / "cache"

    first = ingest.to_text(src, cache_dir=cache_dir)
    second = ingest.to_text(src, cache_dir=cache_dir)

    assert first == second == "contenu converti une seule fois"
    assert len(calls) == 1  # le 2e appel a servi le cache, pas reconverti


def test_cache_est_opt_in_sans_cache_dir_reconvertit(tmp_path, monkeypatch):
    calls = []

    class _FakeConverter:
        def convert(self, path):
            calls.append(path)

            class _R:
                text_content = "texte"

            return _R()

    monkeypatch.setattr(ingest, "available", lambda: True)
    monkeypatch.setattr(ingest, "_get_converter", lambda: _FakeConverter())

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"peu importe")

    ingest.to_text(src)
    ingest.to_text(src)

    assert len(calls) == 2  # sans cache_dir, aucune mémoïsation


def test_cache_invalide_si_fichier_modifie(tmp_path, monkeypatch):
    calls = []

    class _FakeConverter:
        def convert(self, path):
            calls.append(path)

            class _R:
                text_content = f"version {len(calls)}"

            return _R()

    monkeypatch.setattr(ingest, "available", lambda: True)
    monkeypatch.setattr(ingest, "_get_converter", lambda: _FakeConverter())

    src = tmp_path / "doc.pdf"
    cache_dir = tmp_path / "cache"

    src.write_bytes(b"contenu v1")
    first = ingest.to_text(src, cache_dir=cache_dir)
    src.write_bytes(b"contenu v2 plus long")
    second = ingest.to_text(src, cache_dir=cache_dir)

    assert first != second
    assert len(calls) == 2


def test_cache_corrompu_retombe_sur_conversion_fraiche_sans_crash(
    tmp_path, monkeypatch
):
    """Une entrée de cache illisible en UTF-8 (écriture partielle, corruption disque) ne doit
    jamais faire planter le build — retomber sur une conversion fraîche (brief : JAMAIS de
    crash)."""
    calls = []

    class _FakeConverter:
        def convert(self, path):
            calls.append(path)

            class _R:
                text_content = "texte fraîchement reconverti"

            return _R()

    monkeypatch.setattr(ingest, "available", lambda: True)
    monkeypatch.setattr(ingest, "_get_converter", lambda: _FakeConverter())

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"peu importe le contenu binaire ici")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / f"{ingest._cache_key(src)}.txt"
    cache_file.write_bytes(b"\xff\xfe\x00garbage")  # octets non décodables en UTF-8

    result = ingest.to_text(src, cache_dir=cache_dir)

    assert result == "texte fraîchement reconverti"
    assert len(calls) == 1  # cache corrompu ignoré, conversion fraîche effectuée
    assert (
        cache_file.read_text(encoding="utf-8") == "texte fraîchement reconverti"
    )  # réécrit


def test_cache_isole_selon_le_drapeau_ocr_sans_ocr_puis_avec_ocr(tmp_path, monkeypatch):
    """Bug revue finale (important, reproduit) : la clé de cache ne dépendait pas de `ocr`.
    Un doc muet mis en cache SANS --ocr était servi tel quel à un run AVEC --ocr (jamais
    d'OCR déclenché, jamais l'erreur promise). Les deux modes doivent avoir des entrées
    de cache distinctes."""
    pytest.importorskip("markitdown")
    provider_calls = []
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(
        ingest,
        "ocr_provider",
        lambda path: (
            provider_calls.append(path) or "texte océrisé bien plus long " * 10
        ),
    )

    f = tmp_path / "court.pdf"
    f.write_bytes(_minimal_pdf("bref"))
    cache_dir = tmp_path / "cache"

    # 1er appel SANS --ocr : met en cache le texte court markitdown (comportement v1.4).
    text_sans_ocr = ingest.to_text(f, cache_dir=cache_dir, ocr=False)
    assert text_sans_ocr is not None and len(text_sans_ocr) < 200
    assert len(provider_calls) == 0

    # 2e appel AVEC --ocr, même fichier/cache_dir : doit reconvertir et déclencher l'OCR —
    # jamais servir le texte court caché sous la clé sans OCR.
    text_avec_ocr = ingest.to_text(f, cache_dir=cache_dir, ocr=True)
    assert len(provider_calls) == 1
    assert text_avec_ocr is not None and text_avec_ocr.startswith("texte océrisé")


def test_cache_isole_selon_le_drapeau_ocr_avec_ocr_puis_sans_ocr(tmp_path, monkeypatch):
    """Sens inverse : caché AVEC --ocr d'abord, puis relu SANS --ocr ne doit jamais servir
    le texte océrisé — reconversion fraîche (texte court, comportement v1.4 exact)."""
    pytest.importorskip("markitdown")
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(
        ingest,
        "ocr_provider",
        lambda path: "texte océrisé bien plus long " * 10,
    )

    f = tmp_path / "court.pdf"
    f.write_bytes(_minimal_pdf("bref"))
    cache_dir = tmp_path / "cache"

    text_avec_ocr = ingest.to_text(f, cache_dir=cache_dir, ocr=True)
    assert text_avec_ocr is not None and text_avec_ocr.startswith("texte océrisé")

    text_sans_ocr = ingest.to_text(f, cache_dir=cache_dir, ocr=False)
    assert text_sans_ocr is not None
    assert not text_sans_ocr.startswith("texte océrisé")
    assert len(text_sans_ocr) < 200


def test_cache_key_differe_selon_ocr(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"peu importe")
    assert ingest._cache_key(f, ocr=False) != ingest._cache_key(f, ocr=True)


# -- OCR (v1.5) : crochet enfichable, rapidocr_onnxruntime NON installé dans ce lot ----------


def test_available_ocr_false_sans_moteur(monkeypatch):
    """Sans moteur importable, la détection dégrade proprement (False), jamais
    d'exception. (Simulé par monkeypatch : rapidocr est installé sur cette machine
    depuis la validation du 09/08.)"""
    monkeypatch.setattr(ingest, "RapidOCR", None)
    assert ingest.available_ocr() is False


def test_available_ocr_reflects_import(monkeypatch):
    monkeypatch.setattr(ingest, "RapidOCR", object())
    assert ingest.available_ocr() is True
    monkeypatch.setattr(ingest, "RapidOCR", None)
    assert ingest.available_ocr() is False


def test_to_text_ocr_false_defaut_comportement_actuel(tmp_path):
    """Sans --ocr, un convertible dont la conversion rend un texte court reste tel
    quel (jamais d'OCR déclenché, jamais d'erreur) — comportement v1.4 exact."""
    pytest.importorskip("markitdown")
    f = tmp_path / "court.pdf"
    f.write_bytes(_minimal_pdf("bref"))
    text = ingest.to_text(f)
    assert text is not None
    assert len(text) < 200


def test_to_text_ocr_demande_sans_provider_leve_erreur_claire(tmp_path, monkeypatch):
    """Sans moteur OCR (simulé) : --ocr sur un document muet (< 200 caractères
    convertis) doit lever une erreur claire, jamais un faux-succès silencieux."""
    pytest.importorskip("markitdown")
    monkeypatch.setattr(ingest, "RapidOCR", None)
    monkeypatch.setattr(ingest, "_ocr_engine", None)
    f = tmp_path / "court.pdf"
    f.write_bytes(_minimal_pdf("bref"))
    with pytest.raises(ValueError):
        ingest.to_text(f, ocr=True)


def test_to_text_ocr_provider_factice_appele_si_texte_court(tmp_path, monkeypatch):
    pytest.importorskip("markitdown")
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(
        ingest, "ocr_provider", lambda path: "texte océrisé bien plus long " * 10
    )
    f = tmp_path / "court.pdf"
    f.write_bytes(_minimal_pdf("bref"))
    text = ingest.to_text(f, ocr=True)
    assert text is not None and text.startswith("texte océrisé")


def test_to_text_ocr_non_declenche_si_texte_deja_suffisant(tmp_path, monkeypatch):
    """Le crochet OCR ne doit s'activer QUE sous le seuil (< 200 caractères) —
    un convertible déjà bavard ne doit jamais appeler le provider."""
    pytest.importorskip("markitdown")
    calls = []
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(
        ingest,
        "ocr_provider",
        lambda path: calls.append(path) or "ne doit jamais servir",
    )
    long_text = (
        "pose interrupteur différentiel tableau protection cablage disjoncteur " * 4
    )
    f = tmp_path / "long.pdf"
    f.write_bytes(_minimal_pdf(long_text))
    text = ingest.to_text(f, ocr=True)
    assert len(calls) == 0
    assert "ne doit jamais servir" not in text


def test_to_text_ocr_provider_echoue_retombe_sur_texte_existant(tmp_path, monkeypatch):
    """Provider disponible mais qui échoue (None) : jamais de crash, on retombe sur
    le texte de conversion existant (même court)."""
    pytest.importorskip("markitdown")
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(ingest, "ocr_provider", lambda path: None)
    f = tmp_path / "court.pdf"
    f.write_bytes(_minimal_pdf("bref"))
    text = ingest.to_text(f, ocr=True)
    assert text is not None  # le texte markitdown (court) reste servi, pas de crash


def test_to_text_ocr_sans_markitdown_disponible(tmp_path, monkeypatch):
    """OCR peut relayer même si markitdown est absent (texte de départ = None
    déclenche aussi le crochet, sous le même seuil)."""
    monkeypatch.setattr(ingest, "MarkItDown", None)
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(ingest, "ocr_provider", lambda path: "texte océrisé pur")
    f = tmp_path / "scan.pdf"
    f.write_bytes(b"peu importe, jamais lu sans markitdown")
    text = ingest.to_text(f, ocr=True)
    assert text == "texte océrisé pur"


def test_ocr_provider_defaut_retourne_none_sans_moteur(tmp_path):
    """Le provider par défaut (rapidocr sous garde) ne doit jamais crasher quand le
    moteur est absent — None, jamais d'exception qui remonterait à l'appelant."""
    f = tmp_path / "scan.pdf"
    f.write_bytes(b"peu importe")
    assert ingest.ocr_provider(f) is None


# -- Photos et images (v1.6 §B) : IMAGE_EXTS, to_text spécialisé --------------------------------


def test_image_exts_contenu_attendu():
    assert ingest.IMAGE_EXTS == {".jpg", ".jpeg", ".png", ".tiff"}


def test_image_exts_disjoint_de_convertible_exts():
    """Design v1.6 §B : IMAGE_EXTS est un NOUVEL ensemble, jamais fusionné dans
    CONVERTIBLE_EXTS (chaque consommateur décide explicitement de le prendre en compte)."""
    assert ingest.IMAGE_EXTS.isdisjoint(ingest.CONVERTIBLE_EXTS)


def test_to_text_image_sans_ocr_retourne_none_jamais_markitdown(tmp_path, monkeypatch):
    """Sans --ocr, une image est ignorée — jamais de tentative markitdown (aucune couche
    texte possible pour une image), comme un convertible sans markitdown."""
    calls = []

    class _ConverterQuiNeDoitJamaisEtreAppele:
        def convert(self, path):
            calls.append(path)
            raise AssertionError("markitdown ne doit jamais être invoqué sur une image")

    monkeypatch.setattr(ingest, "available", lambda: True)
    monkeypatch.setattr(
        ingest, "_get_converter", lambda: _ConverterQuiNeDoitJamaisEtreAppele()
    )

    f = tmp_path / "plaque.jpg"
    f.write_bytes(b"\xff\xd8\xff\xe0 peu importe le contenu binaire jpeg")

    assert ingest.to_text(f, ocr=False) is None
    assert len(calls) == 0


def test_to_text_image_avec_ocr_appelle_le_provider_directement(tmp_path, monkeypatch):
    """Avec --ocr, une image passe TOUJOURS par l'OCR — le provider reçoit directement
    le chemin image, jamais de markitdown consulté au préalable (pas de seuil de
    caractères : contrairement à un convertible muet, il n'y a pas de texte à mesurer)."""
    calls = []

    class _ConverterQuiNeDoitJamaisEtreAppele:
        def convert(self, path):
            calls.append(path)
            raise AssertionError("markitdown ne doit jamais être invoqué sur une image")

    monkeypatch.setattr(ingest, "available", lambda: True)
    monkeypatch.setattr(
        ingest, "_get_converter", lambda: _ConverterQuiNeDoitJamaisEtreAppele()
    )
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    provider_calls = []
    monkeypatch.setattr(
        ingest,
        "ocr_provider",
        lambda path: provider_calls.append(path) or "texte océrisé de la plaque",
    )

    f = tmp_path / "plaque.jpg"
    f.write_bytes(b"\xff\xd8\xff\xe0 peu importe le contenu binaire jpeg")

    text = ingest.to_text(f, ocr=True)

    assert text == "texte océrisé de la plaque"
    assert len(calls) == 0
    assert provider_calls == [f]


def test_to_text_image_ocr_sans_provider_leve_erreur_claire(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "RapidOCR", None)
    monkeypatch.setattr(ingest, "_ocr_engine", None)
    f = tmp_path / "plaque.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n peu importe")
    with pytest.raises(ValueError):
        ingest.to_text(f, ocr=True)


def test_to_text_image_trop_lourde_ignoree_meme_avec_ocr_provider_non_appele(
    tmp_path, monkeypatch
):
    """Garde de volume v1.6 §B : une image > 12 Mo est ignorée+comptée (scan aberrant),
    sans même consulter le provider OCR (le seuil est un garde-fou de temps, pas une
    tentative avortée)."""
    provider_calls = []
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(
        ingest,
        "ocr_provider",
        lambda path: provider_calls.append(path) or "ne doit jamais servir",
    )

    f = tmp_path / "scan_geant.jpg"
    with open(f, "wb") as fh:
        fh.seek(12 * 1024 * 1024 + 1)
        fh.write(b"\x00")

    text = ingest.to_text(f, ocr=True)

    assert text is None
    assert len(provider_calls) == 0


def test_to_text_image_sous_le_seuil_de_volume_ocr_normalement(tmp_path, monkeypatch):
    """Une image sous 12 Mo n'est jamais concernée par la garde de volume : le provider
    est bien consulté."""
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(ingest, "ocr_provider", lambda path: "texte océrisé")

    f = tmp_path / "petite.jpeg"
    f.write_bytes(b"\xff\xd8\xff\xe0" * 100)  # bien en dessous de 12 Mo

    assert ingest.to_text(f, ocr=True) == "texte océrisé"


def test_to_text_image_provider_echoue_retombe_sur_none_sans_crash(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(ingest, "ocr_provider", lambda path: None)
    f = tmp_path / "plaque.tiff"
    f.write_bytes(b"peu importe")
    assert ingest.to_text(f, ocr=True) is None


# -- Index.build/add : intégration -----------------------------------------------------------


def test_build_mixte_md_et_pdf_reel(tmp_path):
    pytest.importorskip("markitdown")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text(
        "pose interrupteur différentiel tableau", encoding="utf-8"
    )
    (corpus / "devis.pdf").write_bytes(
        _minimal_pdf("pose interrupteur differentiel protection")
    )

    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)

    assert set(idx.ids) == {"note.md", "devis.pdf"}
    assert idx.stats()["docs"] == 2
    assert idx.stats()["ignores"] == 0
    hits = idx.search("interrupteur différentiel", k=2)
    assert {h["id"] for h in hits} == {"note.md", "devis.pdf"}


def test_build_fichier_convertible_corrompu_ignore_et_compte(tmp_path):
    pytest.importorskip("markitdown")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text(
        "pose interrupteur différentiel tableau", encoding="utf-8"
    )
    (corpus / "corrompu.pdf").write_bytes(
        bytes([0xFF, 0x00, 0xDE, 0xAD, 0xBE, 0xEF]) * 50
    )

    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)

    assert idx.ids == ["note.md"]
    assert idx.stats()["docs"] == 1
    assert idx.stats()["ignores"] == 1


def test_ignores_survit_a_open(tmp_path):
    """stats()["ignores"] doit rester correct après une ré-ouverture (persisté dans le meta)."""
    pytest.importorskip("markitdown")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text(
        "pose interrupteur différentiel tableau", encoding="utf-8"
    )
    (corpus / "corrompu.pdf").write_bytes(
        bytes([0xFF, 0x00, 0xDE, 0xAD, 0xBE, 0xEF]) * 50
    )

    Index.build(corpus, tmp_path / "idx", grid=GRID)
    reopened = Index.open(tmp_path / "idx")

    assert reopened.stats()["ignores"] == 1
    assert reopened.stats()["docs"] == 1


def test_build_sans_markitdown_ignore_convertibles_et_compte(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "MarkItDown", None)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text(
        "pose interrupteur différentiel tableau", encoding="utf-8"
    )
    (corpus / "devis.pdf").write_bytes(b"peu importe, jamais lu sans markitdown")
    (corpus / "compta.xlsx").write_bytes(b"peu importe non plus")

    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)

    assert idx.ids == ["note.md"]
    assert idx.stats()["docs"] == 1
    assert idx.stats()["ignores"] == 2


def test_build_ids_conservent_extension_origine(tmp_path):
    pytest.importorskip("markitdown")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    sub = corpus / "chantier"
    sub.mkdir()
    (sub / "devis.pdf").write_bytes(_minimal_pdf("fourniture pose cablage"))
    (corpus / "note.md").write_text("texte simple", encoding="utf-8")

    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)

    assert "chantier/devis.pdf" in idx.ids
    assert "note.md" in idx.ids


def test_add_pdf_reel_convertible(tmp_path):
    pytest.importorskip("markitdown")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text(
        "pose interrupteur différentiel tableau", encoding="utf-8"
    )
    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)

    extra = tmp_path / "devis.pdf"
    extra.write_bytes(_minimal_pdf("chauffe eau ballon raccordement cuivre"))
    idx.add(extra)

    assert "devis.pdf" in idx.ids
    assert idx.stats()["docs"] == 2


def test_cache_ingestion_reutilise_lors_du_build(tmp_path, monkeypatch):
    """Le 2e build (même corpus, même cache_dir) doit relire le cache disque, pas reconvertir.

    Le convertisseur compteur est un objet frais dédié au test, indépendant du singleton
    module-level `ingest._converter` — aucune mutation partagée entre tests (isolation).
    """
    pytest.importorskip("markitdown")
    from markitdown import MarkItDown

    calls = []
    real_converter = MarkItDown()  # instance indépendante, jamais le singleton partagé

    class _CountingConverter:
        def convert(self, path, **kwargs):
            calls.append(path)
            return real_converter.convert(path, **kwargs)

    fake_converter = _CountingConverter()
    monkeypatch.setattr(ingest, "_get_converter", lambda: fake_converter)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "devis.pdf").write_bytes(_minimal_pdf("cablage tableau protection"))
    cache_dir = tmp_path / "cache"

    Index.build(corpus, tmp_path / "idx1", grid=GRID, ingest_cache_dir=cache_dir)
    Index.build(corpus, tmp_path / "idx2", grid=GRID, ingest_cache_dir=cache_dir)

    assert len(calls) == 1  # 2e build : cache disque réutilisé, pas de reconversion


# -- Index.build/add : OCR (v1.5) --------------------------------------------------------------


def test_build_ocr_true_sans_provider_leve_valueerror(tmp_path, monkeypatch):
    """Sans moteur OCR (simulé) : `Index.build(..., ocr=True)` sur un corpus
    contenant un document muet doit échouer clairement (fail-fast), pas produire
    un index tronqué silencieusement."""
    pytest.importorskip("markitdown")
    monkeypatch.setattr(ingest, "RapidOCR", None)
    monkeypatch.setattr(ingest, "_ocr_engine", None)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "scan.pdf").write_bytes(_minimal_pdf("bref"))
    with pytest.raises(ValueError):
        Index.build(corpus, tmp_path / "idx", grid=GRID, ocr=True)


def test_build_ocr_true_avec_provider_factice_indexe_le_texte_ocr(
    tmp_path, monkeypatch
):
    pytest.importorskip("markitdown")
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(
        ingest,
        "ocr_provider",
        lambda path: "chauffe eau ballon raccordement cuivre tuyauterie sanitaire " * 5,
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "scan.pdf").write_bytes(_minimal_pdf("bref"))

    idx = Index.build(corpus, tmp_path / "idx", grid=GRID, ocr=True)

    assert idx.ids == ["scan.pdf"]
    assert idx.ocr is True
    hits = idx.search("chauffe eau ballon", k=1)
    assert hits[0]["id"] == "scan.pdf"


def test_build_ocr_false_defaut(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text(
        "pose interrupteur différentiel tableau", encoding="utf-8"
    )
    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)
    assert idx.ocr is False


def test_ocr_meta_round_trip_et_applique_a_add(tmp_path, monkeypatch):
    """`ocr` persisté au meta : réouvrir puis add() doit continuer d'appliquer l'OCR
    sans qu'on ait besoin de repasser le drapeau (cohérence build <-> add)."""
    pytest.importorskip("markitdown")
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(
        ingest,
        "ocr_provider",
        lambda path: "plomberie chauffe eau ballon cuivre " * 5,
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text(
        "interrupteur différentiel tableau", encoding="utf-8"
    )

    idx = Index.build(corpus, tmp_path / "idx", grid=GRID, ocr=True)
    assert idx.ocr is True
    idx2 = Index.open(tmp_path / "idx")
    assert idx2.ocr is True

    extra = tmp_path / "scan2.pdf"
    extra.write_bytes(_minimal_pdf("bref"))
    idx2.add(extra)  # aucun paramètre ocr ici : suit self.ocr persisté

    hits = idx2.search("plomberie chauffe eau", k=1)
    assert hits[0]["id"] == "scan2.pdf"


def test_build_image_sans_ocr_ignoree_et_comptee(tmp_path):
    """v1.6 §B : sans --ocr, une image est ignorée+comptée comme un convertible sans
    markitdown — jamais indexée sous un texte de conversion inexistant."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text(
        "pose interrupteur différentiel tableau", encoding="utf-8"
    )
    (corpus / "plaque.jpg").write_bytes(
        b"\xff\xd8\xff\xe0 peu importe le contenu binaire jpeg"
    )

    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)

    assert idx.ids == ["note.md"]
    assert idx.stats()["docs"] == 1
    assert idx.stats()["ignores"] == 1


def test_build_image_avec_ocr_indexee_via_provider_factice(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(
        ingest,
        "ocr_provider",
        lambda path: "disjoncteur differentiel 30ma tableau protection type ac " * 5,
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "plaque.jpg").write_bytes(
        b"\xff\xd8\xff\xe0 peu importe le contenu binaire jpeg"
    )

    idx = Index.build(corpus, tmp_path / "idx", grid=GRID, ocr=True)

    assert idx.ids == ["plaque.jpg"]
    assert idx.stats()["docs"] == 1
    assert idx.stats()["ignores"] == 0
    hits = idx.search("disjoncteur differentiel 30ma", k=1)
    assert hits[0]["id"] == "plaque.jpg"


def test_build_image_trop_lourde_ignoree_et_comptee_meme_avec_ocr(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(ingest, "ocr_provider", lambda path: "ne doit jamais servir")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    geant = corpus / "scan_geant.jpg"
    with open(geant, "wb") as fh:
        fh.seek(12 * 1024 * 1024 + 1)
        fh.write(b"\x00")

    idx = Index.build(corpus, tmp_path / "idx", grid=GRID, ocr=True)

    assert idx.ids == []
    assert idx.stats()["ignores"] == 1


def test_add_image_suit_ocr_persiste_du_build(tmp_path, monkeypatch):
    """Cohérence build <-> add (même principe que les PDF v1.5) : un index construit
    avec ocr=True continue d'appliquer l'OCR aux images ajoutées via add(), sans
    repasser le drapeau."""
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(
        ingest,
        "ocr_provider",
        lambda path: "cablage tableau protection differentiel " * 5,
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("interrupteur tableau", encoding="utf-8")
    idx = Index.build(corpus, tmp_path / "idx", grid=GRID, ocr=True)

    extra = tmp_path / "plaque2.png"
    extra.write_bytes(b"\x89PNG\r\n\x1a\n peu importe")
    idx.add(extra)

    assert "plaque2.png" in idx.ids
    hits = idx.search("cablage tableau protection", k=1)
    assert hits[0]["id"] == "plaque2.png"


def test_add_image_sans_ocr_ignoree_et_comptee(tmp_path):
    """Index construit sans OCR (self.ocr == False) : add() sur une image doit
    l'ignorer+compter, jamais l'indexer avec un texte de conversion inexistant."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("interrupteur tableau", encoding="utf-8")
    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)
    assert idx.ocr is False

    extra = tmp_path / "plaque.jpg"
    extra.write_bytes(b"\xff\xd8\xff\xe0 peu importe le contenu binaire jpeg")
    idx.add(extra)

    assert "plaque.jpg" not in idx.ids
    assert idx.stats()["ignores"] == 1


def test_open_index_legacy_ocr_par_defaut_false(tmp_path):
    """Un index sur disque sans clé `ocr` a été construit avant l'existence du
    levier : la ré-ouverture ne doit jamais se mettre à exiger un moteur OCR
    qu'un `add()` ultérieur n'avait pas au build."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text(
        "interrupteur différentiel tableau", encoding="utf-8"
    )
    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)

    from mosaic.store import load_vocab, save_vocab

    p, colloc, lex, meta = load_vocab(tmp_path / "idx")
    meta.pop("ocr", None)
    meta.pop("legacy", None)
    save_vocab(
        tmp_path / "idx",
        p,
        colloc,
        idx.grid,
        lex,
        extra_meta={"profile_weighting": "ppmi", "smoothing_rank": 300},
    )
    idx2 = Index.open(tmp_path / "idx")
    assert idx2.ocr is False


# -- CLI ---------------------------------------------------------------------------------------


def test_cli_build_cache_ingestion_ecrit_dans_le_cache_configure(tmp_path, monkeypatch):
    pytest.importorskip("markitdown")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "devis.pdf").write_bytes(_minimal_pdf("cablage tableau protection"))

    cache_root = tmp_path / "cache_isole"
    monkeypatch.setenv("MOSAIC_INGEST_CACHE", str(cache_root))
    r = subprocess.run(
        [
            *_PY,
            "build",
            str(corpus),
            "-o",
            str(tmp_path / "idx"),
            "--grid",
            "32x32",
            "--cache-ingestion",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    stats = json.loads(r.stdout)
    assert stats["docs"] == 1 and stats["ignores"] == 0

    assert cache_root.is_dir()
    assert list(cache_root.glob("*.txt"))  # cache opt-in créé sous la racine configurée
    assert not any(
        (corpus).glob("*.txt")
    )  # jamais un fichier converti déposé à côté des documents


def test_cli_build_sans_cache_ingestion_ne_cree_rien(tmp_path, monkeypatch):
    pytest.importorskip("markitdown")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "devis.pdf").write_bytes(_minimal_pdf("cablage tableau protection"))

    cache_root = tmp_path / "cache_isole"
    monkeypatch.setenv("MOSAIC_INGEST_CACHE", str(cache_root))
    before = set(cache_root.glob("*.txt")) if cache_root.is_dir() else set()

    r = subprocess.run(
        [*_PY, "build", str(corpus), "-o", str(tmp_path / "idx"), "--grid", "32x32"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr

    after = set(cache_root.glob("*.txt")) if cache_root.is_dir() else set()
    assert after == before  # opt-in strict : rien n'apparaît sans --cache-ingestion


# -- Moteur OCR réel sur une image (v1.6 §B) -----------------------------------------------------


def test_to_text_image_moteur_reel_lit_du_texte_imprime(tmp_path):
    """Bout en bout avec le vrai moteur (rapidocr, sans monkeypatch) : une image PNG
    contenant du texte imprimé net doit produire du texte indexable. Sauté si rapidocr
    n'est pas installé dans cet environnement (extra `ocr`)."""
    pytest.importorskip("rapidocr")
    PIL = pytest.importorskip("PIL.Image")
    from PIL import ImageDraw, ImageFont

    img = PIL.new("RGB", (400, 120), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 30), "DISJONCTEUR 30A", fill="black", font=font)
    f = tmp_path / "etiquette.png"
    img.save(f)

    text = ingest.to_text(f, ocr=True)

    assert text is not None
    assert len(text.strip()) > 0
