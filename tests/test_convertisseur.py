"""Convertisseur d'ingestion alternatif (MOSAIC_CONVERTISSEUR=anydoc, opt-in).

Ce que ces tests protègent : un index doit TOUJOURS savoir par quel convertisseur
son texte a été lu, et refuser de mélanger deux lectures différentes — sans quoi
des documents de natures textuelles distinctes cohabiteraient dans le même espace
sémantique sans le moindre signe.
"""

import pytest

from mosaic import ingest
from mosaic.index import Index

GRID = (16, 16, 3)


@pytest.fixture
def corpus(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "a.md").write_text("disjoncteur différentiel tableau garage", encoding="utf-8")
    (c / "b.md").write_text("câble section conducteur cuivre", encoding="utf-8")
    return c


def test_defaut_inchange(monkeypatch):
    """Sans variable d'environnement : markitdown, comme depuis toujours."""
    monkeypatch.delenv("MOSAIC_CONVERTISSEUR", raising=False)
    assert ingest.convertisseur_demande() == "markitdown"
    assert ingest.convertisseur_effectif() == "markitdown"


def test_valeur_inconnue_refusee(monkeypatch):
    """Une faute de frappe ne doit pas retomber silencieusement sur le défaut."""
    monkeypatch.setenv("MOSAIC_CONVERTISSEUR", "anydocc")
    with pytest.raises(ValueError, match="inconnu"):
        ingest.convertisseur_demande()


def test_anydoc_demande_mais_absent_refuse(monkeypatch):
    """Demander anydoc sans l'avoir installé est un refus NET : un repli muet sur
    markitdown produirait un index étiqueté d'un convertisseur qui ne l'a pas lu."""
    monkeypatch.setenv("MOSAIC_CONVERTISSEUR", "anydoc")
    monkeypatch.setattr(ingest, "anydoc", None)
    with pytest.raises(ValueError, match="firecrawl-anydoc"):
        ingest.convertisseur_effectif()


def test_convertisseur_trace_dans_stats_et_meta(corpus, tmp_path, monkeypatch):
    monkeypatch.delenv("MOSAIC_CONVERTISSEUR", raising=False)
    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)
    assert idx.stats()["convertisseur"] == "markitdown"
    # Au DÉFAUT, aucune clé n'est ajoutée au meta : les index existants gardent
    # exactement le même contenu. (Grep binaire impossible : le mot apparaît dans
    # le lexique embarqué — « buck_converter » -> « convertisseur_abaisseur ».)
    from mosaic.store import load_vocab

    _p, _c, _lex, meta = load_vocab(tmp_path / "idx")
    assert "convertisseur" not in meta


def test_reouverture_garde_le_convertisseur_dorigine(corpus, tmp_path, monkeypatch):
    """Un index lu par markitdown reste « markitdown » même rouvert dans un
    environnement qui demande anydoc — le meta fait foi, jamais l'environnement."""
    monkeypatch.delenv("MOSAIC_CONVERTISSEUR", raising=False)
    Index.build(corpus, tmp_path / "idx", grid=GRID)
    monkeypatch.setenv("MOSAIC_CONVERTISSEUR", "anydoc")
    monkeypatch.setattr(ingest, "anydoc", object())  # simule le paquet installé
    idx = Index.open(tmp_path / "idx")
    assert idx.convertisseur == "markitdown"


def test_add_refuse_un_convertisseur_different(corpus, tmp_path, monkeypatch):
    """La garde qui compte : ajouter un document lu autrement est refusé loud."""
    monkeypatch.delenv("MOSAIC_CONVERTISSEUR", raising=False)
    Index.build(corpus, tmp_path / "idx", grid=GRID)
    idx = Index.open(tmp_path / "idx")
    nouveau = tmp_path / "c.md"
    nouveau.write_text("prise commandée interrupteur", encoding="utf-8")
    monkeypatch.setenv("MOSAIC_CONVERTISSEUR", "anydoc")
    monkeypatch.setattr(ingest, "anydoc", object())
    with pytest.raises(ValueError, match="add\\(\\) refusé"):
        idx.add(nouveau)


def test_cle_de_cache_distingue_les_convertisseurs(tmp_path, monkeypatch):
    """Sans le convertisseur dans la clé, basculer vers anydoc resservait le texte
    markitdown mis en cache — un index « anydoc » bâti sur du texte markitdown."""
    f = tmp_path / "doc.md"
    f.write_text("contenu", encoding="utf-8")
    monkeypatch.delenv("MOSAIC_CONVERTISSEUR", raising=False)
    cle_markitdown = ingest._cache_key(f)
    monkeypatch.setenv("MOSAIC_CONVERTISSEUR", "anydoc")
    assert ingest._cache_key(f) != cle_markitdown


def test_aiguillage_appelle_anydoc(tmp_path, monkeypatch):
    """to_text route bien vers anydoc quand il est demandé, sans passer par markitdown."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 factice")
    monkeypatch.setenv("MOSAIC_CONVERTISSEUR", "anydoc")
    faux = type(
        "M", (), {"to_markdown": staticmethod(lambda p: "| a | b |\n| 1 | 2 |")}
    )
    monkeypatch.setattr(ingest, "anydoc", faux)
    assert ingest.to_text(pdf) == "| a | b |\n| 1 | 2 |"


def test_refus_anydoc_tombe_sur_locr(tmp_path, monkeypatch):
    """Un PDF scanné qu'anydoc refuse doit atteindre le crochet OCR — même chemin
    qu'un markitdown muet, aucun cas nouveau."""
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 scanne")
    monkeypatch.setenv("MOSAIC_CONVERTISSEUR", "anydoc")

    def refuse(_p):
        raise RuntimeError("PDF has no extractable text: OCR is required")

    monkeypatch.setattr(
        ingest, "anydoc", type("M", (), {"to_markdown": staticmethod(refuse)})
    )
    monkeypatch.setattr(ingest, "available_ocr", lambda: True)
    monkeypatch.setattr(ingest, "ocr_provider", lambda _p: "texte issu de l'ocr" * 20)
    assert "ocr" in (ingest.to_text(pdf, ocr=True) or "")
