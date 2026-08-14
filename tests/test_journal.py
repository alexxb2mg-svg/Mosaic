"""Le journal des recherches : il n'écrit rien sans qu'on le demande, il n'échoue jamais
la recherche, et il consigne assez pour que les quatre signaux soient déductibles."""

import json

import pytest

from mosaic import journal
from mosaic.index import Index

GRID = (16, 16, 3)


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "elec1.md").write_text(
        "disjoncteur différentiel protection circuit tableau", encoding="utf-8"
    )
    (d / "elec2.md").write_text(
        "interrupteur différentiel 30mA protection personnes", encoding="utf-8"
    )
    (d / "plomb.md").write_text(
        "chauffe-eau ballon raccordement cuivre sanitaire", encoding="utf-8"
    )
    return d


def test_rien_par_defaut(corpus, tmp_path, monkeypatch):
    """Le défaut est le silence : un dépôt public ne doit pas se mettre à écrire les
    requêtes de quelqu'un qui n'a rien demandé."""
    monkeypatch.delenv(journal.VARIABLE, raising=False)
    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)
    idx.search("disjoncteur", k=2)
    assert list(tmp_path.glob("**/*.jsonl")) == []


def test_consigne_une_ligne_par_recherche(corpus, tmp_path, monkeypatch):
    fichier = tmp_path / "j" / "recherches.jsonl"
    monkeypatch.setenv(journal.VARIABLE, str(fichier))
    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)
    idx.search("disjoncteur différentiel", k=2)
    idx.search("ballon eau chaude", k=2)

    lignes = journal.lire(fichier)
    assert len(lignes) == 2
    assert [li["q"] for li in lignes] == [
        "disjoncteur différentiel",
        "ballon eau chaude",
    ]
    assert all(li["index"] == "idx" for li in lignes)  # le NOM, pas le chemin complet
    assert all(li["k"] == 2 and li["pid"] > 0 and li["t"] for li in lignes)


def test_les_hits_sont_consignes_pour_permettre_le_rejeu(corpus, tmp_path, monkeypatch):
    fichier = tmp_path / "recherches.jsonl"
    monkeypatch.setenv(journal.VARIABLE, str(fichier))
    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)
    attendus = [h["id"] for h in idx.search("chauffe-eau cuivre", k=2)]

    (ligne,) = journal.lire(fichier)
    assert [h["id"] for h in ligne["hits"]] == attendus


def test_seules_les_options_non_par_defaut_sont_ecrites(corpus, tmp_path, monkeypatch):
    """Une ligne de journal doit se relire à l'œil : dix champs à `false` la rendraient
    illisible et personne n'ouvrirait jamais le fichier."""
    fichier = tmp_path / "recherches.jsonl"
    monkeypatch.setenv(journal.VARIABLE, str(fichier))
    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)
    idx.search("disjoncteur", k=2)
    idx.search("disjoncteur", k=2, nettoyer_requete=True)

    nu, nettoye = journal.lire(fichier)
    assert nu["opts"] == {}
    assert nettoye["opts"] == {"nettoyage": True}


def test_les_rangs_par_canal_survivent_en_fusion(corpus, tmp_path, monkeypatch):
    """C'est LE champ qui rend la redondance entre canaux mesurable sans annoter quoi
    que ce soit — si un canal ne remonte jamais rien d'inédit, il ne sert à rien."""
    # --hybride force rerank_vectors, qui exige model2vec : sans lui le job « cœur » de
    # la CI (extras non installés) échouait. Même garde que tests/test_fusion.py.
    pytest.importorskip("model2vec")
    fichier = tmp_path / "recherches.jsonl"
    monkeypatch.setenv(journal.VARIABLE, str(fichier))
    idx = Index.build(corpus, tmp_path / "idx", grid=GRID, hybride=True)
    idx.search("différentiel protection", k=2, fusion=True)

    (ligne,) = journal.lire(fichier)
    assert ligne["opts"] == {"fusion": True}
    assert all("rangs" in h for h in ligne["hits"])
    assert {"grille", "bm25"} <= set(ligne["hits"][0]["rangs"])


def test_une_ecriture_impossible_ne_casse_pas_la_recherche(
    corpus, tmp_path, monkeypatch
):
    """Un journal est un observateur : il n'a pas le droit d'avoir un avis sur le
    déroulement de ce qu'il observe. Ici le chemin est un DOSSIER, donc inouvrable."""
    obstacle = tmp_path / "obstacle"
    obstacle.mkdir()
    monkeypatch.setenv(journal.VARIABLE, str(obstacle))
    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)

    hits = idx.search("disjoncteur différentiel", k=2)

    assert hits and hits[0]["id"] in {"elec1.md", "elec2.md"}


def test_une_ligne_tronquee_ne_perd_pas_le_reste(tmp_path):
    """Une écriture interrompue (coupure, disque plein) laisse une ligne incomplète :
    elle est sautée, elle n'emporte pas l'analyse entière avec elle."""
    fichier = tmp_path / "recherches.jsonl"
    bonne = json.dumps({"q": "intacte", "hits": []}, ensure_ascii=False)
    fichier.write_text(f'{bonne}\n{{"q": "tronq\n{bonne}\n', encoding="utf-8")
    assert [li["q"] for li in journal.lire(fichier)] == ["intacte", "intacte"]


def test_variable_vide_vaut_absente(monkeypatch):
    monkeypatch.setenv(journal.VARIABLE, "   ")
    assert journal.actif() is None


# --- rotation : le journal actif ne grossit pas indéfiniment ---------------------


def test_pas_de_rotation_sous_le_seuil(tmp_path, monkeypatch):
    """Le cas courant ne paie rien : sous le seuil, aucun fichier n'est créé."""
    j = tmp_path / "j.jsonl"
    monkeypatch.setenv("MOSAIC_JOURNAL", str(j))
    monkeypatch.setenv("MOSAIC_JOURNAL_MAX_MO", "1")
    journal.consigner("idx", "question", 5, {}, [{"id": "a", "score": 1.0}])
    assert j.exists()
    assert list(tmp_path.glob("*.gz")) == []


def test_rotation_au_dela_du_seuil(tmp_path, monkeypatch):
    """Au-delà du seuil, le journal est ARCHIVÉ COMPRESSÉ et reprend à vide —
    les lignes ne sont jamais perdues, seulement déplacées."""
    j = tmp_path / "j.jsonl"
    j.write_text('{"t": "vieux", "q": "ancienne question"}\n' * 400, encoding="utf-8")
    monkeypatch.setenv("MOSAIC_JOURNAL", str(j))
    monkeypatch.setenv("MOSAIC_JOURNAL_MAX_MO", "0.001")  # 1 ko : rotation forcée

    journal.consigner("idx", "neuve", 5, {}, [{"id": "a", "score": 1.0}])

    archives = list(tmp_path.glob("j-*.jsonl.gz"))
    assert len(archives) == 1, "l'ancien journal doit être archivé, pas supprimé"
    lignes = journal.lire(j)
    assert len(lignes) == 1 and lignes[0]["q"] == "neuve", "le neuf reprend à vide"

    import gzip

    contenu = gzip.decompress(archives[0].read_bytes()).decode("utf-8")
    assert contenu.count("ancienne question") == 400, "aucune ligne perdue"


def test_une_rotation_impossible_ne_casse_pas_la_recherche(tmp_path, monkeypatch):
    """Même garde que l'écriture : le journal est un témoin, jamais un obstacle."""
    j = tmp_path / "j.jsonl"
    j.write_text("x" * 5000, encoding="utf-8")
    monkeypatch.setenv("MOSAIC_JOURNAL", str(j))
    monkeypatch.setenv("MOSAIC_JOURNAL_MAX_MO", "0.001")
    monkeypatch.setattr(
        journal.gzip, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("disque"))
    )
    journal.consigner("idx", "q", 5, {}, [{"id": "a", "score": 1.0}])  # ne lève pas


def test_seuil_invalide_desactive_la_rotation_sans_bruit(tmp_path, monkeypatch):
    j = tmp_path / "j.jsonl"
    j.write_text("x" * 5000, encoding="utf-8")
    monkeypatch.setenv("MOSAIC_JOURNAL", str(j))
    monkeypatch.setenv("MOSAIC_JOURNAL_MAX_MO", "pas un nombre")
    journal.consigner("idx", "q", 5, {}, [{"id": "a", "score": 1.0}])
    assert list(tmp_path.glob("*.gz")) == []
