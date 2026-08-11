"""Fonctions pures de bench/alloprof.py — conversion Alloprof -> format du banc.

Hors-ligne strict : le téléchargement (_fetch_*) n'est jamais exercé ici, seule
la logique de conversion l'est — un banc dont la préparation est fausse mesure
faux en silence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from alloprof import document_markdown, requetes_bench


def test_document_markdown_titre_en_entete():
    row = {"title": "Le sens concret", "text": "On oppose le concret et l'abstrait."}
    assert document_markdown(row) == (
        "# Le sens concret\n\nOn oppose le concret et l'abstrait.\n"
    )


def test_document_markdown_sans_titre_garde_le_texte_nu():
    assert document_markdown({"title": "  ", "text": "corps seul"}) == "corps seul\n"


def test_requetes_bench_mappe_uuid_vers_fichier_md():
    lignes, ecartees = requetes_bench(
        [{"text": "question ?", "relevant": ["abc", "def"]}], {"abc", "def"}
    )
    assert ecartees == 0
    assert lignes == [{"query": "question ?", "relevant": ["abc.md", "def.md"]}]


def test_requetes_bench_filtre_les_pertinents_inconnus_ligne_a_ligne():
    lignes, ecartees = requetes_bench(
        [{"text": "question ?", "relevant": ["abc", "fantome"]}], {"abc"}
    )
    assert ecartees == 0
    assert lignes == [{"query": "question ?", "relevant": ["abc.md"]}]


def test_requetes_bench_ecarte_requete_sans_aucun_pertinent_ou_sans_texte():
    lignes, ecartees = requetes_bench(
        [
            {"text": "orpheline ?", "relevant": ["fantome"]},
            {"text": "   ", "relevant": ["abc"]},
            {"text": "valide ?", "relevant": ["abc"]},
        ],
        {"abc"},
    )
    assert ecartees == 2
    assert [ligne["query"] for ligne in lignes] == ["valide ?"]
