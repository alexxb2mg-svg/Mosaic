"""Tests des filtres d'import WikDict (§B2) sur une source synthétique.

Note : la spec brief anticipait un "mini-TSV synthétique" — WikDict ne publie en
réalité aucun format TSV (seuls sqlite/stardict/tei/kobo/wdweb existent, voir
scripts/import_wikdict.py pour le détail). La source réelle retenue est SQLite
(stdlib, zéro dépendance) ; la source synthétique ci-dessous reproduit donc le
schéma réel de la table `simple_translation` plutôt qu'un TSV.
"""

import sqlite3

from import_wikdict import (
    EXCLUDED_WORDS,
    best_translation,
    clean_tokens,
    filter_wikdict,
    read_simple_translation,
)

CORE = {"breaker": "disjoncteur", "wire": "fil_électrique"}


def _make_wikdict_db(tmp_path, rows):
    """Reproduit le schéma réel `simple_translation` de WikDict dans un fichier temporaire."""
    db_path = tmp_path / "mini_wikdict.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE simple_translation "
        "(written_rep TEXT, trans_list, max_score, rel_importance)"
    )
    conn.executemany(
        "INSERT INTO simple_translation (written_rep, trans_list, max_score, rel_importance) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


def test_best_translation_prend_le_premier_candidat_ex_aequo():
    assert best_translation("clé | clef") == "clé"
    assert best_translation("disjoncteur") == "disjoncteur"


def test_clean_tokens_rejette_hors_charset_et_trop_long():
    assert clean_tokens("junction box") == ["junction", "box"]
    assert clean_tokens("un deux trois quatre") == ["un", "deux", "trois", "quatre"]
    assert clean_tokens("un deux trois quatre cinq") is None  # 5 tokens > 4
    assert (
        clean_tokens("'s-Hertogenbosch") is None
    )  # apostrophe : hors charset tokenizer
    assert clean_tokens("#MeToo") is None  # symbole
    assert clean_tokens("") is None


def test_read_simple_translation_lit_la_source_sqlite(tmp_path):
    db_path = _make_wikdict_db(
        tmp_path,
        [("wire", "fil | câble", 100, 0.5), ("gadget", "gadget", 50, 0.1)],
    )
    rows = read_simple_translation(db_path, min_score=0)
    assert ("gadget", "gadget", 50) in rows
    assert ("wire", "fil | câble", 100) in rows


def test_read_simple_translation_applique_le_seuil_de_confiance(tmp_path):
    db_path = _make_wikdict_db(
        tmp_path,
        [("wire", "fil", 200, 0.5), ("rare", "rare", 2, 0.0)],
    )
    rows = read_simple_translation(db_path, min_score=100)
    assert rows == [("wire", "fil", 200)]


def test_filtre_exclusion_mots_ambigus():
    rows = [("run", "courir | marcher", 200), ("gadget", "gadget", 50)]
    result = filter_wikdict(rows, core_lexicon={})
    assert "run" not in result
    assert result["gadget"] == "gadget"
    assert "run" in EXCLUDED_WORDS


def test_filtre_priorite_noyau_curate():
    rows = [
        ("breaker", "casseur | concasseuse", 100),  # dans le noyau -> rejeté
        ("spanner", "clé | clef", 80),  # absent du noyau -> gardé
    ]
    result = filter_wikdict(rows, core_lexicon=CORE)
    assert "breaker" not in result  # le noyau gagne, pas écrasé par WikDict
    assert result["spanner"] == "clé"  # meilleur candidat = premier de la liste


def test_filtre_longueur_cle_et_valeur():
    rows = [
        ("junction box", "boîte de dérivation", 90),  # 2 et 3 tokens -> ok
        ("un deux trois quatre cinq", "court", 10),  # clé 5 tokens -> rejeté
        (
            "widget",
            "un long resultat de test invalide vraiment",
            10,
        ),  # valeur 7 tokens -> rejeté
    ]
    result = filter_wikdict(rows, core_lexicon={})
    assert result["junction_box"] == "boîte_de_dérivation"
    assert "un_deux_trois_quatre_cinq" not in result
    assert "widget" not in result


def test_filtre_caracteres_hors_charset_tokenizer():
    rows = [
        ("#MeToo", "mouvement", 5),  # symbole -> rejeté
        ("'s-Hertogenbosch", "Bois-le-Duc", 5),  # apostrophe -> rejeté
        ("gadget", "gadget", 50),
    ]
    result = filter_wikdict(rows, core_lexicon={})
    assert "metoo" not in result
    assert result == {"gadget": "gadget"}


def test_determinisme_premiere_occurrence_gagne_sur_collision_de_casse():
    # ORDER BY written_rep ASC : "Wire" (0x57) < "wire" (0x77) en tri binaire
    rows = [("Wire", "fil", 100), ("wire", "câble", 100)]
    result = filter_wikdict(rows, core_lexicon={})
    assert result["wire"] == "fil"


def test_filtre_homographe_mot_francais_rejete_sauf_identite():
    rows = [
        (
            "fin",
            "nageoire",
            150,
        ),  # "fin" (anglais, nageoire) mais "fin" est aussi un mot FR courant
        (
            "something",
            "fin",
            150,
        ),  # "fin" apparaît ailleurs côté valeurs FR -> empoisonne le mot-clé
        (
            "breaker",
            "disjoncteur",
            150,
        ),  # clé anglaise absente des valeurs FR -> gardée
        (
            "cardigan",
            "cardigan",
            150,
        ),  # identité clé == valeur -> inerte, toujours gardée
    ]
    result = filter_wikdict(rows, core_lexicon={})
    assert "fin" not in result
    assert result["breaker"] == "disjoncteur"
    assert result["cardigan"] == "cardigan"


def test_pipeline_complet_sur_source_sqlite(tmp_path):
    db_path = _make_wikdict_db(
        tmp_path,
        [
            ("run", "courir", 200, 1.0),  # exclu
            ("breaker", "casseur", 100, 0.5),  # dans le noyau
            ("spanner", "clé | clef", 80, 0.3),  # gardé
            ("gadget", "gadget", 50, 0.1),  # gardé
        ],
    )
    rows = read_simple_translation(db_path, min_score=0)
    result = filter_wikdict(rows, core_lexicon=CORE)
    assert result == {"spanner": "clé", "gadget": "gadget"}
