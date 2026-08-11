from mosaic.tokenize import STOPWORDS, tokenize


def test_minuscules_et_accents_conserves():
    assert tokenize("Disjoncteur Différentiel") == ["disjoncteur", "différentiel"]


def test_references_techniques():
    assert tokenize("NF C 15-100, calibre 30mA") == [
        "nf",
        "c",
        "15-100",
        "calibre",
        "30ma",
    ]


def test_apostrophes_et_traits_dunion():
    assert tokenize("l'armoire porte-fusible") == ["l", "armoire", "porte-fusible"]


def test_stopwords_connus():
    assert "le" in STOPWORDS and "dans" in STOPWORDS
    assert "à" in STOPWORDS
    assert "disjoncteur" not in STOPWORDS


def test_texte_vide():
    assert tokenize("") == []


def test_apostrophe_curly():
    assert tokenize("l'armoire électrique") == ["l", "armoire", "électrique"]
