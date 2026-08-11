from mosaic.collocations import detect, merge


def _corpus(pair: list[str], n: int, filler: str) -> list[list[str]]:
    # n docs contenant la paire + 24 mots de fond uniques par doc :
    # PMI(pair) = ln(n · total / (n · n)) = ln(total / n) = ln(26) ≈ 3.26 ≥ 3.0
    return [pair + [f"{filler}{i}_{k}" for k in range(24)] for i in range(n)]


def test_detecte_une_collocation_frequente():
    docs = _corpus(["interrupteur", "différentiel"], 8, "mot")
    assert ("interrupteur", "différentiel") in detect(docs)


def test_ignore_les_paires_rares():
    docs = _corpus(["interrupteur", "différentiel"], 3, "mot")  # < min_count
    assert detect(docs) == set()


def test_ignore_les_paires_de_stopwords():
    # ("de", "la") : count 8 ≥ 5 et PMI = ln(26) ≈ 3.26 ≥ 3.0 —
    # seule l'exclusion stopword peut la rejeter.
    docs = [["de", "la"] + [f"fond{i}_{k}" for k in range(24)] for i in range(8)]
    assert detect(docs) == set()


def test_merge_fusionne_gauche_droite():
    colloc = {("interrupteur", "différentiel")}
    assert merge(["pose", "interrupteur", "différentiel", "30ma"], colloc) == [
        "pose",
        "interrupteur_différentiel",
        "30ma",
    ]


def test_merge_sans_collocation_est_identite():
    assert merge(["a", "b", "c"], set()) == ["a", "b", "c"]
