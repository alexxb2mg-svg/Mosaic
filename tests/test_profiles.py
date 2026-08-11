import numpy as np

from mosaic.profiles import Profiles
from mosaic.signatures import signature

DIM = 3072  # dimension compacte pour la vitesse des tests


def test_learn_accumule_les_voisins():
    p = Profiles(DIM)
    p.learn(["disjoncteur", "différentiel"])
    p.finalize("brut")
    # d=1 → poids 5 : chacun reçoit 5 × la signature de l'autre
    row = p.acc[p.rows["disjoncteur"]]
    assert np.array_equal(row, 5 * signature("différentiel", DIM))


def test_finalize_brut_identique_v1():
    # même corpus → mêmes profils que l'ancienne accumulation directe (vérification
    # par valeur : construire à la main 5*sig(voisin) comme dans test_learn_accumule_les_voisins)
    p = Profiles(DIM)
    p.learn(["disjoncteur", "différentiel"])
    p.finalize("brut")
    assert np.allclose(
        p.acc[p.rows["disjoncteur"]], 5.0 * signature("différentiel", DIM)
    )


def test_ppmi_reduit_le_voisin_frequent():
    # « pose » cooccurre avec TOUT ; « différentiel » seulement avec « interrupteur ».
    # En PPMI, le profil d'« interrupteur » doit être plus proche de sig(différentiel)
    # que de sig(pose), alors qu'en brut c'est comparable (poids de paires identiques).
    p = Profiles(DIM)
    for mot in ("interrupteur", "câble", "tableau", "gaine"):
        for _ in range(6):
            p.learn(["pose", mot])
    for _ in range(6):
        p.learn(["interrupteur", "différentiel"])

    def _cos(a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        return float(a @ b)

    p.finalize("brut")
    brut_pose = _cos(
        p.acc[p.rows["interrupteur"]], signature("pose", DIM).astype(np.float32)
    )
    brut_diff = _cos(
        p.acc[p.rows["interrupteur"]], signature("différentiel", DIM).astype(np.float32)
    )
    p.finalize("ppmi")
    ppmi_pose = _cos(
        p.acc[p.rows["interrupteur"]], signature("pose", DIM).astype(np.float32)
    )
    ppmi_diff = _cos(
        p.acc[p.rows["interrupteur"]], signature("différentiel", DIM).astype(np.float32)
    )
    assert brut_pose >= brut_diff  # le fréquent ne recule pas en brut (sanity)
    assert ppmi_diff > ppmi_pose  # PPMI inverse la dominance


def test_ppmi_independant_de_l_ordre():
    docs = [
        ["pose", "interrupteur", "mural"],
        ["interrupteur", "différentiel", "protection"],
        ["pose", "câble", "cuivre"],
        ["différentiel", "tableau", "protection"],
    ]
    a, b = Profiles(DIM), Profiles(DIM)
    for d in docs:
        a.learn(d)
    for d in reversed(docs):
        b.learn(d)
    a.finalize("ppmi")
    b.finalize("ppmi")
    for token in ("pose", "interrupteur", "différentiel", "protection"):
        assert np.array_equal(a.acc[a.rows[token]], b.acc[b.rows[token]])


def test_stopwords_ignores_dans_les_profils():
    p = Profiles(DIM)
    p.learn(["tableau", "de", "chantier"])
    assert "de" not in p.rows
    p.finalize("brut")
    # "de" est transparent : tableau↔chantier restent voisins (d=2 → poids 4)
    row = p.acc[p.rows["tableau"]]
    assert np.array_equal(row, 4 * signature("chantier", DIM))


def test_independance_de_l_ordre_des_documents():
    a, b = Profiles(DIM), Profiles(DIM)
    d1, d2 = ["cable", "cuivre", "section"], ["cuivre", "prix", "cable"]
    a.learn(d1)
    a.learn(d2)
    b.learn(d2)
    b.learn(d1)
    a.finalize("brut")
    b.finalize("brut")
    assert np.array_equal(a.acc[a.rows["cuivre"]], b.acc[b.rows["cuivre"]])
    assert a.df == b.df and a.n_docs == b.n_docs


def test_word_vector_unitaire_et_mot_inconnu():
    p = Profiles(DIM)
    p.learn(["cable", "cuivre"])
    p.finalize("brut")
    v = p.word_vector("cable")
    assert v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
    inconnu = p.word_vector("zzyzx")
    assert abs(float(np.linalg.norm(inconnu)) - 1.0) < 1e-5


def test_mots_de_contextes_proches_se_ressemblent():
    p = Profiles(DIM)
    for _ in range(30):
        p.learn(["pose", "interrupteur", "mural", "salon"])
        p.learn(["pose", "commutateur", "mural", "salon"])
        p.learn(["achat", "carrelage", "gris", "cuisine"])
    p.finalize("brut")
    sim_syn = float(p.word_vector("interrupteur") @ p.word_vector("commutateur"))
    sim_far = float(p.word_vector("interrupteur") @ p.word_vector("carrelage"))
    assert sim_syn > sim_far + 0.2


def test_idf_decroit_avec_la_frequence_documentaire():
    p = Profiles(DIM)
    p.learn(["cable", "cuivre"])
    p.learn(["cable", "gaine"])
    assert p.idf("cuivre") > p.idf("cable")


def test_finalize_alloue_toutes_les_lignes_sans_perte():
    """Régression : finalize() matérialise l'acc à la bonne taille finale (V×dim)
    en une fois, pour tout le vocabulaire découvert pendant learn() — plus de
    croissance incrémentale pendant l'apprentissage (déplacée côté finalize)."""
    p = Profiles(DIM)
    # 6 tokens distincts non-stopwords
    tokens = ["cable", "cuivre", "section", "gaine", "prix", "pose"]
    p.learn(tokens)

    # Rien n'est matérialisé avant finalize()
    assert p.acc.shape == (0, DIM)

    p.finalize("brut")

    # Vérifier que tous les tokens sont dans rows et que l'acc a la bonne forme
    assert len(p.rows) == 6
    assert p.acc.shape == (6, DIM)
    for token in tokens:
        assert token in p.rows

    # Vérifier que le premier et le dernier token ont des accumulateurs non-nuls
    first_row = p.acc[p.rows["cable"]]
    last_row = p.acc[p.rows["pose"]]
    assert first_row.any(), "Premier token devrait avoir un accumulateur non-nul"
    assert last_row.any(), "Dernier token devrait avoir un accumulateur non-nul"

    # Calculer l'accumulation attendue : "cable" reçoit les signatures
    # pondérées de ses voisins dans la fenêtre (d=1..5, w=5..1)
    expected_cable = (
        5 * signature("cuivre", DIM)
        + 4 * signature("section", DIM)
        + 3 * signature("gaine", DIM)
        + 2 * signature("prix", DIM)
        + 1 * signature("pose", DIM)
    )
    actual_cable = p.acc[p.rows["cable"]]
    assert np.array_equal(actual_cable, expected_cable), (
        "Accumulateur 'cable' incorrect après finalize()"
    )


def test_word_vector_sans_embed_identique_v1():
    p = Profiles(DIM)
    p.learn(["cable", "cuivre"])
    p.finalize("brut")
    sig = signature("cable", DIM).astype(np.float32)
    sig /= np.float32(np.linalg.norm(sig))
    prof = p.acc[p.rows["cable"]].astype(np.float32)
    prof /= np.float32(np.linalg.norm(prof))
    attendu = 0.5 * sig + 0.5 * prof
    attendu /= np.float32(np.linalg.norm(attendu))
    assert np.allclose(p.word_vector("cable"), attendu, atol=1e-6)


def test_word_vector_avec_embed_rapproche_les_synonymes():
    p = Profiles(DIM)
    p.finalize("brut")
    rng = np.random.default_rng(42)
    shared = rng.normal(size=DIM).astype(np.float32)
    shared /= np.linalg.norm(shared)
    va = p.word_vector("interrupteur", embed=shared)
    vb = p.word_vector("commutateur", embed=shared)
    # embeddings identiques, signatures orthogonales → cos ≈ γ_eff²
    assert float(va @ vb) > 0.3
    assert float(va @ p.word_vector("carrelage")) < 0.1


def test_finalize_brut_multi_occurrences_identique_v1():
    p = Profiles(DIM)
    p.learn(["cable", "cuivre"])  # paire (cable,cuivre) w=5
    p.learn(["cuivre", "cable"])  # même paire, encore w=5
    p.learn(["cable", "cable"])  # self-pair w=5
    p.finalize("brut")
    attendu_cable = (
        10.0 * signature("cuivre", DIM).astype(np.float32)  # 2 occurrences × w=5
        + 10.0
        * signature("cable", DIM).astype(np.float32)  # self-pair : double écriture 2×5
    )
    assert np.array_equal(p.acc[p.rows["cable"]], attendu_cable)


def test_word_vector_avant_finalize_leve_runtimeerror_sur_token_connu():
    p = Profiles(DIM)
    p.learn(["cable", "cuivre"])
    try:
        p.word_vector("cable")
        raise AssertionError("aurait dû lever RuntimeError")
    except RuntimeError:
        pass
    # mot jamais appris : retombe sur signature seule, pas d'erreur
    v = p.word_vector("zzyzx")
    assert v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_finalize_weighting_invalide_leve_valueerror():
    p = Profiles(DIM)
    p.learn(["cable", "cuivre"])
    try:
        p.finalize("inconnu")
        raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        pass
