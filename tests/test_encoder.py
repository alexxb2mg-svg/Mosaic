import gzip

import numpy as np

from mosaic.embeddings import Embeddings, prepare
from mosaic.encoder import doc_channel, encode
from mosaic.profiles import Profiles

DIM = 3072


def _profiles() -> Profiles:
    p = Profiles(DIM)
    for _ in range(6):
        p.learn(["pose", "interrupteur", "mural"])
        p.learn(["achat", "carrelage", "gris"])
    p.finalize("brut")
    return p


def test_forme_int8_et_norme():
    p = _profiles()
    q, norm = encode(["pose", "interrupteur"], p)
    assert q.dtype == np.int8 and q.shape == (DIM,)
    assert abs(norm - float(np.linalg.norm(q.astype(np.float32)))) < 1e-3
    assert int(np.abs(q.astype(np.int32)).max()) == 127  # pleine échelle


def test_document_vide():
    q, norm = encode([], _profiles())
    assert norm == 0.0 and not q.any()


def test_stopwords_seuls_equivalent_vide():
    q, norm = encode(["de", "la", "le"], _profiles())
    assert norm == 0.0


def test_documents_similaires_plus_proches():
    p = _profiles()
    a, na = encode(["pose", "interrupteur", "mural"], p)
    b, nb = encode(["pose", "interrupteur"], p)
    c, nc = encode(["achat", "carrelage", "gris"], p)
    sim_ab = float(a.astype(np.int32) @ b.astype(np.int32)) / (na * nb)
    sim_ac = float(a.astype(np.int32) @ c.astype(np.int32)) / (na * nc)
    assert sim_ab > sim_ac + 0.2


def test_deterministe():
    p = _profiles()
    q1, _ = encode(["pose", "interrupteur"], p)
    q2, _ = encode(["pose", "interrupteur"], p)
    assert np.array_equal(q1, q2)


def test_negation_inverse_le_sens():
    p = _profiles()
    a, na = encode(["installation", "conforme"], p)
    b, nb = encode(["installation", "pas", "conforme"], p)
    c, nc = encode(["installation", "conforme"], p)
    sim_affirme = float(a.astype(np.int32) @ c.astype(np.int32)) / (na * nc)
    sim_nie = float(a.astype(np.int32) @ b.astype(np.int32)) / (na * nb)
    assert sim_affirme > 0.99  # identiques
    assert sim_nie < 0.35  # « pas conforme » ne ressemble plus à « conforme »


def test_negation_stopword_intermediaire():
    p = _profiles()
    # « pas de terre » : « de » (stopword) ne consomme pas la négation → −v(terre)
    a, na = encode(["prise", "pas", "de", "terre"], p)
    b, nb = encode(["prise", "terre"], p)
    sim = float(a.astype(np.int32) @ b.astype(np.int32)) / (na * nb)
    assert sim < 0.35


def test_negateur_seul_equivalent_vide():
    q, norm = encode(["pas", "sans"], _profiles())
    assert norm == 0.0


def test_negation_sur_corpus_entraine_ne_repousse_pas_le_contexte():
    p = Profiles(DIM)
    for _ in range(10):
        p.learn(["installation", "conforme", "norme", "contrôle"])
        p.learn(["installation", "pas", "conforme", "danger", "reprise"])
        p.learn(["carrelage", "colle", "joint", "sol"])
    p.finalize("brut")
    q, qn = encode(["installation", "pas", "conforme"], p)
    doc_nie, dn = encode(["installation", "pas", "conforme", "danger", "reprise"], p)
    doc_ok, do = encode(["installation", "conforme", "norme", "contrôle"], p)
    doc_hors, dh = encode(["carrelage", "colle", "joint", "sol"], p)
    sim_nie = float(q.astype(np.int32) @ doc_nie.astype(np.int32)) / (qn * dn)
    sim_ok = float(q.astype(np.int32) @ doc_ok.astype(np.int32)) / (qn * do)
    sim_hors = float(q.astype(np.int32) @ doc_hors.astype(np.int32)) / (qn * dh)
    assert sim_nie > sim_ok  # le doc qui contient la phrase niée gagne
    assert sim_nie > sim_hors  # et bat le hors-sujet
    assert sim_nie > 0.3  # attraction franche, pas juste un moindre mal


class _FakeEmb:
    def __init__(self, dim, table):  # table: mot -> graine
        self.dim0 = dim
        self.table = table

    def vector_grid(self, token, grid_dim):
        seed = self.table.get(token)
        if seed is None:
            return None
        rng = np.random.default_rng(seed)
        v = rng.normal(size=grid_dim).astype(np.float32)
        return v / np.float32(np.linalg.norm(v))


def test_encode_avec_embeddings_ponte_la_paraphrase():
    p = _profiles()
    emb = _FakeEmb(
        300, {"panne": 7, "défaillance": 7, "carrelage": 8}
    )  # synonymes → même graine
    a, na = encode(["panne"], p, embeddings=emb)
    b, nb = encode(["défaillance"], p, embeddings=emb)
    c, nc = encode(["carrelage"], p, embeddings=emb)
    sim_syn = float(a.astype(np.int32) @ b.astype(np.int32)) / (na * nb)
    sim_far = float(a.astype(np.int32) @ c.astype(np.int32)) / (na * nc)
    assert sim_syn > sim_far + 0.3


def test_encode_sans_embeddings_inchange():
    p = _profiles()
    q1, n1 = encode(["pose", "interrupteur"], p)
    q2, n2 = encode(["pose", "interrupteur"], p, embeddings=None)
    assert np.array_equal(q1, q2) and n1 == n2


def test_negation_n_inverse_pas_l_embedding():
    p = _profiles()
    emb = _FakeEmb(300, {"conforme": 5})
    # poids explicites (0.35, 0.25, 0.40) : garde la dérivation ≈0.13 valable
    # indépendamment de WEIGHTS_DEFAULT (0.25, 0.15, 0.60, optimum bench v1.1).
    weights = (0.35, 0.25, 0.40)
    a, na = encode(["conforme"], p, embeddings=emb, weights=weights)
    b, nb = encode(["pas", "conforme"], p, embeddings=emb, weights=weights)
    sim = float(a.astype(np.int32) @ b.astype(np.int32)) / (na * nb)
    # signature inversée mais embedding partagé positif :
    # cos ≈ (γ²−α²)/(γ²+α²) ≈ 0.13 — strictement positif, loin de −1
    assert 0.0 < sim < 0.5


def _table_synthetique(tmp_path):
    """Table où « panne » et « défaillance » partagent le même vecteur brut (même graine),
    « carrelage » et « conforme » divergent (graines indépendantes)."""
    lines = ["4 300"]
    for w, seed in (
        ("panne", 7),
        ("défaillance", 7),
        ("carrelage", 99),
        ("conforme", 55),
    ):
        rng = np.random.default_rng(seed)
        lines.append(w + " " + " ".join(f"{v:.4f}" for v in rng.normal(size=300)))
    src = tmp_path / "mini.vec.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    out = tmp_path / "table.msee"
    prepare(src, out)
    return Embeddings.load(out)


def test_canal_document_ignore_negation_et_tokens_hors_table(tmp_path):
    emb = _table_synthetique(tmp_path)
    p = Profiles(DIM)
    p.learn(["conforme"])
    p.finalize("brut")

    result = doc_channel(["pas", "conforme", "zzyzx"], p, emb, DIM)
    assert result is not None
    assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-4

    expected = emb.project_raw(emb.raw_vector("conforme"), DIM)
    expected /= np.float32(np.linalg.norm(expected))
    assert float(result @ expected) > 0.999  # même direction, à la normalisation près


def test_canal_document_rapproche_les_documents_du_meme_theme(tmp_path):
    emb = _table_synthetique(tmp_path)
    p = Profiles(DIM)
    p.learn(["panne", "moteur", "atelier"])
    p.learn(["défaillance", "ligne", "process"])
    p.learn(["carrelage", "colle", "joint"])
    p.finalize("brut")

    doc_a = ["panne", "moteur", "atelier"]
    doc_b = ["défaillance", "ligne", "process"]  # aucun mot commun avec doc_a

    def cos(delta):
        qa, na = encode(doc_a, p, embeddings=emb, doc_weight=delta)
        qb, nb = encode(doc_b, p, embeddings=emb, doc_weight=delta)
        return float(qa.astype(np.int32) @ qb.astype(np.int32)) / (na * nb)

    sim_0 = cos(0.0)
    sim_5 = cos(0.5)
    assert (
        sim_5 > sim_0 + 0.1
    )  # le canal document rapproche les deux docs de même thème


def test_doc_weight_zero_bit_identique_v12():
    p = _profiles()
    q1, n1 = encode(["pose", "interrupteur"], p)
    q2, n2 = encode(["pose", "interrupteur"], p, doc_weight=0.0)
    assert np.array_equal(q1, q2) and n1 == n2


def test_doc_weight_sans_embeddings_inchange():
    p = _profiles()
    q1, n1 = encode(["pose", "interrupteur"], p)
    q2, n2 = encode(
        ["pose", "interrupteur"], p, doc_weight=0.5
    )  # pas d'embeddings -> canal indisponible
    assert np.array_equal(q1, q2) and n1 == n2


def test_doc_weight_document_vide_inchange(tmp_path):
    emb = _table_synthetique(tmp_path)
    p = _profiles()
    q, n = encode([], p, embeddings=emb, doc_weight=0.5)
    assert n == 0.0 and not q.any()


def test_doc_weight_canal_indisponible_hors_table(tmp_path):
    # Aucun token du document n'est dans la table -> doc_channel retourne None -> chemin v1.2 exact
    emb = _table_synthetique(tmp_path)
    p = _profiles()
    q1, n1 = encode(["pose", "interrupteur"], p)
    q2, n2 = encode(["pose", "interrupteur"], p, embeddings=None)
    q3, n3 = encode(["pose", "interrupteur"], p, embeddings=emb, doc_weight=0.5)
    assert np.array_equal(q1, q2)
    # Sans embeddings vs avec embeddings mais canal indisponible : l'ancien encode() sans
    # embeddings est déjà différent d'un encode() *avec* embeddings (vector_grid par mot
    # change word_vector) — donc q3 se compare à un encode(embeddings=emb, doc_weight=0.0).
    q4, n4 = encode(["pose", "interrupteur"], p, embeddings=emb, doc_weight=0.0)
    assert np.array_equal(q3, q4) and n3 == n4
