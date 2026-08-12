from pathlib import Path

import numpy as np
import pytest

from mosaic.index import Index

GRID = (32, 32, 3)


def _corpus(tmp_path: Path) -> Path:
    c = tmp_path / "corpus"
    c.mkdir(parents=True, exist_ok=True)
    (c / "elec1.md").write_text(
        "pose interrupteur différentiel salle de bain volume protection",
        encoding="utf-8",
    )
    (c / "elec2.md").write_text(
        "remplacement interrupteur disjoncteur différentiel tableau protection circuit",
        encoding="utf-8",
    )
    (c / "carrelage.md").write_text(
        "achat carrelage gris cuisine colle joint pose sol", encoding="utf-8"
    )
    return c


def test_build_search_pertinent(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    hits = idx.search("interrupteur différentiel protection", k=3)
    assert [h["id"] for h in hits[:2]] == ["elec1.md", "elec2.md"] or [
        h["id"] for h in hits[:2]
    ] == ["elec2.md", "elec1.md"]
    assert hits[0]["score"] > hits[2]["score"]


def test_open_retrouve_le_meme_classement(tmp_path):
    Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    idx = Index.open(tmp_path / "idx")
    hits = idx.search("carrelage cuisine", k=1)
    assert hits[0]["id"] == "carrelage.md"


def test_add_incremental(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    extra = tmp_path / "plomberie.md"
    extra.write_text(
        "chauffe-eau ballon eau chaude raccordement cuivre", encoding="utf-8"
    )
    idx.add(extra)
    hits = idx.search("ballon eau chaude", k=1)
    assert hits[0]["id"] == "plomberie.md"
    assert idx.stats()["docs"] == 4


def test_stats(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    s = idx.stats()
    assert s["docs"] == 3 and s["grid"] == [32, 32, 3] and s["vocab"] > 0


def test_stats_expose_index_paths_et_ocr(tmp_path):
    """Revue finale (mineur) : stats() doit exposer les deux leviers persistés au meta
    (index_paths/ocr) — un agent qui inspecte un index doit savoir sans relire le meta
    à la main si les tokens de chemin/l'OCR étaient actifs au dernier build."""
    idx_on = Index.build(
        _corpus(tmp_path / "corpus_on"), tmp_path / "idx_on", grid=GRID
    )
    assert idx_on.stats()["index_paths"] is True
    assert idx_on.stats()["ocr"] is False

    idx_off = Index.build(
        _corpus(tmp_path / "corpus_off"),
        tmp_path / "idx_off",
        grid=GRID,
        index_paths=False,
    )
    assert idx_off.stats()["index_paths"] is False


def test_build_deterministe(tmp_path):
    Index.build(_corpus(tmp_path / "corpus_a"), tmp_path / "ia", grid=GRID)
    Index.build(_corpus(tmp_path / "corpus_b"), tmp_path / "ib", grid=GRID)
    assert (tmp_path / "ia" / "docs.msei").read_bytes() == (
        tmp_path / "ib" / "docs.msei"
    ).read_bytes()


def test_build_corpus_inexistant_leve_valueerror(tmp_path):
    with pytest.raises(ValueError):
        Index.build(tmp_path / "nexiste_pas", tmp_path / "idx", grid=GRID)


# -- v1.5 : chargement paresseux de Index.open (perf recherche) ------------------------------


def test_open_lazy_vs_eager_meme_recherche(tmp_path):
    """Bit-identité recherche : Index.open(lazy=True) (défaut) et Index.open(lazy=False)
    doivent produire exactement les mêmes résultats sur le même index."""
    Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    idx_lazy = Index.open(tmp_path / "idx", lazy=True)
    idx_eager = Index.open(tmp_path / "idx", lazy=False)
    hits_lazy = idx_lazy.search("interrupteur différentiel protection", k=3)
    hits_eager = idx_eager.search("interrupteur différentiel protection", k=3)
    assert hits_lazy == hits_eager


def test_open_par_defaut_est_lazy(tmp_path):
    Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    idx = Index.open(tmp_path / "idx")
    assert isinstance(idx.profiles.acc, np.memmap)


def test_add_apres_open_lazy_fonctionne_et_egale_add_eager(tmp_path):
    """Piège Windows : add() après Index.open(lazy=True) doit pouvoir ré-écrire vocab.msev
    (le memmap hérité de l'ouverture ne doit plus tenir le fichier verrouillé au moment de
    _save()). Les deux chemins (add après open lazy vs add après open eager) doivent produire
    un index bit-identique sur disque."""
    corpus = _corpus(tmp_path)
    Index.build(corpus, tmp_path / "idx_lazy", grid=GRID)
    Index.build(corpus, tmp_path / "idx_eager", grid=GRID)
    extra = tmp_path / "plomberie.md"
    extra.write_text(
        "chauffe-eau ballon eau chaude raccordement cuivre", encoding="utf-8"
    )

    idx_lazy = Index.open(tmp_path / "idx_lazy", lazy=True)
    idx_lazy.add(extra)  # doit réussir sans PermissionError sur Windows

    idx_eager = Index.open(tmp_path / "idx_eager", lazy=False)
    idx_eager.add(extra)

    assert (tmp_path / "idx_lazy" / "docs.msei").read_bytes() == (
        tmp_path / "idx_eager" / "docs.msei"
    ).read_bytes()
    assert (tmp_path / "idx_lazy" / "vocab.msev").read_bytes() == (
        tmp_path / "idx_eager" / "vocab.msev"
    ).read_bytes()

    hits = idx_lazy.search("ballon eau chaude", k=1)
    assert hits[0]["id"] == "plomberie.md"
    assert idx_lazy.stats()["docs"] == 4

    # Ré-ouverture après le add() lazy : le fichier ré-écrit doit rester valide et lisible.
    idx_reload = Index.open(tmp_path / "idx_lazy")
    assert idx_reload.stats()["docs"] == 4
    assert idx_reload.search("ballon eau chaude", k=1)[0]["id"] == "plomberie.md"


def test_recherche_croisee_anglais_vers_francais(tmp_path):
    idx = Index.build(
        _corpus(tmp_path),
        tmp_path / "idx",
        grid=GRID,
        lexicon={
            "breaker": "disjoncteur",
            "panel": "tableau",
            "switch": "interrupteur",
        },
    )
    hits = idx.search("switch panel protection", k=1)
    assert hits[0]["id"] in ("elec1.md", "elec2.md")


def test_recherche_croisee_expression_anglaise(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "boite.md").write_text(
        "pose boîte dérivation mur cuisine raccordement", encoding="utf-8"
    )
    (c / "autre.md").write_text("peinture plafond blanc", encoding="utf-8")
    idx = Index.build(
        c, tmp_path / "idx", grid=GRID, lexicon={"junction_box": "boîte_de_dérivation"}
    )
    assert idx.search("junction box", k=1)[0]["id"] == "boite.md"


def _table(tmp_path):
    import gzip

    import numpy as np

    lines = ["3 300"]
    for w, seed in (("panne", 7), ("défaillance", 7), ("carrelage", 8)):
        rng = np.random.default_rng(seed)
        lines.append(w + " " + " ".join(f"{v:.4f}" for v in rng.normal(size=300)))
    src = tmp_path / "mini.vec.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    out = tmp_path / "table.msee"
    from mosaic.embeddings import prepare

    prepare(src, out)
    return out


def test_build_avec_embeddings_et_reouverture(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "elec.md").write_text("panne moteur atelier intervention", encoding="utf-8")
    (c / "sol.md").write_text("carrelage colle joint sol", encoding="utf-8")
    table = _table(tmp_path)
    from mosaic.embeddings import Embeddings

    idx = Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        embeddings=Embeddings.load(table),
        embeddings_path=table,
    )
    hits = idx.search("défaillance", k=2)  # jamais dans le corpus — pont embedding pur
    assert hits[0]["id"] == "elec.md"
    idx2 = Index.open(tmp_path / "idx")  # recharge la table depuis le meta
    assert idx2.search("défaillance", k=2)[0]["id"] == "elec.md"


def test_build_avec_poids_personnalises_survivent_a_la_reouverture(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "elec.md").write_text("panne moteur atelier intervention", encoding="utf-8")
    table = _table(tmp_path)
    from mosaic.embeddings import Embeddings

    custom_weights = (0.35, 0.25, 0.40)
    idx = Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        embeddings=Embeddings.load(table),
        embeddings_path=table,
        weights=custom_weights,
    )
    assert idx.weights == custom_weights
    idx2 = Index.open(tmp_path / "idx")  # recharge le meta depuis le disque
    assert idx2.weights == custom_weights


def test_open_refuse_table_alteree(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "a.md").write_text("panne moteur", encoding="utf-8")
    table = _table(tmp_path)
    from mosaic.embeddings import Embeddings

    Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        embeddings=Embeddings.load(table),
        embeddings_path=table,
    )
    data = bytearray(table.read_bytes())
    data[-1] ^= 0xFF
    table.write_bytes(bytes(data))
    import pytest

    with pytest.raises(ValueError):
        Index.open(tmp_path / "idx")


def test_index_meta_abtt_round_trip(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "elec.md").write_text("panne moteur atelier intervention", encoding="utf-8")
    (c / "sol.md").write_text("carrelage colle joint sol", encoding="utf-8")
    table = _table(tmp_path)
    from mosaic.embeddings import Embeddings

    idx = Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        embeddings=Embeddings.load(table, abtt=2),
        embeddings_path=table,
        abtt=2,
    )
    assert idx.abtt == 2
    hits = idx.search("défaillance", k=2)
    idx2 = Index.open(tmp_path / "idx")
    assert idx2.abtt == 2
    assert idx2.embeddings.abtt == 2
    assert idx2.search("défaillance", k=2) == hits


def test_index_build_embeddings_abtt_mismatch_leve_valueerror(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "elec.md").write_text("panne moteur atelier intervention", encoding="utf-8")
    table = _table(tmp_path)
    from mosaic.embeddings import Embeddings
    import pytest

    with pytest.raises(ValueError):
        Index.build(
            c,
            tmp_path / "idx",
            grid=GRID,
            embeddings=Embeddings.load(table, abtt=0),
            embeddings_path=table,
            abtt=2,
        )


def test_open_verify_embeddings_false_meme_recherche_que_verify_true(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "elec.md").write_text("panne moteur atelier intervention", encoding="utf-8")
    (c / "sol.md").write_text("carrelage colle joint sol", encoding="utf-8")
    table = _table(tmp_path)
    from mosaic.embeddings import Embeddings

    Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        embeddings=Embeddings.load(table),
        embeddings_path=table,
    )
    idx_verified = Index.open(tmp_path / "idx", verify_embeddings=True)
    idx_lazy = Index.open(tmp_path / "idx", verify_embeddings=False)
    assert idx_verified.search("défaillance", k=2) == idx_lazy.search(
        "défaillance", k=2
    )


def test_open_verify_embeddings_false_avec_abtt_meme_recherche(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "elec.md").write_text("panne moteur atelier intervention", encoding="utf-8")
    (c / "sol.md").write_text("carrelage colle joint sol", encoding="utf-8")
    table = _table(tmp_path)
    from mosaic.embeddings import Embeddings

    Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        embeddings=Embeddings.load(table, abtt=2),
        embeddings_path=table,
        abtt=2,
    )
    idx_verified = Index.open(tmp_path / "idx", verify_embeddings=True)
    idx_lazy = Index.open(tmp_path / "idx", verify_embeddings=False)
    assert idx_verified.search("défaillance", k=2) == idx_lazy.search(
        "défaillance", k=2
    )


def test_open_verify_embeddings_false_stats_signale_non_verifie(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "elec.md").write_text("panne moteur atelier intervention", encoding="utf-8")
    table = _table(tmp_path)
    from mosaic.embeddings import Embeddings

    Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        embeddings=Embeddings.load(table),
        embeddings_path=table,
    )
    idx = Index.open(tmp_path / "idx", verify_embeddings=False)
    assert idx.stats()["embeddings_verifies"] is False
    idx2 = Index.open(tmp_path / "idx", verify_embeddings=True)
    assert "embeddings_verifies" not in idx2.stats()


def test_open_verify_embeddings_false_ignore_table_alteree(tmp_path):
    """Contraste avec test_open_refuse_table_alteree : verify_embeddings=False fait confiance
    à la table sans relire/comparer le sha (chemin recherche assumé) — jamais le défaut."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "a.md").write_text("panne moteur", encoding="utf-8")
    table = _table(tmp_path)
    from mosaic.embeddings import Embeddings

    Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        embeddings=Embeddings.load(table),
        embeddings_path=table,
    )
    data = bytearray(table.read_bytes())
    data[-1] ^= 0xFF
    table.write_bytes(bytes(data))
    idx = Index.open(tmp_path / "idx", verify_embeddings=False)  # ne lève PAS
    assert idx.stats()["embeddings_verifies"] is False


def test_index_sans_embeddings_reste_v1(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    assert idx.search("carrelage cuisine", k=1)[0]["id"] == "carrelage.md"


# -- revue finale v1.5 : add() refuse un index ouvert sans vérification des embeddings ------


def test_add_apres_open_verify_embeddings_false_leve_valueerror(tmp_path):
    """Ajouter après Index.open(verify_embeddings=False) écrirait embed_sha="" via _save()
    (Embeddings.load(verify=False) ne calcule jamais de sha) — l'index redeviendrait
    inouvrable en mode vérifié (défaut). add() doit refuser net plutôt que produire cet état."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "elec.md").write_text("panne moteur atelier intervention", encoding="utf-8")
    table = _table(tmp_path)
    from mosaic.embeddings import Embeddings

    Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        embeddings=Embeddings.load(table),
        embeddings_path=table,
    )

    idx = Index.open(tmp_path / "idx", verify_embeddings=False)
    extra = tmp_path / "extra.md"
    extra.write_text("carrelage colle joint", encoding="utf-8")
    with pytest.raises(ValueError):
        idx.add(extra)

    # index inchangé sur disque : reste ouvrable en mode vérifié (défaut), toujours 1 doc.
    idx_reopen = Index.open(tmp_path / "idx", verify_embeddings=True)
    assert idx_reopen.stats()["docs"] == 1


def test_add_apres_open_verify_embeddings_true_fonctionne_et_reste_ouvrable(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "elec.md").write_text("panne moteur atelier intervention", encoding="utf-8")
    table = _table(tmp_path)
    from mosaic.embeddings import Embeddings

    Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        embeddings=Embeddings.load(table),
        embeddings_path=table,
    )

    idx = Index.open(tmp_path / "idx", verify_embeddings=True)
    extra = tmp_path / "extra.md"
    extra.write_text("carrelage colle joint", encoding="utf-8")
    idx.add(extra)  # ne lève pas

    idx_reopen = Index.open(
        tmp_path / "idx", verify_embeddings=True
    )  # toujours ouvrable
    assert idx_reopen.stats()["docs"] == 2


def test_explain_retrouve_les_concepts_dominants(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    top = idx.explain("carrelage.md", k=5)
    tokens = [t["token"] for t in top]
    assert "carrelage" in tokens
    assert top[0]["poids"] >= top[-1]["poids"]
    with pytest.raises(ValueError):
        idx.explain("inconnu.md")


def test_explain_match_nomme_le_pont(tmp_path):
    idx = Index.build(
        _corpus(tmp_path),
        tmp_path / "idx",
        grid=GRID,
        lexicon={"breaker": "disjoncteur"},
    )
    contribs = idx.explain_match("breaker", "elec2.md", k=3)
    assert contribs[0]["token"] == "disjoncteur"


def test_build_defaut_profile_weighting_ppmi(tmp_path):
    # v1.2 : ppmi/300 sont les défauts de build (bench sur corpus réel 2100 docs) ; brut
    # reste couvert explicitement par test_recherche_croisee_*/test_build_profile_weighting_*.
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, smoothing_rank=0)
    assert idx.profile_weighting == "ppmi"
    assert idx.legacy is False


def test_build_meta_pin_nouveaux_defauts_v1_2(tmp_path):
    """Pin explicite des nouveaux défauts de Index.build (bench v1.2) : un build sans
    arguments explicites doit produire profile_weighting="ppmi" et smoothing_rank=300,
    à la fois sur l'objet en mémoire et dans le meta persisté sur disque."""
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    assert idx.profile_weighting == "ppmi"
    assert idx.smoothing_rank == 300
    from mosaic.store import load_vocab

    _, _, _, meta = load_vocab(tmp_path / "idx")
    assert meta["profile_weighting"] == "ppmi"
    assert meta["smoothing_rank"] == 300


def test_build_profile_weighting_ppmi_survit_a_la_reouverture_et_a_add(tmp_path):
    idx = Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, profile_weighting="ppmi"
    )
    assert idx.profile_weighting == "ppmi"
    idx2 = Index.open(tmp_path / "idx")
    assert idx2.profile_weighting == "ppmi"
    extra = tmp_path / "plomberie.md"
    extra.write_text(
        "chauffe-eau ballon eau chaude raccordement cuivre", encoding="utf-8"
    )
    idx2.add(extra)
    assert idx2.stats()["docs"] == 4
    hits = idx2.search("ballon eau chaude", k=1)
    assert hits[0]["id"] == "plomberie.md"


def test_build_smoothing_rank_defaut_300(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    assert idx.smoothing_rank == 300


def test_build_smoothing_rank_survit_a_la_reouverture_et_search_coherent(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, smoothing_rank=2)
    assert idx.smoothing_rank == 2
    hits = idx.search("interrupteur différentiel protection", k=3)
    assert {h["id"] for h in hits[:2]} == {"elec1.md", "elec2.md"}

    idx2 = Index.open(tmp_path / "idx")
    assert idx2.smoothing_rank == 2
    # acc est déjà lissé sur disque : réouverture -> même recherche, aucun re-lissage.
    hits2 = idx2.search("interrupteur différentiel protection", k=3)
    assert hits2 == hits
    assert np.array_equal(idx2.profiles.acc, idx.profiles.acc)


def test_add_apres_open_relisse_avec_le_rang_stocke(tmp_path):
    Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, smoothing_rank=2)
    idx2 = Index.open(tmp_path / "idx")
    extra = tmp_path / "plomberie.md"
    extra.write_text(
        "chauffe-eau ballon eau chaude raccordement cuivre", encoding="utf-8"
    )
    idx2.add(extra)
    assert idx2.smoothing_rank == 2
    assert idx2.stats()["docs"] == 4

    # finalize() reconstruit acc en entier depuis les comptages épars (aucun
    # lissage intrinsèque) : si add() n'avait pas réappliqué smooth(rank=2),
    # le rang effectif de acc serait bien supérieur à 2. On vérifie donc que
    # les valeurs singulières au-delà du rang 2 sont négligeables.
    v = len(idx2.profiles.rows)
    assert v > 3  # sinon le test ne distingue rien
    s = np.linalg.svd(idx2.profiles.acc[:v], compute_uv=False)
    assert s[2] < 1e-3 * s[0]


def test_doc_weight_zero_bit_identique_v12(tmp_path):
    Index.build(_corpus(tmp_path / "corpus_a"), tmp_path / "ia", grid=GRID)
    Index.build(
        _corpus(tmp_path / "corpus_b"), tmp_path / "ib", grid=GRID, doc_weight=0.0
    )
    assert (tmp_path / "ia" / "docs.msei").read_bytes() == (
        tmp_path / "ib" / "docs.msei"
    ).read_bytes()


def test_doc_weight_meta_round_trip_et_requetes(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "elec.md").write_text("panne moteur atelier intervention", encoding="utf-8")
    (c / "sol.md").write_text("carrelage colle joint sol", encoding="utf-8")
    table = _table(tmp_path)
    from mosaic.embeddings import Embeddings

    idx = Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        embeddings=Embeddings.load(table),
        embeddings_path=table,
        doc_weight=0.25,
    )
    assert idx.doc_weight == 0.25
    hits = idx.search("panne", k=2)
    assert hits[0]["id"] == "elec.md"
    assert "canal_document" not in idx.stats()

    idx2 = Index.open(tmp_path / "idx")
    assert idx2.doc_weight == 0.25
    assert "canal_document" not in idx2.stats()
    assert idx2.search("panne", k=2) == hits


def test_doc_weight_sans_table_inactif(tmp_path):
    Index.build(_corpus(tmp_path / "corpus_a"), tmp_path / "i0", grid=GRID)
    idx = Index.build(
        _corpus(tmp_path / "corpus_b"), tmp_path / "idx", grid=GRID, doc_weight=0.25
    )
    assert idx.stats()["canal_document"] == "inactif (pas de table)"
    # sans table, δ n'a aucun effet : sortie identique au chemin v1.2 (δ=0)
    assert (tmp_path / "i0" / "docs.msei").read_bytes() == (
        tmp_path / "idx" / "docs.msei"
    ).read_bytes()


def test_open_index_legacy_smoothing_rank_zero(tmp_path):
    # index construit avant l'existence du levier (pas de clé smoothing_rank en meta)
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    from mosaic.store import load_vocab, save_vocab

    p, colloc, lex, meta = load_vocab(tmp_path / "idx")
    meta.pop("smoothing_rank", None)
    meta.pop("legacy", None)
    save_vocab(
        tmp_path / "idx",
        p,
        colloc,
        idx.grid,
        lex,
        extra_meta={"profile_weighting": "brut"},
    )
    idx2 = Index.open(tmp_path / "idx")
    assert idx2.smoothing_rank == 0


# -- v1.5 : tokens de chemin (index_paths) ---------------------------------------------------


def _corpus_nomme(tmp_path: Path) -> Path:
    """Corpus où le nom/chemin porte l'information distinctive (« KUMQUAT »), jamais le
    contenu — seule l'injection des tokens de chemin peut la rendre cherchable."""
    c = tmp_path / "devis"
    sub = c / "2026" / "08.2026"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "D26089903_KUMQUAT_devis.md").write_text(
        "fourniture pose tableau électrique cablage protection", encoding="utf-8"
    )
    for i in range(4):
        (sub / f"D260800{i}_client{i}_devis.md").write_text(
            "fourniture pose interrupteur disjoncteur cablage protection",
            encoding="utf-8",
        )
    return c


def test_index_paths_defaut_true_trouve_par_nom_de_fichier(tmp_path):
    idx = Index.build(_corpus_nomme(tmp_path), tmp_path / "idx", grid=GRID)
    assert idx.index_paths is True
    hits = idx.search("kumquat", k=1)
    assert "KUMQUAT" in hits[0]["id"]
    assert hits[0]["score"] > 0.3  # match net (contenu ne mentionne jamais « kumquat »)


def test_no_path_tokens_score_bien_plus_faible_sans_injection(tmp_path):
    """Sans injection, « kumquat » reste un token jamais appris (OOV) : seul un résidu
    de bruit de signature de hachage subsiste, sans commune mesure avec le match
    net obtenu quand l'injection est active (k=5 == tout le corpus : la présence
    dans les résultats n'est donc pas discriminante ici, l'écart de score l'est)."""
    idx_on = Index.build(_corpus_nomme(tmp_path), tmp_path / "idx_on", grid=GRID)
    idx_off = Index.build(
        _corpus_nomme(tmp_path), tmp_path / "idx_off", grid=GRID, index_paths=False
    )
    score_on = next(
        h["score"] for h in idx_on.search("kumquat", k=5) if "KUMQUAT" in h["id"]
    )
    score_off = next(
        h["score"] for h in idx_off.search("kumquat", k=5) if "KUMQUAT" in h["id"]
    )
    assert score_on > 5 * score_off


def test_index_paths_false_saute_totalement_linjection(tmp_path):
    """Le flag OFF doit produire EXACTEMENT le même flux de tokens que l'ancien
    comportement (v1.4) : aucune trace du chemin dans le vocabulaire appris."""
    idx = Index.build(
        _corpus_nomme(tmp_path), tmp_path / "idx", grid=GRID, index_paths=False
    )
    assert "kumquat" not in idx.profiles.rows
    assert "d26080003" not in idx.profiles.rows


def test_index_paths_true_ajoute_les_tokens_du_chemin_au_vocabulaire(tmp_path):
    idx = Index.build(_corpus_nomme(tmp_path), tmp_path / "idx", grid=GRID)
    assert "kumquat" in idx.profiles.rows


def test_no_path_tokens_bit_identique_deux_builds(tmp_path):
    """--no-path-tokens (index_paths=False) : deux builds indépendants du même
    contenu doivent rester bit-identiques (déterminisme conservé, comportement
    v1.4 exact — aucune fuite de chemin n'entre jamais dans le flux)."""
    Index.build(
        _corpus(tmp_path / "corpus_a"), tmp_path / "ia", grid=GRID, index_paths=False
    )
    Index.build(
        _corpus(tmp_path / "corpus_b"), tmp_path / "ib", grid=GRID, index_paths=False
    )
    assert (tmp_path / "ia" / "docs.msei").read_bytes() == (
        tmp_path / "ib" / "docs.msei"
    ).read_bytes()


def test_index_paths_meta_round_trip(tmp_path):
    Index.build(
        _corpus_nomme(tmp_path), tmp_path / "idx_on", grid=GRID, index_paths=True
    )
    assert Index.open(tmp_path / "idx_on").index_paths is True
    Index.build(
        _corpus_nomme(tmp_path), tmp_path / "idx_off", grid=GRID, index_paths=False
    )
    assert Index.open(tmp_path / "idx_off").index_paths is False


def test_index_paths_survit_a_add(tmp_path):
    idx = Index.build(_corpus_nomme(tmp_path), tmp_path / "idx", grid=GRID)
    extra = tmp_path / "2026" / "09.2026" / "D26090001_DURAND_devis.md"
    extra.parent.mkdir(parents=True)
    extra.write_text("fourniture pose cablage protection", encoding="utf-8")
    idx.add(extra)
    hits = idx.search("durand", k=1)
    assert (
        hits[0]["id"] == "D26090001_DURAND_devis.md"
    )  # add() ne garde que le nom du fichier
    assert hits[0]["score"] > 0.3  # match net (contenu ne mentionne jamais « durand »)


def test_no_path_tokens_survit_a_add(tmp_path):
    """Flag OFF persisté : add() ne doit JAMAIS injecter de tokens de chemin qu'un
    build sans injection n'avait pas — vérifié au niveau vocabulaire (déterministe,
    contrairement au score qui reste sujet au bruit résiduel de signature OOV)."""
    idx = Index.build(
        _corpus_nomme(tmp_path), tmp_path / "idx", grid=GRID, index_paths=False
    )
    extra = tmp_path / "2026" / "09.2026" / "D26090001_DURAND_devis.md"
    extra.parent.mkdir(parents=True)
    extra.write_text("fourniture pose cablage protection", encoding="utf-8")
    idx.add(extra)
    assert "durand" not in idx.profiles.rows


def test_add_ne_pollue_pas_le_vocabulaire_avec_les_composants_du_chemin_absolu(
    tmp_path,
):
    """Bug revue finale (critique, reproduit) : add() appelait _path_tokens(file.as_posix())
    sur le chemin TEL QUE PASSÉ par l'appelant — un chemin absolu injecte tous ses composants
    parents (répertoires temporaires, nom d'utilisateur…) dans le vocabulaire PERSISTÉ de
    l'index. build() n'a jamais ce problème (doc_id relatif au corpus). Cohérence attendue :
    add() ne doit tokeniser QUE file.name, exactement ce qui est stocké dans self.ids."""
    idx = Index.build(_corpus(tmp_path / "corpus"), tmp_path / "idx", grid=GRID)
    extra = tmp_path / "un_repertoire_parent" / "un_autre_niveau" / "plomberieAAA.md"
    extra.parent.mkdir(parents=True)
    extra.write_text(
        "chauffe-eau ballon eau chaude raccordement cuivre", encoding="utf-8"
    )

    idx.add(extra)

    # aucun composant des répertoires parents n'entre dans le vocabulaire persisté
    assert "repertoire" not in idx.profiles.rows
    assert "parent" not in idx.profiles.rows
    assert "autre" not in idx.profiles.rows
    assert "niveau" not in idx.profiles.rows
    # les tokens du NOM de fichier restent injectés (cohérent avec self.ids.append(file.name))
    assert "plomberieaaa" in idx.profiles.rows


def test_open_index_legacy_index_paths_par_defaut_false(tmp_path):
    """Un index sur disque sans clé `index_paths` a été construit avant l'existence
    du levier (comportement v1.4) : la ré-ouverture ne doit jamais se mettre à
    injecter des tokens de chemin qu'un `add()` ultérieur n'avait pas au build."""
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    from mosaic.store import load_vocab, save_vocab

    p, colloc, lex, meta = load_vocab(tmp_path / "idx")
    meta.pop("index_paths", None)
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
    assert idx2.index_paths is False


# -- v1.5 : exclusions d'indexation élargies (EXCLUDED_DIRS) ---------------------------------


def test_excluded_dirs_backups_corbeille_cimetiere_poubelleclaude(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "note.md").write_text("contenu réel du dossier", encoding="utf-8")
    for sous_dossier in ("_backups", "_corbeille", "_cimetiere", "poubelleClaude"):
        piege = d / sous_dossier
        piege.mkdir()
        (piege / "fantome.md").write_text(
            "ne doit jamais être indexé", encoding="utf-8"
        )

    idx = Index.build(d, tmp_path / "idx", grid=GRID)

    assert idx.ids == ["note.md"]
    assert idx.stats()["docs"] == 1


def test_excluded_dirs_profondeur_quelconque(tmp_path):
    """Tout composant de chemin correspondant exclut le fichier — pas seulement en racine."""
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "note.md").write_text("contenu réel", encoding="utf-8")
    profond = d / "archives" / "_backups" / "2026" / "old"
    profond.mkdir(parents=True)
    (profond / "fantome.md").write_text("jamais indexé", encoding="utf-8")

    idx = Index.build(d, tmp_path / "idx", grid=GRID)

    assert idx.ids == ["note.md"]


# -- v1.5 : critère d'acceptation (mini-corpus nommé, intégration) --------------------------


def test_critere_kumquat_top3_et_aucun_backup_apres_reconstruction(tmp_path):
    corpus = _corpus_nomme(tmp_path)
    piege = corpus / "_backups"
    piege.mkdir()
    (piege / "D26089903_KUMQUAT_devis_old.md").write_text(
        "ancien brouillon kumquat fourniture pose", encoding="utf-8"
    )

    idx = Index.build(corpus, tmp_path / "idx", grid=GRID)
    hits = idx.search("le devis kumquat", k=3)

    ids = [h["id"] for h in hits]
    assert any("KUMQUAT" in i for i in ids)
    assert not any("_backups" in i for i in ids)
    assert not any("_backups" in i for i in idx.ids)


# -- revue finale v1.6 (Critical) : écritures atomiques de _save() --------------------------


def test_save_atomic_vocab_intact_si_le_2e_fichier_echoue(tmp_path, monkeypatch):
    """`Index._save()` écrit docs.msei PUIS vocab.msev (2e fichier, sans rerank.msrv dans
    cette configuration). Si l'écriture du 2e fichier échoue (crash simulé, ou verrou Windows
    sur un memmap lecteur), vocab.msev doit rester intact (contenu ANCIEN, chargeable) —
    jamais un fichier tronqué/à moitié écrit. `_write()` écrit désormais dans un `.tmp`
    voisin puis `os.replace()` : un échec au remplacement laisse la destination inchangée."""
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    vocab_path = tmp_path / "idx" / "vocab.msev"
    original_vocab_bytes = vocab_path.read_bytes()

    import mosaic.store as store_module

    real_replace = store_module.os.replace

    def failing_replace(src, dst):
        if Path(dst).name == "vocab.msev":
            raise OSError("échec simulé — destination verrouillée (memmap lecteur)")
        return real_replace(src, dst)

    monkeypatch.setattr(store_module.os, "replace", failing_replace)

    nouveau = tmp_path / "corpus" / "nouveau.md"
    nouveau.write_text("nouveau document jamais vu", encoding="utf-8")
    with pytest.raises(OSError):
        idx.add(nouveau)

    # vocab.msev n'a jamais été touché : bit-identique à avant l'add() raté.
    assert vocab_path.read_bytes() == original_vocab_bytes
    # aucun résidu .tmp qui traînerait indéfiniment à côté de l'index.
    assert not (tmp_path / "idx" / "vocab.msev.tmp").exists()
    # vocab.msev reste chargeable normalement (pas un fichier à moitié écrit).
    from mosaic.store import load_vocab

    load_vocab(tmp_path / "idx")  # ne lève pas


def test_save_normal_round_trip_toujours_ok_apres_ecriture_atomique(tmp_path):
    """Non-régression : une sauvegarde normale (aucune panne) round-trip toujours à
    l'identique une fois `_write()` passé par le chemin temp+replace."""
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    ids_avant = list(idx.ids)
    idx2 = Index.open(tmp_path / "idx")
    assert idx2.ids == ids_avant
    hits = idx2.search("interrupteur différentiel", k=2)
    assert hits


def test_open_refuse_index_dechire_docs_vocab_incoherents(tmp_path):
    """Garde de cohérence inter-fichiers (revue v1.6) : si docs.msei et vocab.msev
    portent des nombres de documents différents (écriture _save interrompue), open()
    refuse loud au lieu de servir des scores incohérents en silence."""
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    # fausser n_docs dans vocab.msev (simule un vocab périmé après _save interrompu)
    from mosaic.store import load_vocab, save_vocab

    p, colloc, lex, meta = load_vocab(tmp_path / "idx", lazy=False)
    p.n_docs = p.n_docs + 5  # incohérent avec les 3 lignes de docs.msei
    save_vocab(
        tmp_path / "idx",
        p,
        colloc,
        idx.grid,
        lex,
        extra_meta={
            k: v
            for k, v in meta.items()
            if k not in ("order", "counts", "df", "n_docs", "colloc", "lexicon")
        },
    )
    import pytest as _pytest

    with _pytest.raises(ValueError, match="incohérent"):
        Index.open(tmp_path / "idx")
