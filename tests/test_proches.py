"""Tests — voisinage de mots (Embeddings.proches sur table, Index.proches sur corpus)."""

from mosaic.index import Index

GRID = (32, 32, 3)


def test_proches_cooccurrence_corpus(tmp_path):
    """Voisinage de cooccurrence appris du corpus. alpha n'apparaît QU'avec beta (+ des
    couleurs partagées) ; gamma QU'avec delta. Le profil d'alpha est donc plus proche de celui
    de beta que de gamma. smoothing_rank=0 : PPMI brut, cooccurrence directe (pas de SVD qui
    dégénère sur un vocabulaire minuscule). index_paths=False : pas de pollution par les noms."""
    c = tmp_path / "corpus"
    c.mkdir(parents=True, exist_ok=True)
    # alpha/beta cooccurrent, mais restent une MINORITÉ des docs (sinon le filtre de
    # fréquence documentaire les prendrait pour du boilerplate).
    (c / "d1.md").write_text("alpha beta rouge", encoding="utf-8")
    (c / "d2.md").write_text("alpha beta vert", encoding="utf-8")
    (c / "d3.md").write_text("gamma delta rouge", encoding="utf-8")
    (c / "d4.md").write_text("gamma delta vert", encoding="utf-8")
    for i in range(6):  # docs de remplissage, sujets variés
        (c / f"f{i}.md").write_text(
            f"epsilon zeta sujet{i} contexte{i} materiel{i}", encoding="utf-8"
        )
    idx = Index.build(
        c, tmp_path / "idx", grid=GRID, index_paths=False, smoothing_rank=0
    )
    voisins = idx.proches("alpha", k=3)
    assert voisins is not None
    mots = [m for m, _ in voisins]
    assert "beta" in mots  # cooccurrence exclusive alpha<->beta
    assert mots.index("beta") <= (
        mots.index("gamma") if "gamma" in mots else 99
    )  # beta plus proche que gamma
    assert "alpha" not in mots  # soi-même exclu
    assert idx.proches("mot_totalement_absent") is None  # hors vocabulaire -> None


def test_proches_filtre_dico_lexical(tmp_path):
    """Le filtre `dico` (Profiles.proches) ne garde que les tokens présents dans un vocabulaire de
    vrais mots, et écarte les nombres/codes — même s'ils cooccurrent fortement. Testé au niveau
    Profiles (un index de test n'a pas de table potion embarquée)."""
    c = tmp_path / "corpus"
    c.mkdir(parents=True, exist_ok=True)
    # 'alpha' cooccurre avec un vrai mot (beta), un token hors-dico (zzcode), un nombre (1234).
    # Des docs de remplissage variés créent le contraste PPMI (sinon profil nul).
    (c / "d1.md").write_text("alpha beta zzcode 1234", encoding="utf-8")
    (c / "d2.md").write_text("alpha beta zzcode 1234", encoding="utf-8")
    (c / "d3.md").write_text("alpha beta zzcode 1234", encoding="utf-8")
    for i in range(6):
        (c / f"f{i}.md").write_text(
            f"epsilon zeta theta{i} kappa{i} lambda{i}", encoding="utf-8"
        )
    idx = Index.build(
        c, tmp_path / "idx", grid=GRID, index_paths=False, smoothing_rank=0
    )
    prof = idx.profiles

    sans = {m for m, _ in (prof.proches("alpha", k=10) or [])}
    assert {"beta", "zzcode"} <= sans  # sans dico : tout le voisinage cooccurrent

    dico = frozenset({"alpha", "beta"})  # 'zzcode' et '1234' n'y sont PAS
    avec = {m for m, _ in (prof.proches("alpha", k=10, dico=dico) or [])}
    assert "beta" in avec  # vrai mot -> gardé
    assert "zzcode" not in avec  # hors dico -> filtré
    assert "1234" not in avec  # nombre (ratio de chiffres > 0.3) -> filtré
