"""Tests — n-grammes de caractères (src/mosaic/ngrammes.py).

Un OCR substitue des glyphes : une lettre bascule vers celle qui lui ressemble
à l'impression, et le mot cesse d'exister pour une recherche par mot exact. Les n-grammes de caractères donnent
au moteur une notion de PRESQUE : deux mots à une lettre près partagent la
plupart de leurs fragments, donc se ressemblent encore.
"""

from mosaic.ngrammes import enrichir, ngrammes_du_mot


def test_un_mot_donne_ses_fragments_bornes():
    """Les bornes ^ et $ comptent : elles distinguent un début de mot d'un
    milieu, sinon « tableau » et « tabIeau » se ressembleraient autant que
    « tableau » et « tableaux »."""
    assert ngrammes_du_mot("chat", 3) == ["§^ch", "§cha", "§hat", "§at$"]


def test_le_prefixe_evite_toute_collision_avec_un_vrai_mot():
    """Sans marqueur, le trigramme « cha » entrerait en collision avec le mot
    « cha » s'il existait dans le corpus — et deux mondes se mélangeraient."""
    assert all(g.startswith("§") for g in ngrammes_du_mot("chat", 3))


def test_une_confusion_ocr_preserve_la_plupart_des_fragments():
    """Le cœur de l'affaire, chiffré : deux mots à une lettre près partagent la
    plupart de leurs fragments, là où l'égalité de mots donne 0.

    L'exemple est volontairement NEUTRE : un nom propre serait réécrit par la
    généricisation de l'export, et le test perdrait son sens sans prévenir."""
    a = set(ngrammes_du_mot("facture", 3))
    b = set(ngrammes_du_mot("facrure", 3))
    assert len(a & b) >= 3
    assert "facture" != "facrure"  # l'égalité exacte, elle, ne rattrape rien


def test_les_mots_courts_ne_sont_pas_fragmentes():
    """Fragmenter « de » ou « sur » n'apporte aucun pouvoir discriminant et
    noierait le flux : seuls les mots assez longs pour porter du sens sont
    enrichis."""
    assert ngrammes_du_mot("de", 3) == []
    assert ngrammes_du_mot("sur", 3) == []
    assert ngrammes_du_mot("panel", 3) != []


def test_enrichir_conserve_les_tokens_d_origine():
    """Les n-grammes s'AJOUTENT, ils ne remplacent jamais : un mot exact doit
    rester un mot exact, sinon on troquerait la précision contre la tolérance."""
    sortie = enrichir(["panel", "de", "test"], 3)
    assert sortie[:3] == ["panel", "de", "test"]
    assert len(sortie) > 3


def test_enrichir_est_deterministe_et_ordonne():
    a = enrichir(["disjoncteur", "courbe"], 3)
    b = enrichir(["disjoncteur", "courbe"], 3)
    assert a == b


def test_n_zero_ou_negatif_ne_fait_rien():
    """Garde : un réglage absurde laisse le flux intact plutôt que de le
    corrompre en silence."""
    assert enrichir(["panel"], 0) == ["panel"]
    assert enrichir(["panel"], -1) == ["panel"]


def test_flux_vide():
    assert enrichir([], 3) == []
