import json
from pathlib import Path

from mosaic.lexicon import DEFAULT_LEXICON, canonicalize, compile_lexicon, load_lexicon
from mosaic.tokenize import tokenize

WIKDICT_LEXICON = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mosaic"
    / "data"
    / "lexicon_wikdict_fr_en.json"
)


def test_lexique_embarque_charge():
    lex = load_lexicon()
    assert DEFAULT_LEXICON.exists()
    assert lex["breaker"] == "disjoncteur"


def test_compile_cle_et_valeur_multi_mots():
    compiled = compile_lexicon(
        {"junction_box": "boîte_de_dérivation", "breaker": "disjoncteur"}
    )
    assert compiled[("junction", "box")] == ["boîte", "dérivation"]
    assert compiled[("breaker",)] == ["disjoncteur"]


def test_canonicalize_glouton_multi_mots():
    compiled = compile_lexicon({"junction_box": "boîte_de_dérivation", "box": "carton"})
    out = canonicalize(["install", "junction", "box", "here"], compiled)
    assert out == ["install", "boîte", "dérivation", "here"]  # le plus long gagne


def test_canonicalize_mono_mot_et_inconnus():
    compiled = compile_lexicon({"breaker": "disjoncteur"})
    assert canonicalize(["breaker", "tripped"], compiled) == ["disjoncteur", "tripped"]
    assert canonicalize(["a", "b"], {}) == ["a", "b"]


def test_compile_ignore_les_valeurs_sans_mots_pleins():
    compiled = compile_lexicon({"weird": "de_la", "breaker": "disjoncteur"})
    assert ("weird",) not in compiled
    assert compiled[("breaker",)] == ["disjoncteur"]
    # et le flux reste intact pour la clé ignorée :
    assert canonicalize(["weird", "ok"], compiled) == ["weird", "ok"]


def test_load_lexicon_fusionne_extra_sous_le_noyau(tmp_path):
    extra_path = tmp_path / "extra.json"
    extra_path.write_text(
        json.dumps({"breaker": "casseur", "gadget": "gadget"}), encoding="utf-8"
    )
    lex = load_lexicon(extra=extra_path)
    assert lex["breaker"] == "disjoncteur"  # le noyau gagne sur la collision
    assert lex["gadget"] == "gadget"  # l'extra comble un mot absent du noyau


def test_load_lexicon_sans_extra_inchange():
    assert load_lexicon() == load_lexicon(extra=None)


def test_compile_ignore_les_cles_stopwords_francais():
    compiled = compile_lexicon(
        {"pour": "verser", "son": "fils", "breaker": "disjoncteur"}
    )
    assert ("pour",) not in compiled and ("son",) not in compiled
    assert compiled[("breaker",)] == ["disjoncteur"]


def test_lexique_wikdict_ne_reecrit_pas_le_francais():
    # après régénération, avec le lexique fusionné complet :
    lex = load_lexicon(extra=WIKDICT_LEXICON)
    compiled = compile_lexicon(lex)
    tokens = tokenize(
        "devis pour la fin du chantier : pose du four et peinture du coin cuisine"
    )
    out = canonicalize(tokens, compiled)
    # 'pour', 'fin', 'four', 'coin', 'pose' doivent rester INCHANGÉS
    for mot in ("pour", "fin", "four", "coin", "pose"):
        assert mot in out, f"{mot!r} a disparu de la sortie canonicalisée : {out}"


def test_lexique_sans_cles_dupliquees():
    pairs: list[tuple[str, object]] = []
    json.loads(
        DEFAULT_LEXICON.read_text(encoding="utf-8"),
        object_pairs_hook=lambda p: pairs.extend(p) or dict(p),
    )
    keys = [k for k, _ in pairs]
    dupes = {k for k in keys if keys.count(k) > 1}
    assert dupes == set(), f"clés dupliquées: {sorted(dupes)}"
