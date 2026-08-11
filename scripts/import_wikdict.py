"""Import filtré du dictionnaire WikDict eng-fra vers le format lexique Mosaic.

Source : WikDict eng-fra, format SQLite (le plus simple des formats proposés par
download.wikdict.com — TSV n'existe pas, les autres formats [stardict, TEI, kobo,
wdweb] demandent soit une lib externe soit un dump XML volumineux). Licence
CC-BY-SA — voir `src/mosaic/data/LICENSE_wikdict.txt` pour l'attribution complète.

URL exacte utilisée : https://download.wikdict.com/dictionaries/sqlite/2/en-fr.sqlite3
("2" est le lien vers la version la plus récente, 2_2026-06 au 09/08/2026,
fichier daté du 23-Jun-2026 par le serveur).

Filtres appliqués, dans l'ordre (spec docs/superpowers/specs/2026-08-09-mosaic-v1.1-design.md §B2) :

  1. Traduction principale uniquement (meilleur score WikDict) — la table
     `simple_translation` fournie par WikDict a déjà réduit chaque mot-clé à
     son score maximal (`max_score`) ; `trans_list` peut encore contenir
     plusieurs candidats ex-æquo à ce score, séparés par " | " — on retient
     le premier comme traduction "principale" (voir `best_translation`).
  2. Clé anglaise figurant dans la liste d'exclusion (mots ambigus du
     quotidien, mono-mot) → rejetée.
  2b. Clé mono-mot qui est elle-même un mot français (trouvé côté valeurs de
     la base WikDict — `build_french_vocabulary`) → rejetée, sauf paire
     identité (clé == valeur, inerte). Filtre homographe ajouté après coup
     (revue finale v1.1) : sans lui, des mots anglais courants qui sont AUSSI
     des mots-outils/noms français fréquents (ex. `pour`→`verser`,
     `son`→`fils`, `fin`→`nageoire`, `pendant`→`pendentif`) s'introduisent
     dans le lexique et, une fois canonicalisés, remplacent silencieusement
     ces mots dans du texte français normal.
  3. Clé déjà présente dans le noyau curaté (`lexicon_fr_en.json`) → rejetée,
     le noyau gagne.
  4. Clé ET valeur ≤ 4 tokens, où un "token" est un mot reconnu tel quel par
     le tokenizer réel de mosaic (`mosaic.tokenize.tokenize` — lettres/chiffres
     accentués, tirets internes). Toute entrée dont un mot contient une
     apostrophe, un symbole ou toute autre séquence hors de ce charset est
     rejetée ici : elle ne matcherait de toute façon jamais un texte réel
     tokenisé par mosaic.
  5. Minuscules, espaces → "_".

Filtre additionnel documenté (hors des 5 ci-dessus, ajouté après inspection) :
seuil de confiance `min_score` sur `simple_translation.max_score`. Sans lui,
l'import brut donne ~154 000 paires (tout le dictionnaire général WikDict :
noms propres, taxons biologiques, entrées à sens unique quasi identiques) —
loin de la cible réaliste de la spec ("+5 000 à 15 000 paires sûres"). Les
traductions restent correctes même à score bas (100-120), mais leur utilité
générale décroît (mots très rares, translittérations identité) ; `max_score`
est le signal de confiance calculé par WikDict lui-même (agrégation multi-
sources), donc le seuil naturel pour arbitrer "sûr" au sens volume+qualité.
MIN_SCORE=130 a été choisi par inspection de la distribution des scores
(~11 000 paires après filtres, centré dans la cible) — voir le rapport de
tâche pour le détail de l'inspection à la main.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from mosaic.lexicon import load_lexicon
from mosaic.tokenize import tokenize as _tokenize_word

SOURCE_URL = "https://download.wikdict.com/dictionaries/sqlite/2/en-fr.sqlite3"

MAX_TOKENS = 4

# Seuil de confiance WikDict (max_score) — voir justification dans le docstring du module.
MIN_SCORE = 130.0

# Liste d'exclusion minimale imposée par la spec §B2 (mots anglais courants ambigus).
EXCLUDED_WORDS = frozenset(
    """can may well state mean run set place order right left light back part
    sound present kind fair spring fall match court case charge current power
    load plug board""".split()
)


def read_simple_translation(
    db_path: Path, min_score: float = MIN_SCORE
) -> list[tuple[str, str, float]]:
    """Lit (written_rep, trans_list, max_score) depuis la base WikDict, ordre déterministe.

    `min_score` applique le seuil de confiance (voir docstring du module) : les
    entrées en dessous ne sont même pas lues, pas seulement rejetées plus loin.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT written_rep, trans_list, max_score FROM simple_translation "
            "WHERE max_score >= ? "
            "ORDER BY written_rep, max_score DESC, trans_list ASC",
            (min_score,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def best_translation(trans_list: str) -> str:
    """`trans_list` regroupe les candidats à score maximal, séparés par ' | ' : on
    retient le premier comme traduction principale (filtre 1 de la spec)."""
    return trans_list.split("|")[0].strip()


def clean_tokens(phrase: str, max_tokens: int = MAX_TOKENS) -> list[str] | None:
    """Normalise une phrase en tokens (minuscules) reconnus par le tokenizer mosaic.

    Retourne None si la phrase est vide, dépasse `max_tokens` mots, ou si un mot
    contient un caractère hors du charset du tokenizer réel (apostrophes, symboles,
    parenthèses…) — un tel mot ne matcherait jamais un texte réel tokenisé.
    """
    words = phrase.strip().split()
    if not words or len(words) > max_tokens:
        return None
    tokens: list[str] = []
    for w in words:
        found = _tokenize_word(w)
        if found != [w.lower()]:
            return None
        tokens.append(found[0])
    return tokens


def build_french_vocabulary(
    rows: list[tuple[str, str, float]], max_tokens: int = MAX_TOKENS
) -> frozenset[str]:
    """Ensemble de tous les tokens FR figurant côté valeurs de `rows` (tous les candidats
    de `trans_list`, pas seulement le meilleur) — sert à détecter les clés anglaises qui
    sont en réalité des mots français (filtre homographe, voir `filter_wikdict`)."""
    vocab: set[str] = set()
    for _written_rep, trans_list, _score in rows:
        for candidate in trans_list.split("|"):
            tokens = clean_tokens(candidate.strip(), max_tokens)
            if tokens:
                vocab.update(tokens)
    return frozenset(vocab)


def filter_wikdict(
    rows: list[tuple[str, str, float]],
    core_lexicon: dict[str, str],
    excluded_words: frozenset[str] = EXCLUDED_WORDS,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, str]:
    """Applique les filtres 2 à 5 de la spec §B2 (le filtre 1 est déjà satisfait par
    la source : `rows` ne porte qu'un candidat "meilleur score" par written_rep), plus le
    filtre homographe : une clé mono-mot qui est elle-même un mot français (trouvé côté
    valeurs de `rows`) est rejetée, sauf paire identité (clé == valeur, inerte)."""
    french_vocab = build_french_vocabulary(rows, max_tokens)
    result: dict[str, str] = {}
    for written_rep, trans_list, _score in rows:
        key_tokens = clean_tokens(written_rep, max_tokens)
        if key_tokens is None:
            continue
        if len(key_tokens) == 1 and key_tokens[0] in excluded_words:
            continue  # filtre 2 : mot ambigu du quotidien
        key = "_".join(key_tokens)
        if key in core_lexicon:
            continue  # filtre 3 : le noyau curaté gagne
        value_tokens = clean_tokens(best_translation(trans_list), max_tokens)
        if value_tokens is None:
            continue  # filtre 4 (valeur)
        value = "_".join(value_tokens)  # filtre 5
        if len(key_tokens) == 1 and key_tokens[0] in french_vocab and key != value:
            continue  # filtre homographe : la clé anglaise est elle-même un mot français
        if key not in result:  # déterminisme : premier candidat rencontré (tri SQL)
            result[key] = value
    return result


def build(
    db_path: Path, core_path: Path | None = None, min_score: float = MIN_SCORE
) -> dict[str, str]:
    core = load_lexicon(core_path)
    rows = read_simple_translation(db_path, min_score=min_score)
    return filter_wikdict(rows, core)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "db",
        nargs="?",
        default="data_externes/wikdict_en-fr.sqlite3",
        help="base SQLite WikDict eng-fra (téléchargée depuis %s)" % SOURCE_URL,
    )
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_SCORE,
        help="seuil de confiance WikDict (max_score), défaut %(default)s",
    )
    args = parser.parse_args(argv)

    lexicon = build(Path(args.db), min_score=args.min_score)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(lexicon, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{len(lexicon)} paires écrites dans {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
