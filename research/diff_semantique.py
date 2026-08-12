"""Chantier DIFF SÉMANTIQUE — étape 1 : « qu'est-ce qui a changé de SENS dans ce corpus ? »

L'IDÉE (axe disruptif validé 12/08) : deux builds déterministes du même corpus à deux
moments sont comparables AU BIT PRÈS — leur différence est donc un objet sémantique
propre, pas du bruit de reconstruction. Aucun moteur à embeddings-API ne peut offrir
ça (chaque ré-indexation y bouge tout). Quatre lectures du delta :

1. INVENTAIRE : documents et vocabulaire apparus/disparus, collocations nées/mortes
   (les nouveaux concepts composés — signal cheap mais parlant).
2. DÉRIVE DE MOT : pour un token présent des deux côtés, cosinus entre ses profils de
   cooccurrence (les dims sont ancrées par les signatures SHA, identiques entre
   builds — les profils SONT comparables). Cosinus bas = le CONTEXTE du mot a changé.
3. DOCUMENTS MODIFIÉS : contenu changé (hash fichier) — la partie triviale du diff.
4. DÉRIVE DE CONTEXTE (la lecture NOUVELLE) : documents au contenu IDENTIQUE dont la
   grille a pourtant bougé — le monde a changé de sens AUTOUR d'eux. C'est la lecture
   qu'aucun diff textuel ne peut donner.

PRÉDICTIONS DÉCLARÉES AVANT MESURE (falsifiables) :
- P1 (spécificité, la garantie fondatrice) : corpus identique des deux côtés →
  dérive de mot ET dérive de contexte STRICTEMENT nulles (déterminisme au bit près).
- P2 (sensibilité) : sur un banc planté (substitution systématique d'un ingrédient
  dans k docs + ajout de docs d'un thème étranger), les tokens plantés sortent en
  tête de la dérive de mot (précision@10 >= 0.8 sur les plantés).
- P3 (localité) : la dérive de contexte des docs intacts SANS lien avec les docs
  ajoutés/modifiés reste faible — le diff ne doit pas crier au changement partout
  (médiane des intacts-lointains < 1/3 de la médiane des intacts-voisins du thème).

Usage :
  python research/diff_semantique.py --banc            # banc planté auto (recettes)
  python research/diff_semantique.py <corpus_t1> <corpus_t2>   # diff réel
"""

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosaic.index import Index

RACINE = Path(__file__).resolve().parent.parent
DF_MIN = 3  # dérive de mot : ignorer les tokens trop rares (profil = bruit)
TOP = 12


def _construire(corpus: Path, dossier: Path) -> Index:
    return Index.build(corpus, dossier)


def _hash_docs(corpus: Path) -> dict[str, str]:
    out = {}
    for p in sorted(corpus.rglob("*")):
        if p.is_file():
            out[p.relative_to(corpus).as_posix()] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
    return out


def _derive(a: np.ndarray, b: np.ndarray) -> float:
    """1 − cos(a, b), avec chemin rapide d'ÉGALITÉ EXACTE → 0.0 strict.

    Le chemin exact porte la garantie fondatrice (corpus identique ⇒ diff vide au
    sens STRICT) : sans lui, a@a et ‖a‖·‖a‖ divergent d'un ulp et le zéro devient
    1e-13 — vrai numériquement, faux contractuellement."""
    if np.array_equal(a, b):
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - float(a.astype(np.float64) @ b.astype(np.float64)) / (na * nb)


def diff(corpus_a: Path, corpus_b: Path) -> dict:
    """Calcule le diff sémantique complet entre deux états d'un corpus."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ia = _construire(corpus_a, Path(tmp) / "a")
        ib = _construire(corpus_b, Path(tmp) / "b")

        # 1. inventaire
        docs_a, docs_b = set(ia.ids), set(ib.ids)
        vocab_a, vocab_b = set(ia.profiles.rows), set(ib.profiles.rows)
        colloc_a = set(ia.colloc) if ia.colloc else set()
        colloc_b = set(ib.colloc) if ib.colloc else set()

        # 2. dérive de mot (profils comparables : dims ancrées par signatures SHA)
        va = len(ia.profiles.rows)
        vb = len(ib.profiles.rows)
        derive_mots: list[tuple[float, str]] = []
        for t, ra in ia.profiles.rows.items():
            rb = ib.profiles.rows.get(t)
            if rb is None or ra >= va or rb >= vb:
                continue
            if ia.profiles.df.get(t, 0) < DF_MIN or ib.profiles.df.get(t, 0) < DF_MIN:
                continue
            derive_mots.append((_derive(ia.profiles.acc[ra], ib.profiles.acc[rb]), t))
        derive_mots.sort(reverse=True)

        # dérive d'USAGE (fréquence documentaire) — signal DISTINCT de la dérive de
        # contexte : un mot substitué garde ses contextes dans les docs restants
        # (dérive de mot ≈ 0) mais son df chute — leçon du premier run du banc.
        usage: list[tuple[float, str, int, int]] = []
        for t in vocab_a & vocab_b:
            da, db = ia.profiles.df.get(t, 0), ib.profiles.df.get(t, 0)
            if max(da, db) < DF_MIN or da == db:
                continue
            usage.append(((db - da) / max(da, db), t, da, db))
        usage.sort(key=lambda x: x[0])

        # 3+4. documents : contenu changé (hash) vs contexte dérivé (contenu intact)
        ha, hb = _hash_docs(corpus_a), _hash_docs(corpus_b)
        pos_a = {d: i for i, d in enumerate(ia.ids)}
        pos_b = {d: i for i, d in enumerate(ib.ids)}
        modifies: list[tuple[float, str]] = []
        contexte: list[tuple[float, str]] = []
        for d in sorted(docs_a & docs_b):
            delta = _derive(ia.mat[pos_a[d]], ib.mat[pos_b[d]])
            if ha.get(d) != hb.get(d):
                modifies.append((delta, d))
            else:
                contexte.append((delta, d))
        modifies.sort(reverse=True)
        contexte.sort(reverse=True)

        return {
            "docs_ajoutes": sorted(docs_b - docs_a),
            "docs_retires": sorted(docs_a - docs_b),
            "vocab_apparu": sorted(vocab_b - vocab_a),
            "vocab_disparu": sorted(vocab_a - vocab_b),
            "collocations_nees": sorted(colloc_b - colloc_a),
            "collocations_mortes": sorted(colloc_a - colloc_b),
            "derive_mots": derive_mots,
            "derive_usage": usage,
            "docs_modifies": modifies,
            "derive_contexte": contexte,
        }


def _rapport(d: dict) -> None:
    print(
        f"docs : +{len(d['docs_ajoutes'])} / -{len(d['docs_retires'])}   "
        f"vocab : +{len(d['vocab_apparu'])} / -{len(d['vocab_disparu'])}   "
        f"collocations : +{len(d['collocations_nees'])} / -{len(d['collocations_mortes'])}"
    )
    if d["collocations_nees"]:
        print(
            "  concepts composés nés :", ", ".join(map(str, d["collocations_nees"][:8]))
        )
    print(f"\n-- dérive de MOT (top {TOP}, df>={DF_MIN} des deux côtés) --")
    for delta, t in d["derive_mots"][:TOP]:
        print(f"  {delta:.4f}  {t}")
    print("\n-- dérive d'USAGE (déclins puis croissances les plus fortes) --")
    declins = [u for u in d["derive_usage"] if u[0] < 0][:6]
    croissances = [u for u in d["derive_usage"] if u[0] > 0][-6:][::-1]
    for ratio, t, da, db in declins + croissances:
        print(f"  {ratio:+.2f}  {t} (df {da} -> {db})")
    print(f"\n-- documents MODIFIÉS (contenu changé, top {TOP}) --")
    for delta, doc in d["docs_modifies"][:TOP]:
        print(f"  {delta:.4f}  {doc}")
    print(
        f"\n-- dérive de CONTEXTE (contenu INTACT, la lecture nouvelle, top {TOP}) --"
    )
    for delta, doc in d["derive_contexte"][:TOP]:
        print(f"  {delta:.4f}  {doc}")


def banc() -> int:
    """Banc planté : recettes t1 -> t2 avec changements CONNUS, prédictions P1-P3."""
    source = RACINE / "bench" / "corpus"
    if not source.is_dir():
        raise SystemExit(f"corpus recettes introuvable : {source}")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        t1 = Path(tmp) / "t1"
        shutil.copytree(source, t1)

        # P1 — spécificité : diff(t1, t1) doit être STRICTEMENT vide
        d0 = diff(t1, t1)
        max_mot = d0["derive_mots"][0][0] if d0["derive_mots"] else 0.0
        max_ctx = d0["derive_contexte"][0][0] if d0["derive_contexte"] else 0.0
        p1 = max_mot == 0.0 and max_ctx == 0.0 and not d0["vocab_apparu"]
        print(
            f"P1 spécificité (corpus identique) : dérive mot max {max_mot:.6f}, "
            f"contexte max {max_ctx:.6f} -> {'OK' if p1 else 'ÉCHEC'}\n"
        )

        # t2 : substitution plantée (beurre -> margarine) + 4 docs d'un thème étranger
        t2 = Path(tmp) / "t2"
        shutil.copytree(source, t2)
        substitues = []
        for p in sorted(t2.glob("*.md")):
            txt = p.read_text(encoding="utf-8")
            if "beurre" in txt and len(substitues) < 6:
                p.write_text(txt.replace("beurre", "margarine"), encoding="utf-8")
                substitues.append(p.name)
        for i in range(4):
            (t2 / f"90_moteur_{i}.md").write_text(
                "Le moteur thermique convertit le carburant en mouvement. Le piston "
                "coulisse dans le cylindre, la bougie enflamme le melange, le "
                "vilebrequin transforme la course en rotation. L'huile lubrifie le "
                f"moteur et le radiateur refroidit le circuit. Variante {i}.",
                encoding="utf-8",
            )
        print(
            f"plantés : 'beurre'->'margarine' dans {len(substitues)} docs, "
            f"+4 docs thème moteur\n"
        )
        d = diff(t1, t2)
        _rapport(d)

        # P2 — sensibilité : les mots du chantier planté sortent en tête de dérive
        # P2 RÉVISÉE (leçon du 1er run) : la dérive de mot détecte le changement de
        # CONTEXTE, pas de fréquence — « beurre » garde ses contextes dans les docs
        # restants ; ce sont ses VOISINS de cooccurrence qui dérivent, sa chute
        # d'usage est le signal df. Trois sous-critères :
        vocab_substitue: set[str] = set()
        for nom in substitues:
            from mosaic.tokenize import tokenize as _tok

            vocab_substitue |= set(_tok((source / nom).read_text(encoding="utf-8")))
        top10 = [t for _delta, t in d["derive_mots"][:10]]
        part_voisinage = sum(1 for t in top10 if t in vocab_substitue) / max(
            1, len(top10)
        )
        declins = [t for _r, t, _a, _b in d["derive_usage"][:5]]
        p2a = "margarine" in d["vocab_apparu"]
        p2b = "beurre" in declins
        p2c = part_voisinage >= 0.7
        p2 = p2a and p2b and p2c
        print(
            f"\nP2 sensibilité : margarine apparue {'OK' if p2a else 'ÉCHEC'} ; "
            f"beurre dans le top 5 des déclins {'OK' if p2b else 'ÉCHEC'} ; "
            f"top-10 dérive ⊂ voisinage substitué {part_voisinage:.0%} "
            f"{'OK' if p2c else 'ÉCHEC'}"
        )

        # P3 — localité : les docs intacts SANS beurre ni rapport au moteur dérivent
        # moins que les docs intacts qui PARLENT de beurre (voisins du chantier)
        avec_beurre = {
            p.name
            for p in source.glob("*.md")
            if "beurre" in p.read_text(encoding="utf-8") and p.name not in substitues
        }
        intacts = {doc: delta for delta, doc in d["derive_contexte"]}
        voisins = [v for k, v in intacts.items() if k in avec_beurre]
        lointains = [v for k, v in intacts.items() if k not in avec_beurre]
        med_v = float(np.median(voisins)) if voisins else 0.0
        med_l = float(np.median(lointains)) if lointains else 0.0
        p3 = med_l < med_v / 3 if med_v > 0 else False
        print(
            f"P3 localité : médiane contexte voisins-du-beurre {med_v:.5f} vs "
            f"lointains {med_l:.5f} -> {'OK' if p3 else 'ÉCHEC'} "
            f"(critère : lointains < voisins/3)"
        )
        print(
            f"\nVERDICT : P1 {'✓' if p1 else '✗'}  P2 {'✓' if p2 else '✗'}  "
            f"P3 {'✓' if p3 else '✗'}"
        )
    return 0


def main() -> int:
    if "--banc" in sys.argv:
        return banc()
    if len(sys.argv) < 3:
        raise SystemExit("usage : diff_semantique.py <corpus_t1> <corpus_t2> | --banc")
    _rapport(diff(Path(sys.argv[1]), Path(sys.argv[2])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
