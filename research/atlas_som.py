"""Étape 1 du chantier atlas sémantique (briefing #367) — prototype d'assignation APPRISE.

HYPOTHÈSE : remplacer le hachage SHA (emplacement aléatoire des tokens) par une carte
auto-organisée (SOM/Kohonen, apprise sur les profils de cooccurrence PPMI+SVD déjà
calculés) place les tokens sémantiquement proches sur des cellules voisines — un
document devient une carte de chaleur sur un atlas de concepts, et un FLOU GAUSSIEN
(adoucissement sémantique) permet à une paraphrase de toucher les cellules voisines
de celles du document.

CRITÈRE DU BRIEFING (à battre) : similarité convolutive sur atlas > cosinus plat
(le moteur Mosaic actuel, défauts de build) sur les 12 pièges de paraphrase du banc
recettes, SANS perdre plus d'1 point de recall sur le jeu lexical de contrôle.

PROTOCOLE :
- corpus = bench/corpus (40 recettes), préparation IDENTIQUE au build (canonicalisation
  + collocations + profils PPMI + lissage SVD — via calibration._preparer, garanti en
  phase) ;
- SOM par lots, déterministe (graine fixe, itérations fixes, voisinage gaussien
  décroissant) sur les vecteurs de profil des tokens (réduits par SVD pour la vitesse) ;
- carte de chaleur document/requête = tf×idf déposé sur la cellule de chaque token,
  normalisée ; similarité = cosinus des cartes floutées (σ balayé, σ=0 = atlas brut) ;
- baseline = Index.build défauts + search (le « cosinus plat » réellement livré) ;
- jeux : les 12 pièges de paraphrase (bench/verite.jsonl) + contrôle lexical
  déterministe (top tf×idf de chaque document comme requête, 40 requêtes).

HONNÊTETÉ : ce prototype teste le MÉCANISME atlas seul (cartes de chaleur), pas une
greffe dans l'encodeur complet — si l'atlas seul ne bat pas le moteur complet sur les
pièges, la greffe se discute quand même (canal additif possible) : les résultats
instruisent le dossier, le verdict d'enterrement ou de poursuite se prend avec Alex.
Prior déclaré dans le briefing : faible à moyen (Kanerva a choisi l'aléatoire pour
préserver la capacité de superposition — dégradation mesurée à l'étape 2).
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mosaic.calibration import _preparer
from mosaic.collocations import merge
from mosaic.index import Index
from mosaic.lexicon import canonicalize, compile_lexicon
from mosaic.tokenize import tokenize

RACINE = Path(__file__).resolve().parent.parent
# banc recettes par défaut (12 pièges de paraphrase) ; surchargeable :
# python research/atlas_som.py [corpus] [verite.jsonl]
CORPUS = Path(sys.argv[1]) if len(sys.argv) > 1 else RACINE / "bench" / "corpus"
VERITE = Path(sys.argv[2]) if len(sys.argv) > 2 else RACINE / "bench" / "verite.jsonl"

GRID_BUILD = (64, 64, 3)  # profils appris à la dimension du build par défaut
# cellules de l'atlas — surchargeable (MOSAIC_ATLAS=64) pour tester l'effet des
# collisions token/cellule (22k tokens sur 1 024 cellules = 21 tokens/cellule sur Alloprof)
_ATLAS_COTE = int(os.environ.get("MOSAIC_ATLAS", "32"))
ATLAS = (_ATLAS_COTE, _ATLAS_COTE)
DIM_SOM = 128  # réduction SVD des profils avant SOM (vitesse)
ITERATIONS = 30
SIGMA_DEBUT, SIGMA_FIN = 8.0, 0.8  # voisinage gaussien décroissant (en cellules)
GRAINE = int(sys.argv[3]) if len(sys.argv) > 3 else 42  # robustesse : varier la graine
SIGMAS_FLOU = (0.0, 0.35, 0.75, 1.5, 3.0)  # 0.0 = atlas brut sans convolution
TERMES_LEXICAUX = 4  # taille des requêtes du contrôle lexical
K_EVAL = 10
# gros corpus (ex. Alloprof) : échantillon DÉTERMINISTE (fichiers triés, premiers N) pour
# tenir en RAM/temps — les requêtes dont un document pertinent sort de l'échantillon sont
# écartées (jamais comptées comme des échecs), tailles rapportées dans la sortie
SEUIL_ECHANTILLON = 600
ECHANTILLON_DOCS = 500
MAX_REQUETES = 300
MAX_LEXICALES = 100


def _metriques(classements: list[list[str]], verites: list[list[str]]) -> dict:
    """recall@1 est la métrique DISCRIMINANTE ici : sur 40 docs, recall@10 sature à 1.0
    (25 % du corpus dans le top-10) et ne départage rien."""
    rr, hits, hits1 = [], 0, 0
    for ids, rel in zip(classements, verites):
        rangs = [ids.index(d) + 1 for d in rel if d in ids]
        rr.append(1.0 / min(rangs) if rangs else 0.0)
        hits += 1 if rangs else 0
        hits1 += 1 if rangs and min(rangs) == 1 else 0
    n = max(1, len(classements))
    return {
        "recall": round(hits / n, 4),
        "r1": round(hits1 / n, 4),
        "mrr": round(sum(rr) / n, 4),
    }


def _projeter(mat: np.ndarray, k: int, graine: int = GRAINE) -> np.ndarray:
    """SVD RANDOMISÉE seedée (Halko et al. 2011) : la qualité des composantes principales
    (mesurée nettement meilleure qu'une projection JL brute pour organiser la carte —
    atlas brut 0.92 vs 0.75 R@1 sur les pièges recettes) au coût mémoire d'une projection
    — une SVD pleine sur V×12288 demanderait ~4 Go de matrices de travail à 40k tokens.
    Déterministe (graine fixe) ; sur-échantillonnage 2k pour la précision du sous-espace."""
    rng = np.random.default_rng(graine)
    m32 = mat.astype(np.float32)
    omega = rng.standard_normal((m32.shape[1], min(2 * k, m32.shape[1]))).astype(
        np.float32
    )
    q, _r = np.linalg.qr(m32 @ omega)  # base orthonormée du sous-espace dominant
    b = q.T @ m32  # (2k, dim) — petit
    ub, s, _vt = np.linalg.svd(b, full_matrices=False)
    return ((q @ ub[:, :k]) * s[:k]).astype(np.float32)


def _normaliser(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


def _som(features: np.ndarray, graine: int = GRAINE) -> np.ndarray:
    """SOM par lots, déterministe. Rend la cellule (index plat) de chaque token."""
    h, w = ATLAS
    cellules = h * w
    rng = np.random.default_rng(graine)
    feats = _normaliser(features)
    protos = feats[rng.choice(len(feats), size=cellules, replace=len(feats) < cellules)]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float64)  # (C, 2)
    d2_cellules = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)  # (C, C)
    for it in range(ITERATIONS):
        sigma = SIGMA_DEBUT * (SIGMA_FIN / SIGMA_DEBUT) ** (it / max(1, ITERATIONS - 1))
        bmu = np.argmax(feats @ _normaliser(protos).T, axis=1)
        voisinage = np.exp(-d2_cellules[bmu] / (2.0 * sigma * sigma))  # (T, C)
        masse = voisinage.sum(axis=0)  # (C,)
        masse[masse == 0] = 1.0
        protos = (voisinage.T @ feats) / masse[:, None]
    return np.argmax(feats @ _normaliser(protos).T, axis=1)


def _noyau_gaussien(sigma: float) -> np.ndarray:
    rayon = max(1, int(round(3 * sigma)))
    x = np.arange(-rayon, rayon + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def _flouter(cartes: np.ndarray, sigma: float) -> np.ndarray:
    """Flou gaussien séparable sur un lot de cartes (N, H, W)."""
    if sigma <= 0.0:
        return cartes
    k = _noyau_gaussien(sigma)
    out = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 2, cartes)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 1, out)


def _carte(
    tokens: list[str], position: dict[str, int], idf, h: int, w: int
) -> np.ndarray:
    m = np.zeros(h * w, dtype=np.float64)
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    for t, c in tf.items():
        p = position.get(t)
        if p is not None:
            m[p] += c * idf(t)
    return m.reshape(h, w)


def _classer(cartes_docs: np.ndarray, carte_q: np.ndarray, ids: list[str]) -> list[str]:
    plat_docs = cartes_docs.reshape(len(cartes_docs), -1)
    plat_q = carte_q.ravel()
    nd = np.linalg.norm(plat_docs, axis=1)
    nq = np.linalg.norm(plat_q)
    nd[nd == 0] = 1.0
    scores = (plat_docs @ plat_q) / (nd * (nq if nq else 1.0))
    return [ids[i] for i in np.argsort(-scores, kind="stable")[:K_EVAL]]


def main() -> int:
    pieges: list[tuple[str, list[str]]] = [
        (str(o["query"]), [str(x) for x in o["relevant"]])
        for o in (
            json.loads(line)
            for line in VERITE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    corpus = CORPUS
    dossier_echantillon = tempfile.TemporaryDirectory()  # vit jusqu'à la fin du run
    fichiers = sorted(p for p in CORPUS.iterdir() if p.is_file())
    echantillonne = len(fichiers) > SEUIL_ECHANTILLON
    if echantillonne:
        corpus = Path(dossier_echantillon.name) / "corpus"
        corpus.mkdir()
        garde = fichiers[:ECHANTILLON_DOCS]
        for p in garde:
            (corpus / p.name).write_bytes(p.read_bytes())
        noms = {p.name for p in garde}
        pieges = [(q, rel) for q, rel in pieges if all(r in noms for r in rel)][
            :MAX_REQUETES
        ]
    docs, profiles, colloc, lexicon = _preparer(corpus, None, 300, GRID_BUILD)
    ids = [d for d, _ in docs]
    compiled = compile_lexicon(lexicon)

    def tokens_requete(q: str) -> list[str]:
        return merge(merge(canonicalize(tokenize(q), compiled), colloc), colloc)

    # contrôle lexical déterministe : les termes les plus distinctifs de chaque document
    lexicales: list[tuple[str, list[str]]] = []
    for doc_id, tokens in docs[:MAX_LEXICALES]:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        tops = sorted(
            ((c * profiles.idf(t), t) for t, c in tf.items()),
            key=lambda x: (-x[0], x[1]),
        )
        lexicales.append((" ".join(t for _s, t in tops[:TERMES_LEXICAUX]), [doc_id]))

    # ------- baseline : le moteur livré (cosinus plat, défauts de build) -------
    with tempfile.TemporaryDirectory() as tmp:
        idx = Index.build(corpus, Path(tmp) / "idx", grid=GRID_BUILD)
        idx.chauffer_recherche()
        base = {}
        for nom, jeu in (("pieges", pieges), ("lexical", lexicales)):
            cls = [[h["id"] for h in idx.search(q, k=K_EVAL)] for q, _rel in jeu]
            base[nom] = _metriques(cls, [rel for _q, rel in jeu])

    # ------- atlas : SOM sur les profils de cooccurrence -------
    v = len(profiles.rows)
    features = _projeter(profiles.acc[:v], DIM_SOM)
    bmu = _som(features)
    position = {t: int(bmu[i]) for t, i in profiles.rows.items()}
    h, w = ATLAS
    occupation = len(set(position.values()))

    cartes_docs = np.stack(
        [_carte(toks, position, profiles.idf, h, w) for _d, toks in docs]
    )
    resultats: dict[float, dict] = {}
    for sigma in SIGMAS_FLOU:
        cartes_f = _flouter(cartes_docs, sigma)
        entree = {}
        for nom, jeu in (("pieges", pieges), ("lexical", lexicales)):
            cls = []
            for q, _rel in jeu:
                cq = _flouter(
                    _carte(tokens_requete(q), position, profiles.idf, h, w)[None], sigma
                )[0]
                cls.append(_classer(cartes_f, cq, ids))
            entree[nom] = _metriques(cls, [rel for _q, rel in jeu])
        resultats[sigma] = entree

    # ------- rapport -------
    print("=== Étape 1 — atlas sémantique (SOM) vs cosinus plat (briefing #367) ===")
    print(
        f"corpus : {len(ids)} docs"
        + (f" (échantillon déterministe de {CORPUS.name})" if echantillonne else "")
        + f", vocabulaire {v} tokens, atlas {h}x{w} "
        f"({occupation} cellules occupées), SOM {ITERATIONS} it., graine {GRAINE}"
    )
    print(
        f"jeux : {len(pieges)} pièges/requêtes réelles, {len(lexicales)} contrôles lexicaux"
    )
    print()
    print(
        f"{'système':<28} {'pièges R@1':>11} {'MRR':>7} {'R@10':>6} {'lexical R@1':>12} {'MRR':>7}"
    )
    print(
        f"{'mosaic plat (baseline)':<28} {base['pieges']['r1']:>11} {base['pieges']['mrr']:>7} "
        f"{base['pieges']['recall']:>6} {base['lexical']['r1']:>12} {base['lexical']['mrr']:>7}"
    )
    for sigma, entree in resultats.items():
        nom = "atlas brut (σ=0)" if sigma == 0 else f"atlas convolutif σ={sigma}"
        print(
            f"{nom:<28} {entree['pieges']['r1']:>11} {entree['pieges']['mrr']:>7} "
            f"{entree['pieges']['recall']:>6} {entree['lexical']['r1']:>12} {entree['lexical']['mrr']:>7}"
        )
    print()
    # critère jugé sur le MRR des pièges (le R@10 sature sur 40 docs) ; garde-fou lexical
    # sur le MRR aussi, même tolérance d'1 point
    gagnants = [
        s
        for s, e in resultats.items()
        if e["pieges"]["mrr"] > base["pieges"]["mrr"]
        and e["lexical"]["mrr"] >= base["lexical"]["mrr"] - 0.01
    ]
    if gagnants:
        print(
            f"CRITÈRE DU BRIEFING ATTEINT pour σ ∈ {gagnants} — à confirmer à l'étape 2"
        )
        print("(capacité de superposition) et à discuter avec Alex avant toute greffe.")
    else:
        print(
            "Critère du briefing NON atteint sur ce banc — résultats posés pour DISCUSSION"
        )
        print(
            "avec Alex (variantes non testées : autre taille d'atlas, autres σ, canal"
        )
        print(
            "additif plutôt que remplacement). Aucun enterrement sans cette discussion."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
