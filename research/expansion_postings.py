"""Expansion DÉTERMINISTE des postings — un SPLADE sans réseau (veille 2026, champ n°1).

L'état de l'art 2026 a tranché : entre BM25 et le dense, la recherche éparse APPRISE
gagne parce qu'elle ferme le fossé de vocabulaire — un document/une requête reçoit des
termes qu'il ne contient pas (SPLADE et lignée). Ces modèles exigent un réseau ; or
notre moteur possède déjà l'ingrédient qu'ils apprennent : les PROFILS DE COOCCURRENCE.

HYPOTHÈSE : étendre la requête BM25 par les k voisins de cooccurrence de chaque terme
(appris du corpus, déterministes, pondérés par leur cosinus de profil) ferme une part
du fossé de vocabulaire SANS neurone, sans réseau, sans perte de souveraineté.

MÉCANISME (côté requête, zéro stockage) :
  score(doc) = BM25(requête) + W · Σ_t Σ_{v ∈ voisins(t,K)} cos(t,v) · BM25_terme(v)
Les voisins sont filtrés : df >= DF_MIN (un voisin rare est du bruit), hors stopwords,
cosinus >= SEUIL_COS. Balayage K ∈ {3, 5}, W ∈ {0.2, 0.4}.

PRÉDICTIONS DÉCLARÉES AVANT MESURE (falsifiables) :
- P1 (le terrain cible) : sur Alloprof (lexical, requêtes d'élèves — LE terrain BM25),
  l'expansion gagne >= +2 pts de R@10 sur BM25 pur pour au moins une config (le fossé
  de vocabulaire existe même en terrain lexical : paraphrases partielles).
- P2 (l'innocuité) : sur le banc recettes (contrôle), l'expansion ne coûte pas plus
  de 1 pt de R@10 — un canal qui aide ici et détruit là serait inutilisable en fusion.
- ÉCHEC de P1 => piste morte documentée (le fossé serait déjà fermé par la
  canonicalisation + collocations du flux de tokens).

Usage : python research/expansion_postings.py <corpus> <verite.jsonl>
        (recommandé : Alloprof puis recettes du repo public)
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from mosaic.calibration import _preparer
from mosaic.collocations import merge
from mosaic.encoder import STOPWORDS
from mosaic.lexicon import canonicalize, compile_lexicon
from mosaic.tokenize import tokenize

GRID = (64, 64, 3)
ECHANTILLON_DOCS = 500
MAX_REQUETES = 300
DF_MIN = 3
SEUIL_COS = 0.30
CONFIGS = [(3, 0.2), (3, 0.4), (5, 0.2), (5, 0.4)]  # (K voisins, W poids)


def _metriques(classements, verites):
    recalls, rrs = [], []
    for cls, rel in zip(classements, verites, strict=True):
        rel_set = set(rel)
        recalls.append(len(rel_set & set(cls[:10])) / max(1, len(rel_set)))
        rrs.append(
            next((1.0 / r for r, d in enumerate(cls, start=1) if d in rel_set), 0.0)
        )
    n = max(1, len(classements))
    return sum(recalls) / n, sum(rrs) / n


def main() -> int:
    corpus_src, verite_path = Path(sys.argv[1]), Path(sys.argv[2])
    requetes = [
        (str(o["query"]), [str(x) for x in o["relevant"]])
        for o in (
            json.loads(line)
            for line in verite_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    fichiers = sorted(p for p in corpus_src.iterdir() if p.is_file())
    dossier = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    corpus = corpus_src
    if len(fichiers) > ECHANTILLON_DOCS:
        corpus = Path(dossier.name) / "corpus"
        corpus.mkdir()
        garde = fichiers[:ECHANTILLON_DOCS]
        for p in garde:
            (corpus / p.name).write_bytes(p.read_bytes())
        noms = {p.name for p in garde}
        requetes = [(q, rel) for q, rel in requetes if all(r in noms for r in rel)][
            :MAX_REQUETES
        ]
    verites = [rel for _q, rel in requetes]

    # index_paths=False : les noms de fichiers du banc recettes NOMMENT les plats —
    # les garder donnerait la réponse à BM25 (MRR 1.0 mesuré, contrôle invalide) ;
    # le mécanisme se teste sur le CONTENU seul.
    docs, profiles, colloc, lexicon = _preparer(
        corpus, None, 300, GRID, index_paths=False
    )
    ids = [d for d, _ in docs]
    compiled = compile_lexicon(lexicon)

    from mosaic.bm25 import Bm25

    bm = Bm25.from_docs(docs)  # le canal BM25 du MOTEUR (scores() additif par terme)

    def tokens_requete(q: str) -> list[str]:
        return merge(merge(canonicalize(tokenize(q), compiled), colloc), colloc)

    toks_par_requete = [tokens_requete(q) for q, _rel in requetes]

    # Voisins de cooccurrence BATCHÉS pour l'union des tokens de requêtes (un appel
    # proches() par token referait un produit (V, 12288) à chaque fois). Filtres
    # minimaux script : df >= DF_MIN, hors stopwords, cos >= SEUIL_COS — la greffe
    # éventuelle utiliserait Profiles.proches (nettoyages plus riches).
    uniques = sorted(
        {
            t
            for toks in toks_par_requete
            for t in toks
            if t in profiles.rows and profiles.df.get(t, 0) >= DF_MIN
        }
    )
    v = len(profiles.rows)
    acc = profiles.acc[:v].astype(np.float32)
    normes = np.linalg.norm(acc, axis=1)
    normes[normes == 0] = 1.0
    vocab_liste = list(profiles.rows)
    admissible = np.array(
        [profiles.df.get(t, 0) >= DF_MIN and t not in STOPWORDS for t in vocab_liste]
    )
    voisins: dict[str, list[tuple[str, float]]] = {}
    kmax = max(k for k, _w in CONFIGS)
    for t in uniques:
        ligne = acc[profiles.rows[t]]
        nl = float(np.linalg.norm(ligne))
        if nl == 0.0:
            continue
        cos = (acc @ ligne) / (normes * nl)
        cos[~admissible] = -1.0
        cos[profiles.rows[t]] = -1.0  # pas soi-même
        top = np.argsort(-cos)[: kmax + 4]
        voisins[t] = [
            (vocab_liste[i], float(cos[i])) for i in top if cos[i] >= SEUIL_COS
        ][:kmax]

    print(
        f"corpus {corpus_src.name} : {len(ids)} docs, {len(requetes)} requêtes, "
        f"{len(uniques)} tokens de requête, voisins pour {len(voisins)}"
    )

    # BM25 pur (référence)
    cls_ref = [
        [ids[i] for i in np.argsort(-bm.scores(toks))[:10]] for toks in toks_par_requete
    ]
    r_ref, m_ref = _metriques(cls_ref, verites)
    print(f"BM25 pur                R@10 {r_ref:.4f}  MRR {m_ref:.4f}")

    for k, w in CONFIGS:
        cls = []
        for toks in toks_par_requete:
            s = bm.scores(toks).astype(np.float64)
            for t in set(toks):
                for voisin, cos in voisins.get(t, [])[:k]:
                    s += w * cos * bm.scores([voisin]).astype(np.float64)
            cls.append([ids[i] for i in np.argsort(-s)[:10]])
        r, m = _metriques(cls, verites)
        print(
            f"expansion K={k} W={w}     R@10 {r:.4f} (Δ {100 * (r - r_ref):+.2f} pt)  "
            f"MRR {m:.4f} (Δ {100 * (m - m_ref):+.2f} pt)"
        )
    print(
        "\nPrédictions (en-tête) : P1 >= +2 pts R@10 sur Alloprof pour une config ; "
        "P2 : >= -1 pt sur recettes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
