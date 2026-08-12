"""Quantification BINAIRE des vecteurs rerank (quick win d'inventaire, veille 2026).

LA QUESTION : les empreintes model2vec du repêcheur (rerank.msrv, unitaires 256d float16,
cf. mosaic.rerank / mosaic.store.save_rerank) survivent-elles à la binarisation par signe
(1 bit/dim au lieu de 16) ? Si oui, la table rerank d'un index se range dans 16x moins
d'octets — pour l'index Alexandria c'est la différence entre « tient dans un cache L2 »
et « lit le disque ».

MÉCANISME TESTÉ : binarisation par signe (v > 0 -> +1, sinon -1), stockage PACKÉ en bits
(np.packbits, d/8 octets par document). Le score binaire est le produit scalaire des
vecteurs ±1/sqrt(d) — strictement équivalent à la distance de Hamming h par
cos_bin = (d - 2h)/d, ce que le script PROUVE numériquement via popcount
(np.bitwise_count sur le XOR des représentations packées) avant de mesurer. La qualité
passe par le VRAI chemin de recherche (idx.search(rerank=True), mélange
lambda*cos_grille + (1-lambda)*cos_m2v inchangé — cf. queries._rank_and_rerank) :
on substitue les vecteurs binarisés à idx.rerank_vecs et on binarise la requête par le
même signe (régime binaire-binaire symétrique, celui du stockage packé).

PRÉDICTIONS DÉCLARÉES AVANT MESURE (falsifiables) :
- P1 : perte <= 1 pt (0.01) sur R@10 ET MRR du banc recettes (12 requêtes pièges,
  40 docs) entre rerank float16 et rerank binaire ;
- P2 : compression 16x exactement en mémoire (float16 = 16 bits/dim -> 1 bit/dim) ;
  sur disque, rerank.msrv porte un en-tête, le facteur réel mesuré est rapporté.

USAGE :
    python research/rerank_binaire.py [corpus] [verite.jsonl]
défauts : le banc recettes du repo public jumeau (../mosaic-public/bench/), seul à
porter bench/corpus. Le modèle rerank est celui du moteur (potion-multilingual-128M,
surchargeable par MOSAIC_POTION_MODEL_DIR comme en prod — cf. rerank.py).
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from mosaic import rerank as rerank_module  # noqa: E402
from mosaic.index import Index  # noqa: E402


def _metrics(
    results: list[list[str]], relevant: list[list[str]]
) -> tuple[float, float]:
    """Recall@10 et MRR moyens — même définition que bench/run_bench.py._metrics."""
    recalls, rranks = [], []
    for hits, rel in zip(results, relevant, strict=True):
        rel_set = set(rel)
        recalls.append(len(rel_set & set(hits[:10])) / max(len(rel_set), 1))
        rr = 0.0
        for rank, h in enumerate(hits, start=1):
            if h in rel_set:
                rr = 1.0 / rank
                break
        rranks.append(rr)
    return float(np.mean(recalls)), float(np.mean(rranks))


def _binariser(mat: np.ndarray) -> np.ndarray:
    """Signe -> vecteurs ±1/sqrt(d) float32 (unitaires : le produit scalaire reste un
    cosinus, le chemin de recherche n'a rien à savoir)."""
    d = mat.shape[-1]
    return (np.where(mat > 0, 1.0, -1.0) / np.sqrt(d)).astype(np.float32)


def _lancer(
    idx: Index, queries: list[dict]
) -> tuple[float, float, float, list[list[str]]]:
    """(R@10, MRR, latence médiane ms, hits) du banc avec rerank via le VRAI moteur."""
    results, times = [], []
    for q in queries:
        t0 = time.perf_counter()
        hits = [h["id"] for h in idx.search(q["query"], k=10, rerank=True)]
        times.append((time.perf_counter() - t0) * 1000)
        results.append(hits)
    r10, mrr = _metrics(results, [q["relevant"] for q in queries])
    return r10, mrr, float(np.median(times)), results


def main() -> int:
    defaut_bench = RACINE.parent / "mosaic-public" / "bench"
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else defaut_bench / "corpus"
    verite = Path(sys.argv[2]) if len(sys.argv) > 2 else defaut_bench / "verite.jsonl"
    if not corpus.is_dir() or not verite.is_file():
        raise SystemExit(
            f"banc introuvable : {corpus} / {verite} (passer corpus et verite.jsonl "
            "en arguments — le banc recettes vit dans le repo public jumeau)"
        )
    queries = [
        json.loads(line)
        for line in verite.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with tempfile.TemporaryDirectory() as tmp:
        idx = Index.build(corpus, Path(tmp) / "index", rerank_vectors=True)
        assert idx.rerank_vecs is not None
        vecs = idx.rerank_vecs  # float32 en mémoire, arrondi float16 (cf. _round_f16)
        n, d = vecs.shape
        print(f"banc : {n} docs, {len(queries)} requêtes, vecteurs rerank {d}d")

        # --- référence float16 (l'état actuel), vrai chemin de recherche ---
        r10_f, mrr_f, lat_f, hits_f = _lancer(idx, queries)

        # --- représentation packée + preuve d'équivalence Hamming/produit ±1 ---
        bits = np.packbits(vecs > 0, axis=1)  # (n, d/8) uint8 — LE stockage mesuré
        signes = _binariser(vecs)
        rng = np.random.default_rng(0x20260812)
        temoin = _binariser(rng.standard_normal((7, d)))
        temoin_bits = np.packbits(temoin > 0, axis=1)
        hamming = np.bitwise_count(bits[:, None, :] ^ temoin_bits[None, :, :]).sum(
            axis=2
        )
        cos_hamming = (d - 2.0 * hamming) / d
        cos_produit = signes @ temoin.T
        ecart = float(np.abs(cos_hamming - cos_produit).max())
        print(f"équivalence Hamming(popcount) == produit ±1 : écart max {ecart:.2e}")
        assert ecart < 1e-5, "le stockage packé ne reproduit pas les scores ±1"

        # --- rerank binaire : mêmes requêtes, même moteur, canal m2v binarisé ---
        idx.rerank_vecs = signes
        encode_query_f16 = rerank_module.encode_query
        rerank_module.encode_query = (  # ty: ignore[invalid-assignment] — harnais de mesure
            lambda text: _binariser(encode_query_f16(text))
        )
        try:
            r10_b, mrr_b, lat_b, hits_b = _lancer(idx, queries)
        finally:
            rerank_module.encode_query = encode_query_f16
            idx.rerank_vecs = vecs

        # --- compression réelle ---
        octets_f16 = n * d * 2
        octets_bits = bits.nbytes
        msrv = Path(tmp) / "index" / "rerank.msrv"
        octets_msrv = msrv.stat().st_size if msrv.is_file() else 0

    print("\n--- qualité (banc recettes, rerank via idx.search) ---")
    print(f"float16 : R@10 {r10_f:.4f}  MRR {mrr_f:.4f}  latence méd. {lat_f:.2f} ms")
    print(f"binaire : R@10 {r10_b:.4f}  MRR {mrr_b:.4f}  latence méd. {lat_b:.2f} ms")
    delta_r10, delta_mrr = r10_f - r10_b, mrr_f - mrr_b
    print(f"perte   : R@10 {delta_r10:+.4f}  MRR {delta_mrr:+.4f}")
    for q, hf, hb in zip(queries, hits_f, hits_b, strict=True):
        if hf[:10] != hb[:10]:
            print(f"  top10 modifié : {q['query'][:60]!r} -> {hb[:3]}")

    print("\n--- compression ---")
    print(f"float16 en mémoire : {octets_f16} octets ({n}x{d}x2)")
    print(f"binaire packé      : {octets_bits} octets ({n}x{d}/8)")
    print(f"facteur mémoire    : {octets_f16 / octets_bits:.1f}x")
    if octets_msrv:
        print(
            f"rerank.msrv disque : {octets_msrv} octets "
            f"(facteur vs packé : {octets_msrv / octets_bits:.1f}x)"
        )

    p1 = delta_r10 <= 0.01 and delta_mrr <= 0.01
    p2 = octets_f16 / octets_bits == 16.0
    print(
        f"\nVerdict : P1 (perte <= 1 pt) {'TENUE' if p1 else 'RÉFUTÉE'} · "
        f"P2 (compression mémoire 16x) {'TENUE' if p2 else 'RÉFUTÉE'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
