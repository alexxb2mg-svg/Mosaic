"""Piste A — reranking par CROSS-ENCODEUR (BGE-reranker-v2-m3, GGUF Q8_0).

Le repêcheur actuel est un bi-encodeur (potion-128M, cosinus). Un cross-encodeur
lit la paire requête+document ENSEMBLE — structurellement supérieur en précision
de tête. Toute l'infrastructure existe (fenêtre de candidats) ; seul le scoreur
change, servi par llama-server `--reranking`.

PROTOCOLE (FILE_RECHERCHE piste A) : Alloprof, reranger les 50 premiers candidats
du meilleur pipeline actuel (fusion 4 canaux) et comparer R@10 / MRR.

DÉVIATION DÉCLARÉE AVANT LA MESURE (14/08, 01h40) : le protocole d'origine dit
2 316 requêtes ; à ~15 s de rerank par requête sur ce CPU, le banc complet
coûterait ~10 h. On mesure sur un ÉCHANTILLON de 300 requêtes tiré au sort
(graine 42, déclarée ici) — la taille de SciFact, suffisante pour trancher un
écart de 3 points de MRR. Les DEUX bras sont mesurés sur le même échantillon.

PRÉDICTIONS DÉCLARÉES (13/08, FILE_RECHERCHE — inchangées) :
  P-A1 — MRR : +3 points ou plus, rappel inchangé (il ne fait que réordonner).
  P-A2 — coût par requête : plusieurs SECONDES sur CPU sans GPU (~50 inférences).
         Premier signal (14/08, 3 paires à 1 thread) : ~0,8 s/paire.
  P-A3 — si P-A2 tient, usage en traitement DIFFÉRÉ seulement — ligne de carte,
         pas défaut.

CRITÈRE D'ADOPTION, déclaré avant : +3 points de MRR minimum ET moins de 3 s par
requête. Gain sans la latence => différé (P-A3). Ni l'un ni l'autre => enterré.

⚠ À exécuter sur MACHINE CALME (aucun entraînement ni rebuild en fond) : la
latence est un critère du verdict, une machine chargée la fausse.

Usage : python research/rerank_croise.py [--echantillon 300] [--fenetre 50]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from mosaic.index import Index  # noqa: E402

# Chemins relatifs au dépôt, surchargeables par l'environnement : ce banc doit
# être rejouable ailleurs que sur la machine où il a été écrit.
CORPUS = Path(os.environ.get("MOSAIC_BENCH_CORPUS", RACINE / "bench/alloprof/corpus"))
VERITE = Path(
    os.environ.get("MOSAIC_BENCH_VERITE", RACINE / "bench/alloprof/verite.jsonl")
)
POTION = Path(
    os.environ.get("MOSAIC_POTION", RACINE / "data_externes/potion_fr_abtt2.msee")
)
# llama-server (llama.cpp) et le reranker GGUF : hors dépôt, chemins à donner.
SERVEUR = Path(os.environ.get("LLAMA_SERVER", "llama-server"))
RERANKER = Path(os.environ.get("RERANKER_GGUF", "bge-reranker-v2-m3-Q8_0.gguf"))
PORT = 8089
GRAINE = 42


def demarrer_serveur(threads: int) -> subprocess.Popen:
    # -b/-ub 8192 : en mode rerank, chaque paire requête+document doit tenir dans le
    # batch PHYSIQUE (l'ubatch de 512 par défaut rend un 500 « input too large » dès
    # ~700 tokens — mesuré 14/08). 8192 = le contexte natif de BGE-m3.
    proc = subprocess.Popen(
        [
            str(SERVEUR),
            "-m",
            str(RERANKER),
            "--reranking",
            "--port",
            str(PORT),
            "-t",
            str(threads),
            "-c",
            "8192",
            "-b",
            "8192",
            "-ub",
            "8192",
        ],
        stdout=(Path(__file__).with_name("serveur_rerank.log")).open("w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(120):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/health", timeout=2
            ) as r:
                if b"ok" in r.read():
                    return proc
        except OSError:
            time.sleep(1)
    proc.kill()
    raise RuntimeError("llama-server ne répond pas sur /health")


def reranger(query: str, documents: list[str]) -> list[int]:
    """Rend les indices des documents réordonnés par le cross-encodeur."""
    corps = {"model": "bge", "query": query, "documents": documents}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/rerank",
        data=json.dumps(corps, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return [
        res["index"]
        for res in sorted(d["results"], key=lambda x: -x["relevance_score"])
    ]


def scores(rangs_par_requete: list[tuple[list[str], set[str]]], k: int = 10) -> dict:
    rappel = mrr = 0.0
    for ids, pertinents in rangs_par_requete:
        rappel += len(set(ids[:k]) & pertinents) / len(pertinents)
        if any(d in pertinents for d in ids):
            premier = next(i for i, d in enumerate(ids) if d in pertinents)
            mrr += 1.0 / (premier + 1)
    n = max(1, len(rangs_par_requete))
    return {"rappel@10": round(rappel / n, 4), "mrr": round(mrr / n, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--echantillon", type=int, default=300)
    ap.add_argument("--fenetre", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    requetes = [
        json.loads(li)
        for li in VERITE.read_text(encoding="utf-8").splitlines()
        if li.strip()
    ]
    echantillon = random.Random(GRAINE).sample(requetes, args.echantillon)
    textes = {
        p.name: p.read_text(encoding="utf-8", errors="replace")
        for p in CORPUS.iterdir()
    }

    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        idx = Index.build(
            CORPUS,
            Path(tmp) / "idx",
            embeddings_path=POTION,
            abtt=2,
            weights=(0.25, 0.15, 0.60),
            rerank_vectors=True,
            type_doc=True,
            hybride=True,
        )
        print(
            f"index construit en {time.perf_counter() - t0:.0f}s ({len(idx.ids)} docs)"
        )

        # Bras 1 — référence : fusion 4 canaux, top fenêtre
        base: list[tuple[list[str], set[str]]] = []
        candidats: list[tuple[str, list[str]]] = []
        for q in echantillon:
            hits = idx.search(q["query"], k=args.fenetre, fusion=True)
            ids = [h["id"] for h in hits]
            base.append((ids, set(q["relevant"])))
            candidats.append((q["query"], ids))
        ref = scores(base)
        print(f"fusion seule      : {ref}")

    # Bras 2 — mêmes candidats, réordonnés par le cross-encodeur
    proc = demarrer_serveur(args.threads)
    try:
        rerange: list[tuple[list[str], set[str]]] = []
        latences = []
        for (query, ids), (_, pertinents) in zip(candidats, base):
            docs = [textes[i][:2000] for i in ids]
            t0 = time.perf_counter()
            ordre = reranger(query, docs)
            latences.append(time.perf_counter() - t0)
            rerange.append(([ids[j] for j in ordre], pertinents))
            fait = len(rerange)
            if fait % 25 == 0:
                print(
                    f"  {fait}/{len(candidats)} — latence moyenne "
                    f"{sum(latences) / fait:.2f}s/requête"
                )
        rer = scores(rerange)
    finally:
        proc.kill()

    lat = sum(latences) / len(latences)
    print(f"fusion + croisé   : {rer}")
    print(f"latence rerank    : {lat:.2f}s/requête (fenêtre {args.fenetre})")
    d_mrr = (rer["mrr"] - ref["mrr"]) * 100
    print(
        f"\nP-A1  ΔMRR = {d_mrr:+.2f} pts -> " + ("TENUE" if d_mrr >= 3 else "FAUSSE")
    )
    print(
        f"P-A2  latence {lat:.2f}s -> "
        + ("TENUE (plusieurs s)" if lat > 1 else "FAUSSE")
    )
    verdict = (
        "ADOPTÉ (interactif)"
        if d_mrr >= 3 and lat < 3
        else "DIFFÉRÉ seulement (P-A3)"
        if d_mrr >= 3
        else "ENTERRÉ"
    )
    print(f"CRITÈRE (+3 MRR ET <3s) -> {verdict}")

    Path(__file__).with_name("resultats_rerank_croise.json").write_text(
        json.dumps(
            {
                "echantillon": args.echantillon,
                "graine": GRAINE,
                "fenetre": args.fenetre,
                "threads": args.threads,
                "fusion": ref,
                "fusion_croise": rer,
                "latence_s": round(lat, 2),
                "verdict": verdict,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
