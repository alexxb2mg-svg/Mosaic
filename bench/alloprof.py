"""Corpus de banc « grand format » : Alloprof (2 556 documents, 2 316 requêtes réelles).

Télécharge le jeu lyon-nlp/alloprof (aide aux devoirs québécoise, français natif,
vérité terrain annotée — le même jeu que MTEB utilise pour évaluer le retrieval
français) et le convertit au format du banc Mosaic :

    bench/alloprof/corpus/<uuid>.md   — un document par fichier (titre + texte)
    bench/alloprof/verite.jsonl       — {"query", "relevant": ["<uuid>.md", ...]}

Usage :
    python bench/alloprof.py                # télécharge dans bench/alloprof/
    python bench/run_bench.py bench/alloprof/corpus bench/alloprof/verite.jsonl \
        --config alloprof
    python bench/baseline_model.py bench/alloprof/corpus bench/alloprof/verite.jsonl

Les données ne sont PAS redistribuées avec ce dépôt (licence CC BY-NC-SA 4.0,
distincte de la licence Apache-2.0 du code) : ce script les récupère à la demande
via l'API publique datasets-server de Hugging Face — aucune dépendance à installer,
urllib seulement. ~16 Mo, une cinquantaine de requêtes HTTP paginées.

Outillage de banc, hors produit.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "lyon-nlp/alloprof"
API = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # maximum accepté par l'API
DEFAULT_OUT = Path(__file__).resolve().parent / "alloprof"


def _fetch_page(
    config: str, split: str, offset: int, tentatives: int = 4
) -> list[dict]:
    """Une page de lignes via l'API datasets-server, avec retries à pas croissant.

    L'API rend {"rows": [{"row": {...}}, ...]} ; on ne garde que les payloads."""
    params = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": config,
            "split": split,
            "offset": offset,
            "length": PAGE,
        }
    )
    derniere: Exception | None = None
    for essai in range(tentatives):
        try:
            with urllib.request.urlopen(f"{API}?{params}", timeout=60) as resp:
                payload = json.load(resp)
            return [r["row"] for r in payload["rows"]]
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
        ) as exc:
            derniere = exc
            time.sleep(2**essai)
    raise RuntimeError(
        f"API datasets-server injoignable ({config} offset={offset}) : {derniere}"
    )


def _fetch_all(config: str, split: str) -> list[dict]:
    """Toutes les lignes d'un split, paginées ; s'arrête sur la première page courte."""
    rows: list[dict] = []
    offset = 0
    while True:
        page = _fetch_page(config, split, offset)
        rows.extend(page)
        print(f"  {config}/{split} : {len(rows)} lignes", file=sys.stderr)
        if len(page) < PAGE:
            return rows
        offset += PAGE


def document_markdown(row: dict) -> str:
    """Un document Alloprof rendu en markdown : titre en en-tête, texte au corps."""
    titre = (row.get("title") or "").strip()
    texte = (row.get("text") or "").strip()
    return f"# {titre}\n\n{texte}\n" if titre else f"{texte}\n"


def requetes_bench(
    queries: list[dict], uuids_connus: set[str]
) -> tuple[list[dict], int]:
    """Convertit les requêtes Alloprof au schéma verite.jsonl du banc.

    Une requête dont AUCUN document pertinent n'existe dans le corpus est écartée
    (compteur retourné) : la mesurer serait un échec garanti pour tous les moteurs,
    du bruit plutôt qu'un signal. Les pertinents partiellement absents sont filtrés
    ligne à ligne pour la même raison."""
    lignes, ecartees = [], 0
    for q in queries:
        texte = (q.get("text") or "").strip()
        relevant = [f"{u}.md" for u in q.get("relevant", ()) if u in uuids_connus]
        if not texte or not relevant:
            ecartees += 1
            continue
        lignes.append({"query": texte, "relevant": relevant})
    return lignes, ecartees


def main() -> None:
    parser = argparse.ArgumentParser(prog="alloprof")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"répertoire de sortie (défaut {DEFAULT_OUT})",
    )
    args = parser.parse_args()
    out = Path(args.out)
    corpus_dir = out / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    documents = _fetch_all("documents", "test")
    uuids: set[str] = set()
    for row in documents:
        uuid = row["uuid"]
        uuids.add(uuid)
        (corpus_dir / f"{uuid}.md").write_text(document_markdown(row), encoding="utf-8")

    queries = _fetch_all("queries", "test")
    lignes, ecartees = requetes_bench(queries, uuids)
    with (out / "verite.jsonl").open("w", encoding="utf-8") as fh:
        for ligne in lignes:
            fh.write(json.dumps(ligne, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "documents": len(uuids),
                "requetes": len(lignes),
                "requetes_ecartees": ecartees,
                "corpus": str(corpus_dir),
                "verite": str(out / "verite.jsonl"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
