"""Bancs ANGLAIS standards de la suite BEIR — préparation d'un corpus + sa vérité terrain.

Pourquoi ces bancs existent ici : tous les bancs maison sont en français (Alloprof,
le banc cuisine, les corpus métier privés). Un lecteur anglophone ne peut donc
**vérifier aucun de nos chiffres** — il doit nous croire sur parole, ce qui est
exactement ce que ce projet reproche aux autres. BEIR est la suite d'évaluation
standard de la recherche d'information : s'y mesurer rend le travail reproductible
par n'importe qui, sur un terrain qu'il connaît déjà.

Mais l'intérêt n'est pas que la langue. Nos trois bancs (Alloprof, SciFact, corpus
métier) posent tous la MÊME forme de problème : requête courte, un ou quelques
documents pertinents, pertinence = « ce document parle du sujet ». Ils font varier
la langue et le bruit, jamais la STRUCTURE de la question. Les jeux ci-dessous
changent la structure, et c'est là qu'on peut découvrir un angle mort.

Jeux couverts, et ce que CHACUN met à l'épreuve :

  scifact   vérification de faits. Requêtes = affirmations propres, documents =
            résumés d'articles. Le terrain « propre » de référence.
  arguana   recherche de CONTRE-argument : le bon document RÉFUTE la requête. La
            similarité de surface y est activement trompeuse — le document cible
            parle du même sujet en disant l'inverse. Seul banc capable de mettre à
            l'épreuve le canal grammatical (négateurs, voix passive). Particularité :
            requêtes très longues (un argument entier), et le corpus contient des
            arguments du même pool que les requêtes.
  nfcorpus  décalage de REGISTRE extrême : questions de santé grand public contre
            articles scientifiques. Terrain naturel de HyDE.
  fiqa      questions financières d'opinion : la réponse ne partage presque aucun
            mot avec la question. Sémantique pur.

Format produit, identique aux autres bancs du dépôt :
    bench/<jeu>/corpus/<id>.md   — un document par fichier (titre + texte)
    bench/<jeu>/verite.jsonl     — {"query", "relevant": ["<id>.md", ...]}

Usage :
    python bench/beir.py --jeu arguana
    python bench/run_bench.py bench/arguana/corpus bench/arguana/verite.jsonl \\
        --config arguana --no-path-tokens

Les données ne sont PAS redistribuées ici (licences propres à chaque jeu, distinctes
de l'Apache-2.0 du code) : ce script les récupère à la demande via l'API publique
datasets-server de Hugging Face — urllib seulement, aucune dépendance.

Outillage de banc, hors produit.
"""

import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

# Source PRINCIPALE : l'archive officielle du papier BEIR (UKP / TU Darmstadt), en
# JSONL — une seule requête, aucune limite de débit, aucune dépendance (zipfile et
# json sont dans la bibliothèque standard).
# L'API datasets-server de Hugging Face reste en REPLI : elle sert les mêmes données
# mais page par page, et elle rend des 429 dès qu'on enchaîne les pages (vécu le
# 13/08). Le miroir Hugging Face du dataset lui-même est en Parquet, illisible sans
# pyarrow — donc hors de question ici.
ZIP_OFFICIEL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
API = "https://datasets-server.huggingface.co/rows"
TAILLE_PAGE = 100

# jeu -> (dataset HF pour le repli, split des jugements)
JEUX: dict[str, tuple[str, str]] = {
    "scifact": ("BeIR/scifact", "test"),
    "arguana": ("BeIR/arguana", "test"),
    "nfcorpus": ("BeIR/nfcorpus", "test"),
    "fiqa": ("BeIR/fiqa", "test"),
}


def _depuis_zip(jeu: str, split: str) -> tuple[list[dict], list[dict], list[dict]]:
    """(corpus, requêtes, jugements) lus dans l'archive officielle, EN MÉMOIRE.

    Rien n'est extrait sur le disque : on lit les trois membres par leur nom exact.
    C'est aussi la parade au zip-slip — un nom de membre piégé (`../..`) n'a aucun
    effet puisqu'aucun chemin d'archive ne sert jamais de chemin d'écriture."""
    url = f"{ZIP_OFFICIEL}/{jeu}.zip"
    print(f"[{jeu}] téléchargement de l'archive officielle…", flush=True)
    with urllib.request.urlopen(url, timeout=300) as r:
        brut = r.read()
    print(f"[{jeu}] {len(brut) / 1e6:.1f} Mo, lecture…", flush=True)

    def jsonl(nom: str) -> list[dict]:
        return [
            json.loads(li)
            for li in zf.read(nom).decode("utf-8").splitlines()
            if li.strip()
        ]

    with zipfile.ZipFile(io.BytesIO(brut)) as zf:
        corpus = jsonl(f"{jeu}/corpus.jsonl")
        requetes = jsonl(f"{jeu}/queries.jsonl")
        # les qrels sont en TSV : query-id \t corpus-id \t score, avec une ligne d'en-tête
        lignes = zf.read(f"{jeu}/qrels/{split}.tsv").decode("utf-8").splitlines()
        qrels = []
        for li in lignes[1:]:
            champs = li.split("\t")
            if len(champs) >= 3:
                qrels.append(
                    {
                        "query-id": champs[0],
                        "corpus-id": champs[1],
                        "score": champs[2],
                    }
                )
    return corpus, requetes, qrels


def _page(dataset: str, config: str, split: str, offset: int) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": TAILLE_PAGE,
        }
    )
    for essai in range(5):
        try:
            with urllib.request.urlopen(f"{API}?{params}", timeout=60) as r:
                return json.loads(r.read())["rows"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if essai == 4:
                raise RuntimeError(f"API HuggingFace injoignable : {exc}") from None
            time.sleep(2 * (essai + 1))  # l'API limite le débit : on patiente
    return []


def _tout(dataset: str, config: str, split: str, quoi: str) -> list[dict]:
    lignes: list[dict] = []
    while True:
        page = _page(dataset, config, split, len(lignes))
        if not page:
            break
        lignes += [p["row"] for p in page]
        print(f"  {quoi} : {len(lignes)}", end="\r", flush=True)
        if len(page) < TAILLE_PAGE:
            break
    print(f"  {quoi} : {len(lignes)}   ", flush=True)
    return lignes


_NOM_SUR = re.compile(r"[^A-Za-z0-9_.-]")


def _fichier(doc_id: str) -> str:
    """Nom de fichier sûr ET stable : les identifiants BEIR sont numériques sur
    certains jeux, alphanumériques sur d'autres (arguana) — on neutralise tout."""
    return f"{_NOM_SUR.sub('_', str(doc_id))}.md"


def preparer(jeu: str, racine: Path) -> tuple[int, int]:
    dataset, split_qrels = JEUX[jeu]
    corpus_dir = racine / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    try:
        corpus, requetes, qrels = _depuis_zip(jeu, split_qrels)
    except (urllib.error.URLError, TimeoutError, KeyError, zipfile.BadZipFile) as exc:
        # Repli explicite et BRUYANT : si l'archive officielle devient indisponible,
        # on le dit avant de repartir sur l'API paginée — pas de bascule silencieuse
        # vers un chemin plus fragile.
        print(f"[{jeu}] archive officielle indisponible ({exc}) — repli sur l'API HF")
        corpus = _tout(dataset, "corpus", "corpus", "documents")
        requetes = _tout(dataset, "queries", "queries", "requêtes")
        qrels = _tout(f"{dataset}-qrels", "default", split_qrels, "jugements")

    ecrits = 0
    for d in corpus:
        titre = (d.get("title") or "").strip()
        texte = (d.get("text") or "").strip()
        if not texte:
            continue
        (corpus_dir / _fichier(d["_id"])).write_text(
            f"# {titre}\n\n{texte}\n" if titre else f"{texte}\n", encoding="utf-8"
        )
        ecrits += 1

    par_id = {str(q["_id"]): (q.get("text") or "").strip() for q in requetes}
    pertinents: dict[str, list[str]] = {}
    for j in qrels:
        # BEIR : score 0 = non pertinent, on ne garde que les jugements positifs
        if int(j.get("score", 0)) <= 0:
            continue
        qid = str(j["query-id"])
        pertinents.setdefault(qid, []).append(_fichier(j["corpus-id"]))

    presents = {p.name for p in corpus_dir.glob("*.md")}
    lignes = []
    for qid, docs in sorted(pertinents.items()):
        texte = par_id.get(qid, "").strip()
        gardes = [d for d in docs if d in presents]
        if texte and gardes:  # une requête sans document présent ne prouve rien
            lignes.append({"query": texte, "relevant": sorted(set(gardes))})

    (racine / "verite.jsonl").write_text(
        "\n".join(json.dumps(li, ensure_ascii=False) for li in lignes) + "\n",
        encoding="utf-8",
    )
    return ecrits, len(lignes)


def main() -> int:
    ap = argparse.ArgumentParser(description="Prépare un banc BEIR (corpus + vérité)")
    ap.add_argument("--jeu", choices=sorted(JEUX), required=True)
    ap.add_argument("--sortie", default=None, help="dossier (défaut : bench/<jeu>)")
    args = ap.parse_args()
    racine = Path(args.sortie) if args.sortie else Path("bench") / args.jeu

    docs, requetes = preparer(args.jeu, racine)
    print(
        f"\n{docs} documents écrits dans {racine / 'corpus'}\n"
        f"{requetes} requêtes dans {racine / 'verite.jsonl'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
