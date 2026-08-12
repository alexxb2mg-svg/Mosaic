"""Préparation banc LSCD SemEval-2020 T1 LATIN pour mosaic diff.

- réécrit les marqueurs d'homonymie lemme#N -> lemme-N (le tokenizer casse sur #,
  préserve le tiret — même geste que _nn -> -nn en anglais)
- t1 en entier (4 804 docs de 20 phrases), t2 sous-échantillonné par stride au
  même nombre de docs (masses de tokens quasi égales : 1,75M vs ~1,95M)
- df cibles calculés sur les corpus réellement utilisés -> dfs_lat.json
"""

import gzip
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
RAW = BASE / "lat_raw" / "semeval2020_ulscd_lat"
CHUNK = 20

MARK = re.compile(r"#(\d+)\b")


def fix(s: str) -> str:
    return MARK.sub(r"-\1", s)


def read_docs(gz_path: Path) -> list[str]:
    docs, buf = [], []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = fix(line.strip())
            if not line:
                continue
            buf.append(line)
            if len(buf) == CHUNK:
                docs.append("\n".join(buf))
                buf.clear()
    if buf:
        docs.append("\n".join(buf))
    return docs


def write_docs(docs: list[str], out_dir: Path, targets: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = {t: 0 for t in targets}
    tset = set(targets)
    tok = 0
    for i, doc in enumerate(docs):
        (out_dir / f"doc-{i:05d}.txt").write_text(doc, encoding="utf-8")
        toks = doc.split()
        tok += len(toks)
        for t in tset.intersection(toks):
            df[t] += 1
    return {"docs": len(docs), "tokens": tok, "df": df}


def main():
    targets = (RAW / "targets.txt").read_text().split()
    graded = {}
    for line in (RAW / "truth" / "graded.txt").read_text().splitlines():
        if line.strip():
            w, v = line.split("\t")
            graded[w] = float(v)
    (BASE / "graded_lat.json").write_text(json.dumps(graded, indent=1))

    d1 = read_docs(RAW / "corpus1" / "lemma" / "LatinISE1.txt.gz")
    d2 = read_docs(RAW / "corpus2" / "lemma" / "LatinISE2.txt.gz")
    idx = np.unique(np.linspace(0, len(d2) - 1, len(d1)).round().astype(int))
    d2s = [d2[i] for i in idx]

    stats = {
        "t1": write_docs(d1, BASE / "lat_t1", targets),
        "t2": write_docs(d2s, BASE / "lat_t2s", targets),
    }
    (BASE / "dfs_lat.json").write_text(json.dumps(stats, indent=1))
    for k in ("t1", "t2"):
        print(k, stats[k]["docs"], "docs,", stats[k]["tokens"], "tokens")
    weak = {
        t: (stats["t1"]["df"][t], stats["t2"]["df"][t])
        for t in targets
        if stats["t1"]["df"][t] < 3 or stats["t2"]["df"][t] < 3
    }
    print("cibles sous DF_MIN=3 dans une période :", weak or "aucune")


if __name__ == "__main__":
    main()
