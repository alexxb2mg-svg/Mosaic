"""Préparation banc LSCD SemEval-2020 T1 EN pour mosaic diff.

- lit les corpus LEMME gzip des deux périodes
- réécrit _nn -> -nn, _vb -> -vb PARTOUT
- découpe en documents de 20 phrases (variante B) -> t1/ et t2/
- réécrit targets.txt et truth/graded.txt avec les mêmes suffixes
- calcule les df documentaires des cibles dans chaque période -> dfs.json
"""

import gzip
import json
import re
from pathlib import Path

BASE = Path(__file__).parent
RAW = BASE / "eng_raw" / "semeval2020_ulscd_eng"
CHUNK = 20

SUFF = re.compile(r"_(nn|vb)\b")


def fix(s: str) -> str:
    return SUFF.sub(r"-\1", s)


def build_period(gz_path: Path, out_dir: Path, targets: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = {t: 0 for t in targets}
    tset = set(targets)
    n_docs = 0
    n_sents = 0
    buf: list[str] = []

    def flush():
        nonlocal n_docs
        if not buf:
            return
        doc = "\n".join(buf)
        (out_dir / f"doc-{n_docs:05d}.txt").write_text(doc, encoding="utf-8")
        present = tset.intersection(doc.split())
        for t in present:
            df[t] += 1
        n_docs += 1
        buf.clear()

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = fix(line.strip())
            if not line:
                continue
            buf.append(line)
            n_sents += 1
            if len(buf) == CHUNK:
                flush()
    flush()
    return {"docs": n_docs, "sents": n_sents, "df": df}


def main():
    targets = [fix(t) for t in (RAW / "targets.txt").read_text().split()]
    (BASE / "targets_fixed.txt").write_text("\n".join(targets), encoding="utf-8")

    graded = {}
    for line in (RAW / "truth" / "graded.txt").read_text().splitlines():
        if not line.strip():
            continue
        w, v = line.split("\t")
        graded[fix(w)] = float(v)
    (BASE / "graded_fixed.json").write_text(json.dumps(graded, indent=1))

    stats = {}
    stats["t1"] = build_period(
        RAW / "corpus1" / "lemma" / "ccoha1.txt.gz", BASE / "t1", targets
    )
    stats["t2"] = build_period(
        RAW / "corpus2" / "lemma" / "ccoha2.txt.gz", BASE / "t2", targets
    )
    (BASE / "dfs.json").write_text(json.dumps(stats, indent=1))
    print(
        "t1:",
        stats["t1"]["docs"],
        "docs /",
        stats["t1"]["sents"],
        "phrases |",
        "t2:",
        stats["t2"]["docs"],
        "docs /",
        stats["t2"]["sents"],
        "phrases",
    )
    zero = [
        t for t in targets if stats["t1"]["df"][t] == 0 or stats["t2"]["df"][t] == 0
    ]
    print("cibles à df=0 dans une période :", zero or "aucune")


if __name__ == "__main__":
    main()
