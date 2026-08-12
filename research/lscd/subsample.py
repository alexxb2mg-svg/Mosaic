"""Sous-échantillon équilibré : N docs par période, stride déterministe (couvre
toute la période), recalcul des df cibles sur le sous-échantillon."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000


def pick(src: Path, dst: Path, n: int, targets: list[str]) -> dict:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    docs = sorted(src.glob("*.txt"))
    idx = np.unique(np.linspace(0, len(docs) - 1, n).round().astype(int))
    df = {t: 0 for t in targets}
    tset = set(targets)
    tok = 0
    for i in idx:
        p = docs[i]
        text = p.read_text(encoding="utf-8")
        shutil.copy2(p, dst / p.name)
        toks = text.split()
        tok += len(toks)
        for t in tset.intersection(toks):
            df[t] += 1
    return {"docs": int(len(idx)), "tokens": tok, "df": df}


def main():
    targets = (BASE / "targets_fixed.txt").read_text().split()
    stats = {
        "t1": pick(BASE / "t1", BASE / "t1s", N, targets),
        "t2": pick(BASE / "t2", BASE / "t2s", N, targets),
    }
    (BASE / "dfs_sub.json").write_text(json.dumps(stats, indent=1))
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
