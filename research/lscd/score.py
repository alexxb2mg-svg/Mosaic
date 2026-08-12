"""Scoring banc LSCD : Spearman entre les deltas mosaic diff et truth/graded.txt.

Spearman en numpy pur : rangs avec moyenne des ex-aequo, puis Pearson des rangs.
Contrôles : (a) canal derive_usage seul (baseline fréquence interne) ;
            (b) corrélation delta ~ |Δdf| (fuite de fréquence, piège n°1 du champ).
"""

import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent


def ranks(x: np.ndarray) -> np.ndarray:
    """Rangs 1..n avec moyenne des ex-aequo."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # moyenne des ex-aequo
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return r


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(a @ b) / den


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    # vérifié à la main sur cas ex-aequo : [1,2,2,3,5] vs [1,3,2,4,4] -> 9/9,5 = 0,9474
    return pearson(ranks(a), ranks(b))


def main(diff_path: str, graded_path: str, dfs_path: str):
    diff = json.loads(Path(diff_path).read_text(encoding="utf-8"))
    graded = json.loads(Path(graded_path).read_text(encoding="utf-8"))
    dfs = json.loads(Path(dfs_path).read_text(encoding="utf-8"))

    targets = sorted(graded)
    truth = np.array([graded[t] for t in targets])

    # 1) notre delta : derive_mots (cosinus des profils de cooccurrence)
    dmots = {e["mot"]: e["delta"] for e in diff["derive_mots"]}
    delta = np.array([dmots.get(t, 0.0) for t in targets])
    absents = [t for t in targets if t not in dmots]

    # 2) contrôle (a) : canal derive_usage seul — |ratio de variation de df|
    usage = {}
    for e in diff["derive_usage"]["declins"]:
        usage[e["mot"]] = abs(e["ratio"])
    for e in diff["derive_usage"]["croissances"]:
        usage[e["mot"]] = abs(e["ratio"])
    u = np.array([usage.get(t, 0.0) for t in targets])

    # 3) contrôle (b) : |Δdf| depuis nos comptages directs (dfs.json)
    df1 = dfs["t1"]["df"]
    df2 = dfs["t2"]["df"]
    # normalisation par taille de corpus (t2 a 1,39x plus de docs que t1)
    n1, n2 = dfs["t1"]["docs"], dfs["t2"]["docs"]
    ddf_rel = np.array(
        [
            abs(df2[t] / n2 - df1[t] / n1) / max(df2[t] / n2, df1[t] / n1)
            for t in targets
        ]
    )
    ddf_raw = np.array([abs(df2[t] - df1[t]) for t in targets])

    print(f"cibles évaluées : {len(targets)}")
    print(f"cibles absentes de derive_mots (delta=0) : {absents or 'aucune'}")
    print()
    print(f"SPEARMAN delta (derive_mots) vs graded : {spearman(delta, truth):+.4f}")
    print(f"controle (a) derive_usage seul vs graded : {spearman(u, truth):+.4f}")
    print(
        f"controle (b) delta vs |Δdf| relatif      : {spearman(delta, ddf_rel):+.4f}"
        f" (pearson {pearson(delta, ddf_rel):+.4f})"
    )
    print(f"controle (b bis) delta vs |Δdf| brut     : {spearman(delta, ddf_raw):+.4f}")
    print()
    for t in targets:
        print(
            f"  {t:22s} delta={dmots.get(t, 0.0):.4f} truth={graded[t]:.4f} "
            f"df {df1[t]}->{df2[t]}"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
