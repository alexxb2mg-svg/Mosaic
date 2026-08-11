"""Détection de collocations par PMI : les expressions métier deviennent des tokens."""

import math
from collections import Counter

from mosaic.tokenize import STOPWORDS

MIN_COUNT = 5
MIN_PMI = 3.0


def detect(
    docs_tokens: list[list[str]], min_count: int = MIN_COUNT, min_pmi: float = MIN_PMI
) -> set[tuple[str, str]]:
    uni: Counter[str] = Counter()
    bi: Counter[tuple[str, str]] = Counter()
    for tokens in docs_tokens:
        uni.update(tokens)
        bi.update(zip(tokens, tokens[1:], strict=False))
    total = max(sum(uni.values()), 1)
    result: set[tuple[str, str]] = set()
    for (a, b), n_ab in bi.items():
        if n_ab < min_count or a in STOPWORDS or b in STOPWORDS:
            continue
        pmi = math.log(n_ab * total / (uni[a] * uni[b]))
        if pmi >= min_pmi:
            result.add((a, b))
    return result


def merge(tokens: list[str], colloc: set[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) in colloc:
            out.append(tokens[i] + "_" + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out
