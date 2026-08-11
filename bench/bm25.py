"""Baseline BM25 — implémentation de référence, pur Python."""

import math
from collections import Counter

K1 = 1.5
B = 0.75


class BM25:
    def __init__(self, docs: list[list[str]]) -> None:
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / len(docs)) if docs else 0.0
        self.df: Counter[str] = Counter()
        for c in self.docs:
            self.df.update(c.keys())
        self.n = len(docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], k: int = 10) -> list[int]:
        scores = [0.0] * self.n
        for term in query_tokens:
            if term not in self.df:
                continue
            idf = self._idf(term)
            for i, counts in enumerate(self.docs):
                tf = counts.get(term, 0)
                if tf:
                    denom = tf + K1 * (1 - B + B * self.lens[i] / self.avg)
                    scores[i] += idf * tf * (K1 + 1) / denom
        ranked = sorted(range(self.n), key=lambda i: -scores[i])
        return [i for i in ranked if scores[i] > 0.0][:k]
