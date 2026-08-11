"""Canal lexical BM25 — l'un des trois canaux de la fusion hybride.

Validé par la mesure (banc Alloprof, 2 556 docs / 2 316 requêtes réelles) : la fusion RRF
grille + BM25 + embeddings atteint 0.517 R@10, au-dessus du standard du marché
BM25 + embeddings (0.498) et de chaque canal seul (BM25 0.482, grille 0.385). Le duo
grille + BM25 SANS embeddings a été mesuré NUISIBLE (0.460 < BM25 seul) — c'est pourquoi
la fusion exige les trois canaux (cf. queries.search_fusion).

Le canal opère sur le MÊME flux de tokens que la grille (canonicalisation + collocations,
tokens de chemin selon index_paths) : les deux canaux voient le même monde, seule la
géométrie diffère (comptages exacts ici, superposition distribuée là).

Persistance : bm25.msbm (postings par terme + longueurs de documents). Un sac de mots est
plus révélateur que la grille (reconstruction partielle du vocabulaire d'un document) : le
canal est OPT-IN au build (--hybride), jamais stocké sans demande explicite — le principe
du prisme (aucun texte persisté) reste tenu, la dérogation sur les COMPTAGES est consentie.
"""

import math

import numpy as np

# Paramètres Okapi standard (Robertson) — mêmes valeurs que le banc Alloprof.
K1 = 1.5
B = 0.75


class Bm25:
    """Postings par terme (layout colonne : un terme -> ses documents et tf), le layout
    du chemin chaud : scorer une requête de m termes coûte O(somme des df), jamais O(N·V).

    Invariants : `indptr` (V+1) délimite les tranches de `doc_idx`/`tf` (nnz), triées par
    (terme, document) ; `df(t) == indptr[t+1] - indptr[t]` (jamais stocké séparément) ;
    `vocab_termes[j]` est le terme de la colonne j, `doc_lens[i]` la longueur du document i
    (même ordre que `Index.ids`)."""

    def __init__(
        self,
        vocab_termes: list[str],
        indptr: np.ndarray,
        doc_idx: np.ndarray,
        tf: np.ndarray,
        doc_lens: np.ndarray,
    ) -> None:
        self.vocab_termes = vocab_termes
        self.vocab: dict[str, int] = {t: j for j, t in enumerate(vocab_termes)}
        self.indptr = indptr
        self.doc_idx = doc_idx
        self.tf = tf
        self.doc_lens = doc_lens

    @property
    def n_docs(self) -> int:
        return int(self.doc_lens.shape[0])

    @classmethod
    def from_docs(cls, docs: list[tuple[str, list[str]]]) -> "Bm25":
        """Construit les postings depuis les documents tokenisés du build (ordre = ids).
        Déterministe : vocabulaire en ordre de première apparition (l'ordre des documents
        est lui-même déterministe — fichiers triés au build)."""
        vocab: dict[str, int] = {}
        triplets: list[tuple[int, int, int]] = []  # (terme, document, tf)
        doc_lens = np.zeros(len(docs), dtype=np.int32)
        for i, (_doc_id, tokens) in enumerate(docs):
            doc_lens[i] = len(tokens)
            counts: dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                j = vocab.setdefault(t, len(vocab))
                triplets.append((j, i, c))
        triplets.sort()
        v = len(vocab)
        indptr = np.zeros(v + 1, dtype=np.int64)
        doc_idx = np.zeros(len(triplets), dtype=np.int32)
        tf = np.zeros(len(triplets), dtype=np.int32)
        for pos, (j, i, c) in enumerate(triplets):
            indptr[j + 1] += 1
            doc_idx[pos] = i
            tf[pos] = c
        np.cumsum(indptr, out=indptr)
        termes = sorted(vocab, key=vocab.__getitem__)
        return cls(termes, indptr, doc_idx, tf, doc_lens)

    def scores(self, tokens: list[str]) -> np.ndarray:
        """Scores BM25 de CHAQUE document contre la requête tokenisée (float32, (n,)).
        Les occurrences multiples d'un terme dans la requête comptent (convention Okapi
        classique, même que le banc). Termes hors vocabulaire : ignorés."""
        n = self.n_docs
        out = np.zeros(n, dtype=np.float32)
        if n == 0 or not tokens:
            return out
        avgdl = float(self.doc_lens.mean())
        if avgdl == 0.0:
            return out
        denom_len = np.float32(K1) * (
            np.float32(1.0 - B)
            + np.float32(B) * self.doc_lens.astype(np.float32) / np.float32(avgdl)
        )
        for t in tokens:
            j = self.vocab.get(t)
            if j is None:
                continue
            start, end = int(self.indptr[j]), int(self.indptr[j + 1])
            df = end - start
            rows = self.doc_idx[start:end]
            tf = self.tf[start:end].astype(np.float32)
            idf = np.float32(math.log(1.0 + (n - df + 0.5) / (df + 0.5)))
            out[rows] += idf * tf * np.float32(K1 + 1.0) / (tf + denom_len[rows])
        return out

    def add_doc(self, tokens: list[str]) -> None:
        """Ajoute UN document (ligne suivante, même ordre que Index.ids). Reconstruit les
        tranches de postings touchées — O(nnz), du même ordre que le vstack de add() sur
        la matrice documents : add() reste l'opération occasionnelle, search le chemin chaud."""
        i = self.n_docs
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        self.doc_lens = np.append(self.doc_lens, np.int32(len(tokens)))
        nouveaux = [t for t in counts if t not in self.vocab]
        anciens = {self.vocab[t]: counts[t] for t in counts if t in self.vocab}
        if anciens:
            # insertion en fin de tranche de chaque terme existant (i est le plus grand
            # index de document : l'ordre (terme, document) des postings est préservé)
            v = len(self.vocab_termes)
            grossit = np.zeros(v + 1, dtype=np.int64)
            for j in anciens:
                grossit[j + 1] = 1
            np.cumsum(grossit, out=grossit)
            nnz = int(self.indptr[-1]) + len(anciens)
            doc_idx = np.zeros(nnz, dtype=np.int32)
            tf = np.zeros(nnz, dtype=np.int32)
            for j in range(v):
                s, e = int(self.indptr[j]), int(self.indptr[j + 1])
                ns = s + int(grossit[j])
                doc_idx[ns : ns + (e - s)] = self.doc_idx[s:e]
                tf[ns : ns + (e - s)] = self.tf[s:e]
                if j in anciens:
                    doc_idx[ns + (e - s)] = i
                    tf[ns + (e - s)] = anciens[j]
            self.indptr = self.indptr + grossit
            self.doc_idx = doc_idx
            self.tf = tf
        for t in nouveaux:
            j = len(self.vocab_termes)
            self.vocab_termes.append(t)
            self.vocab[t] = j
            self.indptr = np.append(self.indptr, self.indptr[-1] + 1)
            self.doc_idx = np.append(self.doc_idx, np.int32(i))
            self.tf = np.append(self.tf, np.int32(counts[t]))
