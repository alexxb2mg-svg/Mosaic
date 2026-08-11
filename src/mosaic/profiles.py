"""Profils de cooccurrence : le sens appris du corpus, comptages épars puis finalisation pondérée."""

import math
from collections.abc import Callable

import numpy as np

from mosaic.encoder import WEIGHTS_DEFAULT
from mosaic.signatures import signature
from mosaic.tokenize import STOPWORDS

WINDOW = 5

# Retour du chargeur différé (v1.5 chargement paresseux) : (pair_counts, marginals, total_mass) —
# même triplet que ce que load_vocab(lazy=True) matérialiserait immédiatement en mode eager.
SparseLoader = Callable[
    [], tuple[dict[tuple[int, int], float], dict[int, float], float]
]


class Profiles:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.rows: dict[str, int] = {}
        self.acc = np.zeros((0, dim), dtype=np.float32)
        # Comptages épars accumulés par learn() — vérité brute, jamais écrasée.
        # Clé symétrique (row_min, row_max) : chaque paire non ordonnée stockée une fois.
        self.pair_counts: dict[tuple[int, int], float] = {}
        self.marginals: dict[int, float] = {}
        self.total_mass: float = 0.0
        self.counts: dict[str, int] = {}
        self.df: dict[str, int] = {}
        self.n_docs = 0
        self._sig_cache: dict[str, np.ndarray] = {}
        # v1.5 chargement paresseux (store.load_vocab(..., lazy=True)) : quand acc est un
        # np.memmap read-only sur vocab.msev, le bloc épars (pair_counts/marginals/total_mass)
        # n'a pas été analysé au chargement — ce callable, posé par store.py, fait le travail
        # complet à la demande. None = rien en attente (cas normal : learn()/finalize() direct).
        self._pending_sparse: SparseLoader | None = None

    def _materialize_sparse(self) -> None:
        """Déclenche le chargeur différé une fois, avant tout usage de pair_counts/marginals/
        total_mass. No-op si rien n'est en attente (immense majorité des appels)."""
        if self._pending_sparse is not None:
            loader = self._pending_sparse
            self._pending_sparse = None
            self.pair_counts, self.marginals, self.total_mass = loader()

    def _sig(self, token: str) -> np.ndarray:
        s = self._sig_cache.get(token)
        if s is None:
            s = signature(token, self.dim)
            self._sig_cache[token] = s
        return s

    def _row(self, token: str) -> int:
        idx = self.rows.get(token)
        if idx is None:
            idx = len(self.rows)
            self.rows[token] = idx
        return idx

    def learn(self, tokens: list[str]) -> None:
        """Apprend un document : accumule des comptages de paires épars (bruts).

        Commutatif et additif : l'ordre des documents (et des appels learn()
        successifs) est indifférent. Ne matérialise aucune matrice — voir finalize().
        """
        self._materialize_sparse()
        self.n_docs += 1
        for t in set(tokens):
            self.df[t] = self.df.get(t, 0) + 1
        content = [(i, t) for i, t in enumerate(tokens) if t not in STOPWORDS]
        for t in (t for _, t in content):
            self.counts[t] = self.counts.get(t, 0) + 1
        for a in range(len(content)):
            i, t_a = content[a]
            for b in range(a + 1, len(content)):
                j, t_b = content[b]
                d = j - i
                if d > WINDOW:
                    break
                w = float(6 - d)
                idx_a = self._row(t_a)
                idx_b = self._row(t_b)
                key = (idx_a, idx_b) if idx_a <= idx_b else (idx_b, idx_a)
                self.pair_counts[key] = self.pair_counts.get(key, 0.0) + w
                self.marginals[idx_a] = self.marginals.get(idx_a, 0.0) + w
                self.marginals[idx_b] = self.marginals.get(idx_b, 0.0) + w
                self.total_mass += w

    def finalize(self, weighting: str = "brut") -> None:
        """Matérialise self.acc (float32, V×dim) à partir des comptages épars.

        - "brut" : poids = comptage brut de la paire (comportement v1 exact).
        - "ppmi" : poids = max(0, ln(total_mass·comptage / (marginal_i·marginal_j))),
          PPMI symétrique.

        Itère les paires triées par clé (row_min, row_max) : le résultat ne dépend
        que des comptages accumulés, jamais de l'ordre d'insertion dans le dict.
        Comme self.acc est alloué d'un coup à sa taille finale (V×dim), les indices
        de lignes sont résolus (via rows) avant toute écriture — pas de tableau qui
        grandit pendant l'accumulation, donc pas d'aliasing possible.
        """
        if weighting not in ("brut", "ppmi"):
            raise ValueError(
                f"weighting inconnu : {weighting!r} (attendu 'brut' ou 'ppmi')"
            )
        self._materialize_sparse()
        n = len(self.rows)
        # Réallocation complète et inconditionnelle : c'est ce qui libère un acc memmap
        # (v1.5 chargement paresseux) hérité de store.load_vocab(lazy=True) — plus aucune
        # référence Python vers l'ancien memmap après cette ligne, donc plus de handle
        # ouvert sur vocab.msev quand _save() voudra le ré-écrire (piège Windows : un
        # memmap actif verrouille le fichier contre l'écrasement).
        self.acc = np.zeros((n, self.dim), dtype=np.float32)
        if n == 0 or not self.pair_counts:
            return
        tokens_by_row: list[str] = [""] * n
        for token, idx in self.rows.items():
            tokens_by_row[idx] = token
        for key in sorted(self.pair_counts):
            i, j = key
            w = self.pair_counts[key]
            if weighting == "brut":
                poids = w
            else:
                denom = self.marginals[i] * self.marginals[j]
                poids = 0.0
                if denom > 0.0:
                    ratio = self.total_mass * w / denom
                    if ratio > 0.0:
                        poids = max(0.0, math.log(ratio))
            if poids == 0.0:
                continue
            sig_i = self._sig(tokens_by_row[i]).astype(np.float32)
            sig_j = self._sig(tokens_by_row[j]).astype(np.float32)
            self.acc[i] += np.float32(poids) * sig_j
            self.acc[j] += np.float32(poids) * sig_i

    def idf(self, token: str) -> float:
        return math.log(1.0 + self.n_docs / (1.0 + self.df.get(token, 0)))

    def word_vector(
        self,
        token: str,
        sig_sign: int = 1,
        embed: np.ndarray | None = None,
        weights: tuple[float, float, float] = WEIGHTS_DEFAULT,
    ) -> np.ndarray:
        a, b, g = weights
        # When embed is None, use (0.5, 0.5, 0.0) for v1 exact compatibility
        if embed is None:
            a, b, g = 0.5, 0.5, 0.0
        sig = self._sig(token).astype(np.float32)
        sig *= np.float32(sig_sign)
        sig /= np.float32(np.linalg.norm(sig))
        parts: list[tuple[float, np.ndarray]] = [(a, sig)]
        idx = self.rows.get(token)
        if idx is not None:
            if idx >= self.acc.shape[0]:
                raise RuntimeError("finalize() doit être appelé avant word_vector()")
            if self.acc[idx].any():
                prof = self.acc[idx].astype(np.float32)
                prof /= np.float32(np.linalg.norm(prof))
                parts.append((b, prof))
        if embed is not None:
            parts.append((g, embed))
        total = sum(w for w, _ in parts)
        v = np.zeros(self.dim, dtype=np.float32)
        for w, comp in parts:
            v += np.float32(w / total) * comp
        return v / np.float32(np.linalg.norm(v))

    def proches(
        self,
        token: str,
        k: int = 10,
        exclus: frozenset[str] = frozenset(),
        seuil_df: float = 0.4,
        dico: frozenset[str] | None = None,
    ) -> list[tuple[str, float]] | None:
        """Voisinage SÉMANTIQUE MÉTIER : les `k` mots les plus proches par cosinus dans l'espace
        de cooccurrence APPRIS du corpus (profils PPMI/SVD-lissés, `self.acc`) — « mots vus dans
        des contextes similaires ici ». None si le token est absent ou sans masse de cooccurrence.
        Distinct de la surface (signature) : c'est le sens distributionnel, pas la morphologie.
        `exclus` retire des mots du résultat (ex. tokens de chemin injectés au build).

        Trois nettoyages : (1) les mots-outils français (`STOPWORDS`) ; (2) le boilerplate
        UNIVERSEL par fréquence documentaire (mot dans plus de `seuil_df` des docs = non
        informatif) ; (3) les mots GÉNÉRIQUES — profil trop proche de la moyenne du corpus
        (verbes/mots vagues présents dans tous les contextes) — les 15 % les plus génériques.

        `dico` (optionnel) : ne garder que les tokens présents dans ce vocabulaire de VRAIS mots
        français (ex. les 160k mots de la table potion, déjà chargés). Écarte proprement le
        boilerplate de contenu (collocations d'en-tête société, codes, dates) qui échappe
        aux trois filtres ci-dessus — au prix des codes-métier utiles (16a, ip20…) absents du
        dictionnaire. À n'activer que si l'on veut un voisinage strictement lexical. Sans `dico`,
        un nom propre mono-mot fréquent (un nom de société) peut subsister (cœur de la NER)."""
        idx = self.rows.get(token)
        v = len(self.rows)
        if idx is None or idx >= self.acc.shape[0] or not self.acc[idx].any():
            return None
        mat = self.acc[:v].astype(np.float32)
        norms = np.linalg.norm(mat, axis=1)
        vivants = norms > 0.0
        q = mat[idx] / norms[idx]
        sims = (mat @ q) / np.where(norms == 0.0, np.float32(1.0), norms)
        moyen = mat[vivants].mean(axis=0)
        moyen = moyen / (float(np.linalg.norm(moyen)) + 1e-9)
        generique = (mat @ moyen) / np.where(norms == 0.0, np.float32(1.0), norms)
        seuil_gen = (
            float(np.percentile(generique[vivants], 85)) if vivants.any() else 1.0
        )
        seuil_df = seuil_df * self.n_docs
        mots = sorted(self.rows, key=self.rows.__getitem__)
        out: list[tuple[str, float]] = []
        for i in np.argsort(-sims):
            mot = mots[i]
            if (
                i == idx
                or norms[i] == 0.0
                or mot in exclus
                or mot in STOPWORDS
                or self.df.get(mot, 0) > seuil_df  # boilerplate universel
                or generique[i] > seuil_gen  # mot générique (proche de la moyenne)
                or (
                    dico is not None
                    and (
                        mot not in dico  # hors dictionnaire réel
                        or sum(c.isdigit() for c in mot) / len(mot) > 0.3  # nombre/code
                    )
                )
            ):
                continue
            out.append((mot, round(float(sims[i]), 4)))
            if len(out) >= k:
                break
        return out
