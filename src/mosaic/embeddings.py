"""Table d'embeddings statiques : sous-ensemble fastText projeté vers l'espace grille."""

import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

_MAGIC = b"MSEE"
_VERSION = (1, 0)
_HEADER = struct.Struct("<4sBBIH")
PROJ_SEED = 0x4D4F5341

_proj_cache: dict[tuple[int, int], np.ndarray] = {}


def _projection(dim_src: int, dim_grid: int) -> np.ndarray:
    key = (dim_src, dim_grid)
    mat = _proj_cache.get(key)
    if mat is None:
        rng = np.random.default_rng(PROJ_SEED)
        mat = rng.choice(
            np.array([-1.0, 0.0, 1.0], dtype=np.float32),
            size=(dim_src, dim_grid),
            p=[1 / 6, 2 / 3, 1 / 6],
        )
        _proj_cache[key] = mat
    return mat


def prepare(
    vec_gz: Path,
    out: Path,
    keep: int = 200_000,
    extra_words: set[str] | None = None,
    abtt: int = 0,
) -> dict:
    """`abtt` (v1.6 §C) : applique all-but-the-top À LA PRÉPARATION (sur la matrice complète,
    avant écriture) plutôt qu'au chargement — élimine le coût PCA (~1-1.6 s) du premier
    `Embeddings.load(..., abtt=N)` quand N correspond à la valeur pré-appliquée. La table
    écrite bascule alors en vmin=1 et porte un octet `abtt_applique` juste après l'en-tête
    (cf. `Embeddings.load` pour la lecture symétrique). abtt=0 (défaut) : format inchangé
    depuis v1.0, vmin=0, aucun octet supplémentaire — un ancien lecteur relit ces fichiers à
    l'identique."""
    if abtt < 0 or abtt > 255:
        raise ValueError(f"abtt doit être dans [0, 255] (octet u8) : {abtt}")
    extra = {w.lower() for w in (extra_words or set())}
    words: list[str] = []
    rows: list[np.ndarray] = []
    seen: set[str] = set()
    with gzip.open(vec_gz, "rt", encoding="utf-8", errors="replace") as f:
        _n, dim = (int(x) for x in f.readline().split())
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split(" ")
            if len(parts) != dim + 1:
                continue
            word = parts[0].lower()
            if word in seen or (i >= keep and word not in extra):
                continue
            seen.add(word)
            words.append(word)
            rows.append(np.asarray(parts[1:], dtype=np.float32).astype(np.float16))
    mat = np.vstack(rows) if rows else np.zeros((0, dim), dtype=np.float16)
    if abtt > 0 and mat.shape[0] > 0:
        mat = _all_but_the_top(mat.astype(np.float32), abtt).astype(np.float16)
    vmin = 1 if abtt > 0 else 0
    blob = json.dumps(words, ensure_ascii=False).encode("utf-8")
    with open(out, "wb") as fo:
        fo.write(_HEADER.pack(_MAGIC, _VERSION[0], vmin, len(words), dim))
        if vmin >= 1:
            fo.write(struct.pack("<B", abtt))
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        fo.write(np.ascontiguousarray(mat).tobytes())
    return {"kept": len(words), "dim": dim}


def _all_but_the_top(mat: np.ndarray, abtt: int) -> np.ndarray:
    """Retire la moyenne et les `abtt` composantes principales dominantes (Mu & Viswanath 2018).

    Déterministe : covariance 300×300 (symétrique) diagonalisée par `np.linalg.eigh`
    (ordre d'eigenvalues croissant garanti par LAPACK), vecteurs propres des plus
    grandes valeurs propres pris en ordre décroissant, signe normalisé (composante de
    plus grande valeur absolue imposée positive) pour que le résultat ne dépende pas
    d'un choix de signe arbitraire du solveur.
    """
    mu = mat.mean(axis=0, dtype=np.float32)
    centered = mat - mu
    cov = (centered.T @ centered.astype(np.float64)) / centered.shape[0]
    _eigvals, eigvecs = np.linalg.eigh(cov)
    top = eigvecs[:, -abtt:][:, ::-1].astype(
        np.float32
    )  # `abtt` plus grandes, ordre décroissant
    flip = top[np.argmax(np.abs(top), axis=0), np.arange(top.shape[1])] < 0
    top[:, flip] *= -1.0
    projections = centered @ top  # (n, abtt)
    return (centered - projections @ top.T).astype(np.float32)


class Embeddings:
    def __init__(
        self,
        words: list[str],
        mat: np.ndarray | None,
        dim: int,
        sha: str,
        abtt: int = 0,
        verified: bool = True,
        raw_f16: np.ndarray | None = None,
    ) -> None:
        self.rows = {w: i for i, w in enumerate(words)}
        self.mat = mat
        self.dim = dim
        self.sha = sha
        self.abtt = abtt
        # v1.5 : True (défaut) = table lue intégralement + sha256 vérifiable (build()/add(),
        # et Index.open() par défaut). False = chargement paresseux (verify=False, cf. load()),
        # réservé au chemin recherche — `sha` vaut "" et n'a jamais été comparé à embed_sha.
        self.verified = verified
        # memmap float16 (n, dim) en attente de transformation abtt globale — non None
        # seulement entre load(verify=False, abtt>0) et le premier accès réel (_materialized()).
        self._raw_f16 = raw_f16
        self._grid_cache: dict[tuple[str, int], np.ndarray] = {}

    @classmethod
    def load(cls, path: Path, abtt: int = 0, verify: bool = True) -> "Embeddings":
        """`verify=True` (défaut) : lecture complète du fichier + sha256 d'intégrité —
        comportement historique, utilisé par `build()`/`add()` (jamais de compromis sur
        l'intégrité au moment d'écrire un index) et par `Index.open()` par défaut.

        `verify=False` (v1.5, réservé au chemin recherche via `Index.open(...,
        verify_embeddings=False)`) : la matrice float16 est ouverte en `np.memmap` read-only
        au lieu d'être lue intégralement (évite ~84 Mo de lecture + le calcul sha256 sur le
        plus gros index de prod), et aucun sha n'est calculé (`.sha == ""`, `.verified ==
        False` — l'appelant saute alors la comparaison avec `embed_sha` du meta ; cf.
        `Index.stats()["embeddings_verifies"]`).

        Avec `abtt > 0`, la transformation all-but-the-top a besoin de la moyenne/covariance
        GLOBALE de la table : impossible ligne par ligne à la demande. En mode paresseux,
        elle est donc différée au premier accès réel (`vector_grid`/`raw_vector`, via
        `_materialized()`) plutôt que calculée ici — `load()` reste instantané, le coût
        (une lecture complète du memmap, une fois par process) se déplace sur la première
        requête plutôt que sur l'ouverture.

        v1.6 §C — table pré-nettoyée : si le fichier porte `abtt_applique > 0` (écrit par
        `prepare(..., abtt=N)`, vmin >= 1), la transformation a déjà eu lieu à la
        préparation. `abtt_applique == abtt` demandé -> AUCUNE transformation ici (fast
        path : la matrice sur disque est utilisée telle quelle, `.abtt` vaut `abtt`) ;
        `abtt_applique > 0` et différent de `abtt` demandé -> `ValueError` claire (la table
        ne peut pas être "dé-nettoyée" ni re-nettoyée à une autre valeur au chargement) ;
        `abtt_applique == 0` (vmin=0, ancien format ou `prepare(..., abtt=0)`) -> comportement
        historique inchangé (transformation appliquée ici si `abtt > 0`).
        """
        path = Path(path)
        if verify:
            return cls._load_eager(path, abtt)
        return cls._load_lazy(path, abtt)

    @classmethod
    def _unpack_header(cls, data: bytes, path_name: str) -> tuple[int, int, int, int]:
        try:
            magic, vmaj, vmin, n, dim = _HEADER.unpack_from(data, 0)
        except struct.error:
            raise ValueError(f"table {path_name} tronquée ou corrompue") from None
        if magic != _MAGIC or vmaj != _VERSION[0]:
            raise ValueError(f"table {path_name} invalide (magic={magic!r})")
        return vmin, n, dim, _HEADER.size

    @classmethod
    def _check_abtt_applique(
        cls, abtt_applique: int, abtt: int, path_name: str
    ) -> None:
        if abtt_applique > 0 and abtt != abtt_applique:
            raise ValueError(
                f"table {path_name} pré-nettoyée à la préparation avec abtt={abtt_applique} "
                f"(cf. `mosaic embed-prepare --abtt`) : incompatible avec abtt={abtt} demandé "
                "au chargement — recharger avec le même abtt ou reconstruire la table"
            )

    @classmethod
    def _load_eager(cls, path: Path, abtt: int) -> "Embeddings":
        data = path.read_bytes()
        vmin, n, dim, off = cls._unpack_header(data, path.name)
        abtt_applique = 0
        if vmin >= 1:
            try:
                (abtt_applique,) = struct.unpack_from("<B", data, off)
            except struct.error:
                raise ValueError(f"table {path.name} tronquée ou corrompue") from None
            off += 1
        cls._check_abtt_applique(abtt_applique, abtt, path.name)
        try:
            (blob_len,) = struct.unpack_from("<Q", data, off)
        except struct.error:
            raise ValueError(f"table {path.name} tronquée ou corrompue") from None
        off += 8
        words = json.loads(data[off : off + blob_len].decode("utf-8"))
        mat = np.frombuffer(
            data, dtype=np.float16, offset=off + blob_len, count=n * dim
        ).reshape(n, dim)
        sha = hashlib.sha256(data).hexdigest()
        if abtt_applique > 0:
            # fast path (v1.6 §C) : déjà nettoyée à la préparation, aucun recalcul PCA.
            mat = mat.copy()
        elif abtt > 0 and n > 0:
            mat = _all_but_the_top(mat.astype(np.float32), abtt)
        else:
            mat = mat.copy()
        return cls(words, mat, dim, sha, abtt=abtt, verified=True)

    @classmethod
    def _load_lazy(cls, path: Path, abtt: int) -> "Embeddings":
        with open(path, "rb") as f:
            header_bytes = f.read(_HEADER.size)
            vmin, n, dim, _off = cls._unpack_header(header_bytes, path.name)
            abtt_applique = 0
            if vmin >= 1:
                abtt_byte = f.read(1)
                try:
                    (abtt_applique,) = struct.unpack("<B", abtt_byte)
                except struct.error:
                    raise ValueError(
                        f"table {path.name} tronquée ou corrompue"
                    ) from None
            cls._check_abtt_applique(abtt_applique, abtt, path.name)
            blob_len_bytes = f.read(8)
            try:
                (blob_len,) = struct.unpack("<Q", blob_len_bytes)
            except struct.error:
                raise ValueError(f"table {path.name} tronquée ou corrompue") from None
            blob = f.read(blob_len)
            if len(blob) < blob_len:
                raise ValueError(f"table {path.name} tronquée ou corrompue")
            words = json.loads(blob.decode("utf-8"))
            mat_offset = f.tell()

        if n == 0:
            return cls(
                words,
                np.zeros((0, dim), dtype=np.float16),
                dim,
                "",
                abtt=abtt,
                verified=False,
            )

        expected = n * dim * 2  # float16 = 2 octets
        if mat_offset + expected > path.stat().st_size:
            raise ValueError(f"table {path.name} tronquée ou corrompue")

        raw = np.memmap(
            path, dtype=np.float16, mode="r", offset=mat_offset, shape=(n, dim)
        )
        if abtt_applique > 0:
            # fast path (v1.6 §C) : déjà nettoyée à la préparation, memmap exposé tel quel,
            # jamais de PCA différée (contrairement au cas abtt_applique == 0 ci-dessous).
            return cls(words, raw, dim, "", abtt=abtt, verified=False)
        if abtt > 0:
            # PCA globale différée au premier accès réel — cf. docstring de load().
            return cls(words, None, dim, "", abtt=abtt, verified=False, raw_f16=raw)
        return cls(words, raw, dim, "", abtt=abtt, verified=False)

    def _materialized(self) -> np.ndarray:
        """Matrice utilisable ligne par ligne : matérialise le transform all-but-the-top une
        seule fois si le chargement était paresseux avec abtt > 0 (cf. `load()`). No-op
        sinon — `self.mat` est déjà utilisable (ndarray réel en mode eager, ou memmap direct
        en mode paresseux avec abtt == 0, indexable ligne par ligne sans lecture complète)."""
        if self.mat is None:
            full = np.asarray(self._raw_f16).astype(np.float32)
            self.mat = _all_but_the_top(full, self.abtt)
            self._raw_f16 = None  # libère la référence au memmap, plus jamais utile
        return self.mat

    def raw_vector(self, token: str) -> np.ndarray | None:
        """Ligne BRUTE de la table (espace source, non normalisée par mot), ou None si absent.

        Contrairement à `vector_grid`, ne projette pas et ne normalise pas — utilisé par le
        canal document (somme pondérée IDF dans l'espace brut, normalisation une seule fois
        après sommation, cf. mosaic.encoder.doc_channel). Copie systématique (jamais une vue
        sur self.mat) : l'appelant peut accumuler dessus sans risque d'aliasing.
        """
        idx = self.rows.get(token)
        if idx is None:
            return None
        return self._materialized()[idx].astype(np.float32)

    def proches(self, token: str, k: int = 10) -> list[tuple[str, float]] | None:
        """Voisinage sémantique d'un mot : les `k` mots les plus proches par cosinus dans la
        table (esprit Cémantix, mais sur CETTE table — donc CE domaine et CETTE langue). None
        si le mot est absent de la table ; le mot lui-même est exclu du résultat."""
        q = self.raw_vector(token)
        if q is None:
            return None
        mat = self._materialized().astype(np.float32)
        qn = q / (float(np.linalg.norm(q)) + 1e-9)
        sims = (mat @ qn) / (np.linalg.norm(mat, axis=1) + 1e-9)
        mots = sorted(self.rows, key=self.rows.__getitem__)  # mots en ordre d'indice
        out: list[tuple[str, float]] = []
        for i in np.argsort(-sims):
            mot = mots[i]
            if mot == token:
                continue
            out.append((mot, round(float(sims[i]), 4)))
            if len(out) >= k:
                break
        return out

    def project_raw(self, vec: np.ndarray, grid_dim: int) -> np.ndarray:
        """Projette un vecteur brut (espace source `self.dim`) vers la grille `grid_dim`.

        Réutilise le même cache module `_projection` que `vector_grid` — expose la
        projection publiquement sans exposer l'attribut privé. Ne normalise PAS le
        résultat (l'appelant décide, cf. formule D̂(doc) = normaliser(projeter(D(doc)))).
        """
        return vec.astype(np.float32) @ _projection(self.dim, grid_dim)

    def vector_grid(self, token: str, grid_dim: int) -> np.ndarray | None:
        key = (token, grid_dim)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        idx = self.rows.get(token)
        if idx is None:
            return None
        vec = self._materialized()[idx].astype(np.float32) @ _projection(
            self.dim, grid_dim
        )
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return None
        vec /= np.float32(norm)
        self._grid_cache[key] = vec
        return vec
