"""Formats binaires .msei (documents) et .msev (vocabulaire + profils)."""

import json
import os
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mosaic.profiles import Profiles

if TYPE_CHECKING:
    from mosaic.bm25 import Bm25

_HEADER = struct.Struct("<4sBBHHBBI")
_MAGIC_DOCS = b"MSEI"
_MAGIC_VOCAB = b"MSEV"
_MAGIC_RERANK = b"MSRV"
_MAGIC_RELATIONS = b"MSRL"
_MAGIC_BM25 = b"MSBM"
_MAGIC_ATLAS = b"MSAT"
_VERSION_MAJOR = 1
_VMIN_DOCS = 0
_VMIN_RELATIONS = 0
# v1.1 vocab (additif) : bloc acc float32 (au lieu de int32) + comptages épars
# (paires + marginaux + masse totale) pour permettre add() incrémental sans
# tout ré-apprendre depuis le corpus. v1.0 (vmin=0) reste lisible en lecture seule.
_VMIN_VOCAB = 1
_VMIN_RERANK = 0
_VMIN_BM25 = 0
_VMIN_ATLAS = 0


def _write(
    path: Path,
    magic: bytes,
    grid: tuple[int, int, int],
    n: int,
    meta: dict,
    arrays: list[np.ndarray],
    vmin: int = 0,
    extra: bytes = b"",
) -> None:
    """Écrit `path` de façon atomique : le contenu complet est d'abord posé dans un fichier
    `.tmp` VOISIN (même dossier, donc même volume), puis `os.replace(tmp, path)` (atomique
    POSIX/NTFS sur un même volume — jamais un état intermédiaire visible par un lecteur).

    Revue finale v1.6 (Critical) : écrire directement dans `path` (ancien comportement)
    laissait un fichier tronqué/à moitié écrit en cas de crash pendant l'écriture — un
    rebuild interrompu, ou un `add()` concurrent d'un lecteur MCP qui garde `vocab.msev`
    ouvert en `np.memmap`, produisait un index torn (docs neufs + vocab corrompu/à moitié
    lu). Avec le fichier temp, si l'écriture elle-même échoue (disque plein…) `path` n'a
    jamais été touché ; si c'est `os.replace()` qui échoue (destination verrouillée par un
    memmap actif, cas Windows), l'échec survient AVANT que l'ancien fichier ne disparaisse —
    échec propre, `path` reste l'ancien contenu intact, jamais un état intermédiaire. Le
    `.tmp` est nettoyé dans tous les cas d'échec (jamais de résidu qui traîne)."""
    blob = json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with open(tmp_path, "wb") as f:
            f.write(
                _HEADER.pack(
                    magic, _VERSION_MAJOR, vmin, grid[0], grid[1], grid[2], 0, n
                )
            )
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            for arr in arrays:
                f.write(arr.tobytes())
            f.write(extra)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass  # nettoyage best-effort — jamais masquer l'exception d'origine
        raise


def _read(
    path: Path, magic: bytes
) -> tuple[tuple[int, int, int], int, dict, int, bytes]:
    data = path.read_bytes()
    try:
        m, vmaj, vmin, w, h, layers, _rsv, n = _HEADER.unpack_from(data, 0)
    except struct.error:
        raise ValueError(f"fichier {path.name} tronqué ou corrompu") from None
    if m != magic or vmaj != _VERSION_MAJOR:
        raise ValueError(f"fichier {path.name} invalide (magic={m!r}, version={vmaj})")
    off = _HEADER.size
    try:
        (blob_len,) = struct.unpack_from("<Q", data, off)
    except struct.error:
        raise ValueError(f"fichier {path.name} tronqué ou corrompu") from None
    off += 8
    meta = json.loads(data[off : off + blob_len].decode("utf-8"))
    return (w, h, layers), n, meta, vmin, data[off + blob_len :]


def save_docs(
    path: Path,
    mat: np.ndarray,
    norms: np.ndarray,
    ids: list[str],
    grid: tuple[int, int, int],
    suffixe: str = "",
) -> None:
    """`suffixe` (v4 grilles typées) : "" = fichier historique docs.msei ; "_ref" ->
    docs_ref.msei — même format, un couple docs/vocab par grille."""
    _write(
        path / f"docs{suffixe}.msei",
        _MAGIC_DOCS,
        grid,
        mat.shape[0],
        {"ids": ids},
        [norms.astype(np.float32), np.ascontiguousarray(mat, dtype=np.int8)],
        vmin=_VMIN_DOCS,
    )


def load_docs(
    path: Path,
    suffixe: str = "",
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[int, int, int]]:
    grid, n, meta, _vmin, raw = _read(path / f"docs{suffixe}.msei", _MAGIC_DOCS)
    dim = grid[0] * grid[1] * grid[2]
    norms = np.frombuffer(raw, dtype=np.float32, count=n)
    mat = np.frombuffer(raw, dtype=np.int8, offset=4 * n).reshape(n, dim).copy()
    return mat, norms.copy(), list(meta["ids"]), grid


def save_atlas(path: Path, positions: np.ndarray, cote: int) -> None:
    """atlas.msat (canal atlas, #367) : cellule de chaque token, int32 ALIGNÉ sur
    l'ordre du vocab (`profiles.rows`, le même ordre que vocab.msev) — aucune liste de
    tokens dupliquée. Le tuple grille du header porte (cote, cote, 1)."""
    _write(
        path / "atlas.msat",
        _MAGIC_ATLAS,
        (cote, cote, 1),
        len(positions),
        {},
        [positions.astype(np.int32)],
        vmin=_VMIN_ATLAS,
    )


def load_atlas(path: Path) -> tuple[np.ndarray, int] | None:
    """None si atlas.msat absent (index construit sans --atlas, dégradation propre).
    ValueError si présent mais corrompu — même contrat que load_bm25."""
    file = path / "atlas.msat"
    if not file.is_file():
        return None
    (cote, _c2, _un), n, _meta, vmin, raw = _read(file, _MAGIC_ATLAS)
    if vmin != _VMIN_ATLAS:
        raise ValueError(f"{file.name} : version mineure non supportée (vmin={vmin})")
    if len(raw) < 4 * n:
        raise ValueError(f"fichier {file.name} tronqué ou corrompu")
    return np.frombuffer(raw, dtype=np.int32, count=n).copy(), int(cote)


def save_rerank(path: Path, mat: np.ndarray, model: str) -> None:
    """rerank.msrv (v1.4 le repêcheur) : vecteurs document model2vec, unitaires, float16,
    même ordre que `ids` de docs.msei (ligne i = document i).

    Réutilise `_write` (magic/version/struct + meta JSON additif) comme docs.msei/vocab.msev —
    le triplet grid sert ici à porter `(dim, 0, 0)`, la 3e composante des autres formats
    (grille mosaïque) n'ayant pas de sens pour une table de vecteurs plats.
    """
    n, dim = mat.shape
    _write(
        path / "rerank.msrv",
        _MAGIC_RERANK,
        (dim, 0, 0),
        n,
        {"model": model},
        [np.ascontiguousarray(mat, dtype=np.float16)],
        vmin=_VMIN_RERANK,
    )


def load_rerank(path: Path) -> tuple[np.ndarray, str] | None:
    """None si rerank.msrv absent (dégradation propre, pas une erreur). ValueError si présent
    mais corrompu/tronqué/version mineure non supportée — même contrat que load_docs/load_vocab."""
    file = path / "rerank.msrv"
    if not file.is_file():
        return None
    (dim, _h, _layers), n, meta, vmin, raw = _read(file, _MAGIC_RERANK)
    if vmin != _VMIN_RERANK:
        raise ValueError(f"{file.name} : version mineure non supportée (vmin={vmin})")
    expected = n * dim * 2  # float16 = 2 octets
    if len(raw) < expected:
        raise ValueError(f"fichier {file.name} tronqué ou corrompu")
    mat = np.frombuffer(raw, dtype=np.float16, count=n * dim).reshape(n, dim).copy()
    return mat, str(meta.get("model", ""))


def save_bm25(path: Path, bm25: "Bm25") -> None:
    """bm25.msbm (fusion hybride) : postings BM25 par terme, même ordre de documents que
    `ids` de docs.msei. Le vocabulaire (liste de termes, ordre = colonnes) voyage dans le
    meta JSON comme `ids` dans docs.msei ; les tableaux suivent dans l'ordre fixe
    indptr(int64, V+1) · doc_idx(int32, nnz) · tf(int32, nnz) · doc_lens(int32, n).
    Le triplet grid du header n'a pas de sens ici -> (0, 0, 0), comme rerank.msrv."""
    nnz = int(bm25.indptr[-1])
    _write(
        path / "bm25.msbm",
        _MAGIC_BM25,
        (0, 0, 0),
        bm25.n_docs,
        {"vocab": bm25.vocab_termes, "nnz": nnz},
        [
            bm25.indptr.astype(np.int64),
            bm25.doc_idx.astype(np.int32),
            bm25.tf.astype(np.int32),
            bm25.doc_lens.astype(np.int32),
        ],
        vmin=_VMIN_BM25,
    )


def load_bm25(path: Path) -> "Bm25 | None":
    """None si bm25.msbm absent (index construit sans --hybride, dégradation propre).
    ValueError si présent mais corrompu/tronqué — même contrat que load_docs/load_rerank."""
    from mosaic.bm25 import Bm25  # import local : store ne dépend du module qu'ici

    file = path / "bm25.msbm"
    if not file.is_file():
        return None
    _grid, n, meta, vmin, raw = _read(file, _MAGIC_BM25)
    if vmin != _VMIN_BM25:
        raise ValueError(f"{file.name} : version mineure non supportée (vmin={vmin})")
    termes = [str(t) for t in meta["vocab"]]
    v, nnz = len(termes), int(meta["nnz"])
    expected = 8 * (v + 1) + 4 * nnz + 4 * nnz + 4 * n
    if len(raw) < expected:
        raise ValueError(f"fichier {file.name} tronqué ou corrompu")
    off = 0
    indptr = np.frombuffer(raw, dtype=np.int64, count=v + 1, offset=off).copy()
    off += 8 * (v + 1)
    doc_idx = np.frombuffer(raw, dtype=np.int32, count=nnz, offset=off).copy()
    off += 4 * nnz
    tf = np.frombuffer(raw, dtype=np.int32, count=nnz, offset=off).copy()
    off += 4 * nnz
    doc_lens = np.frombuffer(raw, dtype=np.int32, count=n, offset=off).copy()
    return Bm25(termes, indptr, doc_idx, tf, doc_lens)


def save_relations(
    path: Path,
    mat: np.ndarray,
    norms: np.ndarray,
    manifest: Mapping[str, Iterable[str]],
    grid: tuple[int, int, int],
) -> None:
    """relations.msrel (v2.0, canal de relations — spec §Stockage et format) : mêmes
    conventions que docs.msei (bloc norms float32 puis matrice int8, même ordre que
    `ids`), plus un manifeste JSON des entités connues (`{entite: [roles]}`, triés) pour
    l'énumération/auto-complétion et pour `Index.related(role=None)`."""
    meta = {"manifest": {k: sorted(v) for k, v in manifest.items()}}
    _write(
        path / "relations.msrel",
        _MAGIC_RELATIONS,
        grid,
        mat.shape[0],
        meta,
        [norms.astype(np.float32), np.ascontiguousarray(mat, dtype=np.int8)],
        vmin=_VMIN_RELATIONS,
    )


def load_relations(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]], tuple[int, int, int]] | None:
    """None si relations.msrel absent (dégradation propre — index construit sans
    `--relations`, cf. spec). ValueError si présent mais corrompu/tronqué/version
    mineure non supportée — même contrat que load_docs/load_rerank."""
    file = path / "relations.msrel"
    if not file.is_file():
        return None
    grid, n, meta, vmin, raw = _read(file, _MAGIC_RELATIONS)
    if vmin != _VMIN_RELATIONS:
        raise ValueError(f"{file.name} : version mineure non supportée (vmin={vmin})")
    dim = grid[0] * grid[1] * grid[2]
    expected = 4 * n + n * dim  # norms float32 (4o/doc) + matrice int8 (1o/doc/dim)
    if len(raw) < expected:
        raise ValueError(f"fichier {file.name} tronqué ou corrompu")
    norms = np.frombuffer(raw, dtype=np.float32, count=n)
    mat = (
        np.frombuffer(raw, dtype=np.int8, count=n * dim, offset=4 * n)
        .reshape(n, dim)
        .copy()
    )
    manifest = {
        str(k): [str(r) for r in v] for k, v in meta.get("manifest", {}).items()
    }
    return mat, norms.copy(), manifest, grid


def save_vocab(
    path: Path,
    profiles: Profiles,
    colloc: set[tuple[str, str]],
    grid: tuple[int, int, int],
    lexicon: dict[str, str],
    extra_meta: dict | None = None,
    suffixe: str = "",
) -> None:
    order = sorted(profiles.rows, key=profiles.rows.__getitem__)
    meta = {
        "order": order,
        "counts": profiles.counts,
        "df": profiles.df,
        "n_docs": profiles.n_docs,
        "colloc": sorted(list(c) for c in colloc),
        "lexicon": lexicon,
    }
    if extra_meta:
        meta.update(extra_meta)
    acc = np.ascontiguousarray(profiles.acc[: len(order)], dtype=np.float32)

    # Bloc épars, itéré trié — déterminisme indépendant de l'ordre d'insertion des dicts.
    pair_keys = sorted(profiles.pair_counts)
    n_pairs = len(pair_keys)
    rows_arr = np.array([k[0] for k in pair_keys], dtype=np.uint32)
    cols_arr = np.array([k[1] for k in pair_keys], dtype=np.uint32)
    poids_arr = np.array([profiles.pair_counts[k] for k in pair_keys], dtype=np.float32)

    marg_keys = sorted(profiles.marginals)
    n_marg = len(marg_keys)
    marg_rows_arr = np.array(marg_keys, dtype=np.uint32)
    marg_arr = np.array([profiles.marginals[k] for k in marg_keys], dtype=np.float32)

    extra = (
        struct.pack("<Q", n_pairs)
        + rows_arr.tobytes()
        + cols_arr.tobytes()
        + poids_arr.tobytes()
        + struct.pack("<I", n_marg)
        + marg_rows_arr.tobytes()
        + marg_arr.tobytes()
        + struct.pack("<d", profiles.total_mass)
    )
    _write(
        path / f"vocab{suffixe}.msev",
        _MAGIC_VOCAB,
        grid,
        len(order),
        meta,
        [acc],
        vmin=_VMIN_VOCAB,
        extra=extra,
    )


def _parse_sparse_block(
    raw: bytes, off: int = 0
) -> tuple[dict[tuple[int, int], float], dict[int, float], float]:
    """Analyse le bloc épars (paires + marginaux + masse totale) à partir de `off` dans `raw`.

    Factorisé pour être appelable soit immédiatement (load_vocab eager), soit en différé
    depuis le chargeur posé par load_vocab(lazy=True) sur Profiles._pending_sparse — même
    format, même erreurs, un seul endroit qui sait le lire."""
    try:
        (n_pairs,) = struct.unpack_from("<Q", raw, off)
        off += 8
        rows_arr = np.frombuffer(raw, dtype=np.uint32, count=n_pairs, offset=off)
        off += 4 * n_pairs
        cols_arr = np.frombuffer(raw, dtype=np.uint32, count=n_pairs, offset=off)
        off += 4 * n_pairs
        poids_arr = np.frombuffer(raw, dtype=np.float32, count=n_pairs, offset=off)
        off += 4 * n_pairs
        (n_marg,) = struct.unpack_from("<I", raw, off)
        off += 4
        marg_rows_arr = np.frombuffer(raw, dtype=np.uint32, count=n_marg, offset=off)
        off += 4 * n_marg
        marg_arr = np.frombuffer(raw, dtype=np.float32, count=n_marg, offset=off)
        off += 4 * n_marg
        (total_mass,) = struct.unpack_from("<d", raw, off)
    except struct.error:
        raise ValueError("vocab.msev tronqué ou corrompu") from None

    pair_counts = {
        (int(r), int(c)): float(w)
        for r, c, w in zip(
            rows_arr.tolist(), cols_arr.tolist(), poids_arr.tolist(), strict=True
        )
    }
    marginals = {
        int(r): float(w)
        for r, w in zip(marg_rows_arr.tolist(), marg_arr.tolist(), strict=True)
    }
    return pair_counts, marginals, float(total_mass)


def load_vocab(
    path: Path, lazy: bool = False, suffixe: str = ""
) -> tuple[Profiles, set[tuple[str, str]], dict[str, str], dict]:
    """`lazy=True` (v1.5, réservé à Index.open côté recherche) : évite de lire les ~V·dim·4
    octets du bloc acc (et le bloc épars, jamais utile à search()) en RAM. `acc` devient un
    np.memmap read-only ouvert directement sur vocab.msev (lignes lues à la demande par
    word_vector()) et le bloc épars n'est pas analysé — un chargeur est posé sur
    `Profiles._pending_sparse`, déclenché par le premier learn()/finalize() (add()).
    Fichiers legacy v1.0 : comportement inchangé quel que soit `lazy` (pas de memmap
    pertinent — acc y est en int32, add() est de toute façon refusé côté Index)."""
    file = path / f"vocab{suffixe}.msev"
    if not lazy:
        return _load_vocab_eager(file)
    return _load_vocab_lazy(file)


def _load_vocab_eager(
    file: Path,
) -> tuple[Profiles, set[tuple[str, str]], dict[str, str], dict]:
    grid, n, meta, vmin, raw = _read(file, _MAGIC_VOCAB)
    dim = grid[0] * grid[1] * grid[2]
    p = Profiles(dim)
    p.rows = {t: i for i, t in enumerate(meta["order"])}
    p.counts = dict(meta["counts"])
    p.df = dict(meta["df"])
    p.n_docs = int(meta["n_docs"])
    colloc = {(a, b) for a, b in meta["colloc"]}
    result_meta = dict(meta)

    if vmin == 0:
        # v1.0 : acc int32, aucun comptage épars — lecture seule (add() refusé côté Index).
        acc_i32 = np.frombuffer(raw, dtype=np.int32, count=n * dim).reshape(n, dim)
        p.acc = acc_i32.astype(np.float32).copy()
        p.pair_counts = {}
        p.marginals = {}
        p.total_mass = 0.0
        result_meta["legacy"] = True
        return p, colloc, dict(meta.get("lexicon", {})), result_meta

    if vmin != 1:
        raise ValueError(f"vocab.msev : version mineure non supportée (vmin={vmin})")

    p.acc = (
        np.frombuffer(raw, dtype=np.float32, count=n * dim, offset=0)
        .reshape(n, dim)
        .copy()
    )
    p.pair_counts, p.marginals, p.total_mass = _parse_sparse_block(raw, off=4 * n * dim)
    result_meta["legacy"] = False
    return p, colloc, dict(meta.get("lexicon", {})), result_meta


def _load_vocab_lazy(
    file: Path,
) -> tuple[Profiles, set[tuple[str, str]], dict[str, str], dict]:
    with open(file, "rb") as f:
        header_bytes = f.read(_HEADER.size)
        try:
            m, vmaj, vmin, w, h, layers, _rsv, n = _HEADER.unpack(header_bytes)
        except struct.error:
            raise ValueError(f"fichier {file.name} tronqué ou corrompu") from None
        if m != _MAGIC_VOCAB or vmaj != _VERSION_MAJOR:
            raise ValueError(
                f"fichier {file.name} invalide (magic={m!r}, version={vmaj})"
            )
        blob_len_bytes = f.read(8)
        try:
            (blob_len,) = struct.unpack("<Q", blob_len_bytes)
        except struct.error:
            raise ValueError(f"fichier {file.name} tronqué ou corrompu") from None
        blob = f.read(blob_len)
        if len(blob) < blob_len:
            raise ValueError(f"fichier {file.name} tronqué ou corrompu")
        meta = json.loads(blob.decode("utf-8"))
        acc_offset = f.tell()

    if vmin == 0:
        # legacy : rien de paresseux à en tirer (int32, add() refusé) — même chemin qu'eager.
        return _load_vocab_eager(file)
    if vmin != 1:
        raise ValueError(f"vocab.msev : version mineure non supportée (vmin={vmin})")

    grid = (w, h, layers)
    dim = grid[0] * grid[1] * grid[2]
    file_size = file.stat().st_size
    acc_bytes = n * dim * 4
    if acc_offset + acc_bytes > file_size:
        raise ValueError("vocab.msev tronqué ou corrompu")

    p = Profiles(dim)
    p.rows = {t: i for i, t in enumerate(meta["order"])}
    p.counts = dict(meta["counts"])
    p.df = dict(meta["df"])
    p.n_docs = int(meta["n_docs"])
    colloc = {(a, b) for a, b in meta["colloc"]}
    result_meta = dict(meta)

    # memmap read-only direct sur le fichier : aucune lecture des V·dim float32 en RAM ici,
    # les lignes sont matérialisées à la demande par word_vector() (search() n'en touche
    # qu'une poignée — celles des tokens de la requête).
    p.acc = np.memmap(
        file, dtype=np.float32, mode="r", offset=acc_offset, shape=(n, dim)
    )
    sparse_offset = acc_offset + acc_bytes

    def _sparse_loader() -> tuple[
        dict[tuple[int, int], float], dict[int, float], float
    ]:
        with open(file, "rb") as f2:
            f2.seek(sparse_offset)
            raw = f2.read()
        return _parse_sparse_block(raw, off=0)

    p._pending_sparse = _sparse_loader
    result_meta["legacy"] = False
    return p, colloc, dict(meta.get("lexicon", {})), result_meta
