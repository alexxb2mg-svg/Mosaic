"""Modes d'interrogation de l'Index : search/search_like/related/explain/explain_match.

Fonctions libres prenant l'index en premier argument (`idx`) — les méthodes homonymes de
`Index` ne sont que de fines délégations vers ce module (cf. `mosaic.index`). Pur
déplacement depuis `Index` : aucune logique modifiée.
"""

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mosaic import rerank as rerank_module
from mosaic.collocations import merge
from mosaic.connecteurs import decouper
from mosaic.docio import _EXTS, _read_text, _read_text_convertible
from mosaic.encoder import _signed_counts, encode, quantize
from mosaic.lexicon import canonicalize
from mosaic import ingest
from mosaic.meta import K_RRF_DEFAULT
from mosaic import typage as typage_module
from mosaic import atlas as atlas_module
from mosaic import grammaire as grammaire_module
from mosaic.relations import bind, entites_du_canal, normalize_entity
from mosaic.tokenize import tokenize

if TYPE_CHECKING:
    from mosaic.index import Index


def search(
    idx: "Index",
    text: str,
    k: int = 10,
    rerank: bool = False,
    rerank_lambda: float = 0.70,
    rerank_depth: int = 50,
) -> list[dict]:
    tokens = merge(
        merge(canonicalize(tokenize(text), idx._compiled), idx.colloc),
        idx.colloc,
    )
    q, qnorm = encode(
        tokens,
        idx.profiles,
        embeddings=idx.embeddings,
        weights=idx.weights,
        doc_weight=idx.doc_weight,
    )
    rerank_qvec = None
    # Repêcheur (v1.4) : sans rerank.msrv (au build) ou sans model2vec (au runtime), on
    # refuse net avec une erreur claire — jamais un rerank silencieusement ignoré. Ordre
    # préservé : une requête qui s'annule (qnorm == 0) court-circuite AVANT ces gardes
    # (cf. _rank_and_rerank), exactement comme avant ce refactor.
    if rerank and qnorm != 0.0 and len(idx.ids):
        if idx.rerank_vecs is None:
            raise ValueError(
                "index sans rerank.msrv : reconstruire avec `mosaic build --rerank-vectors` "
                "pour utiliser --rerank"
            )
        if not rerank_module.available():
            raise ValueError(
                '--rerank nécessite model2vec — pip install "model2vec==0.8.2"'
            )
        rerank_qvec = rerank_module.encode_query(text)
    return _rank_and_rerank(
        idx, q, qnorm, k, rerank, rerank_lambda, rerank_depth, rerank_qvec=rerank_qvec
    )


def _cos_all(idx: "Index", text: str) -> np.ndarray:
    """Cosinus de CHAQUE document contre la sous-requête `text` (même encodage que `search`)."""
    n = len(idx.ids)
    if not text.strip() or not n:
        return np.zeros(n, dtype=np.float32)
    tokens = merge(
        merge(canonicalize(tokenize(text), idx._compiled), idx.colloc), idx.colloc
    )
    q, qnorm = encode(
        tokens,
        idx.profiles,
        embeddings=idx.embeddings,
        weights=idx.weights,
        doc_weight=idx.doc_weight,
    )
    if qnorm == 0.0:
        return np.zeros(n, dtype=np.float32)
    scores = idx.mat_recherche @ q.astype(np.float32)
    denom = idx.norms * np.float32(qnorm)
    denom = np.where(denom == 0, np.float32(1.0), denom)
    return scores.astype(np.float32) / denom


def _canaux_typee(
    idx: "Index", text: str
) -> tuple[list[tuple[str, float, np.ndarray]], bool]:
    """Lectures par grille d'un index typé : [(type, masse_idf, cos)], et le drapeau
    « la requête porte un IDENTIFIANT » (tokens ref tous rares, df <= DF_MAX_IDENTIFIANT
    — cf. mosaic.typage, leçon du banc produits). Une grille sans signal est absente."""
    assert idx.grilles is not None
    config = typage_module.config_grilles(idx.profil)
    tokens = merge(
        merge(canonicalize(tokenize(text), idx._compiled), idx.colloc), idx.colloc
    )
    flux = typage_module.router_flux(tokens, [], config, (idx.profil or {}).get("refs"))
    canaux: list[tuple[str, float, np.ndarray]] = []
    ref_identifiant = False
    for t, qtoks in flux.items():
        if not qtoks:
            continue
        if t == "sens":
            prof, mat, norms = idx.profiles, idx.mat_recherche, idx.norms
            emb, poids = idx.embeddings, idx.weights
        else:
            g = idx.grilles.get(t)
            if g is None:
                continue
            prof, mat, norms = g.profiles, g.mat, g.norms
            cfg = g.config
            emb = idx.embeddings if cfg.get("embeddings") else None
            poids = tuple(cfg["poids"]) if cfg.get("poids") else idx.weights
        q, qn = encode(qtoks, prof, embeddings=emb, weights=poids)
        if qn == 0.0:
            continue
        denom = norms * np.float32(qn)
        denom = np.where(denom == 0, np.float32(1.0), denom)
        cos = (mat @ q.astype(np.float32)).astype(np.float32) / denom
        if not np.any(cos):
            continue
        canaux.append((t, sum(prof.idf(x) for x in qtoks), cos))
        if t == "ref":
            ref_identifiant = all(
                prof.df.get(x, 0) <= typage_module.DF_MAX_IDENTIFIANT for x in qtoks
            )
    return canaux, ref_identifiant


def search_typee(
    idx: "Index",
    text: str,
    k: int = 10,
    rerank: bool = False,
    rerank_lambda: float = 0.70,
    rerank_depth: int = 50,
) -> list[dict]:
    """Recherche sur index à GRILLES TYPÉES (v4) : chaque grille est lue avec SA recette,
    la synthèse recombine — pondération par la masse idf de la requête par type, et
    PRÉSÉANCE lexicographique de la lecture ref quand la requête porte un identifiant
    (rare, df <= DF_MAX_IDENTIFIANT = 2, seuil calibré par la mesure). Les deux règles
    sont MESURÉES (banc produits réels, vrai moteur : noyade 0.825 -> 0.90 contre le
    standard avec boost réf facette, désignation préservée par le gate). `lectures`
    expose le cosinus par grille (explicabilité, même esprit que `rangs` de la fusion).

    Repêcheur : comme sur l'index standard (spec v4 « rerank inchangé — la synthèse
    typée remplace le seul canal grille »), le mélange λ·synthèse + (1-λ)·cos_m2v
    re-trie les `rerank_depth` premiers. La préséance ref reste la clé PRIMAIRE sous
    rerank : un cosinus d'embedding ne peut pas détrôner le porteur exact d'un
    identifiant — c'est la garantie mesurée du banc produits, elle survit par
    construction, pas par chance."""
    n = len(idx.ids)
    if n == 0:
        return []
    canaux, ref_identifiant = _canaux_typee(idx, text)
    if not canaux:
        return []
    total = sum(m for _t, m, _c in canaux)
    pondere = np.zeros(n, dtype=np.float64)
    for _t, masse, cos in canaux:
        pondere += (masse / total) * cos.astype(np.float64)
    cos_ref = next((cos for t, _m, cos in canaux if t == "ref"), None)
    # Clé de préséance : lecture ref arrondie, uniquement quand la requête porte un
    # identifiant — None sinon (la variable porte le rétrécissement de type).
    preseance = (
        np.round(cos_ref.astype(np.float64), 4)
        if ref_identifiant and cos_ref is not None
        else None
    )
    if preseance is not None:
        ordre = np.lexsort((-pondere, -preseance))
    else:
        ordre = np.argsort(-pondere, kind="stable")
    lectures = {t: cos for t, _m, cos in canaux}

    def _hit(i: int, extra: dict | None = None) -> dict:
        h = {
            "id": idx.ids[i],
            "score": round(float(pondere[i]), 6),
            "lectures": {t: round(float(c[i]), 4) for t, c in lectures.items()},
        }
        if extra:
            h.update(extra)
        return h

    if not rerank:
        return [_hit(i) for i in ordre[:k]]

    # Mêmes refus nets que l'index standard — jamais un rerank silencieusement ignoré.
    if idx.rerank_vecs is None:
        raise ValueError(
            "index sans rerank.msrv : reconstruire avec `mosaic build --rerank-vectors` "
            "pour utiliser --rerank"
        )
    if not rerank_module.available():
        raise ValueError(
            '--rerank nécessite model2vec — pip install "model2vec==0.8.2"'
        )
    depth = min(rerank_depth, len(ordre))
    depth_idx = ordre[:depth]
    cos_m2v = idx.rerank_vecs[depth_idx] @ rerank_module.encode_query(text)
    blended = np.float64(rerank_lambda) * pondere[depth_idx] + np.float64(
        1.0 - rerank_lambda
    ) * cos_m2v.astype(np.float64)
    if preseance is not None:
        local_order = np.lexsort((-blended, -preseance[depth_idx]))
    else:
        local_order = np.argsort(-blended, kind="stable")
    resorted_idx = depth_idx[local_order]
    blended_sorted = blended[local_order]
    results = [
        _hit(i, {"score_rerank": round(float(b), 6)})
        for i, b in zip(resorted_idx[:k], blended_sorted[:k], strict=True)
    ]
    if len(results) < k:
        for i in ordre[depth : depth + (k - len(results))]:
            results.append(_hit(i))
    return results


def search_grammatical(idx: "Index", text: str, k: int = 10) -> list[dict]:
    """Recherche avec le canal grammatical (opt-in --grammatical, brief 12/08) :
    score = cos(grille) + 0.5·cos(canal structural) — le λ=0.5 est celui du banc P1
    (33/34 paires à mots identiques/sens opposé séparées ; le moteur nu en confond
    25/34 à cosinus 1.0000 exactement). La requête est analysée par les MÊMES règles
    déterministes que les documents (mosaic.grammaire) ; une requête sans rôle donne
    un canal nul — le classement retombe alors sur la grille seule, sans bruit.
    `score_grammatical` expose la contribution structurale (explicabilité)."""
    assert idx.gram_mat is not None and idx.gram_norms is not None
    n = len(idx.ids)
    if n == 0:
        return []
    cos = _cos_all(idx, text)
    dim = idx.grid[0] * idx.grid[1] * idx.grid[2]
    q_g, nq = grammaire_module.canal_document(text, dim)
    cos_g = np.zeros(n, dtype=np.float32)
    if nq > 0.0:
        denom = idx.gram_norms * np.float32(nq)
        denom = np.where(denom == 0, np.float32(1.0), denom)
        cos_g = (idx.gram_mat.astype(np.float32) @ q_g.astype(np.float32)) / denom
    combine = cos.astype(np.float64) + 0.5 * cos_g.astype(np.float64)
    ordre = np.argsort(-combine, kind="stable")[:k]
    return [
        {
            "id": idx.ids[i],
            "score": round(float(combine[i]), 6),
            "score_grammatical": round(float(cos_g[i]), 4),
        }
        for i in ordre
    ]


def search_fusion(idx: "Index", text: str, k: int = 10) -> list[dict]:
    """Fusion RRF à TROIS canaux — grille (sémantique) + BM25 (lexical exact) + embeddings
    (model2vec) — validée par la mesure (banc Alloprof : trio 0.517 R@10 > standard du
    marché BM25+embeddings 0.498 > chaque canal seul). Le duo grille+BM25 sans embeddings
    a été mesuré NUISIBLE (0.460 < BM25 seul 0.482) : la fusion exige donc les trois
    canaux — index construit avec --hybride, refus net sinon (jamais un duo silencieux).

    Chaque canal classe TOUT le corpus ; RRF (même constante K que mosaic.meta) somme
    1/(K+rang). Un canal SANS signal sur cette requête (tous scores nuls : requête hors
    vocabulaire pour BM25, annulée pour la grille) est écarté de la somme — pas de bruit
    de rang injecté par un canal aveugle. Déterministe : égalités départagées par ordre
    de document (argsort stable). `rangs` expose le rang 1-based par canal (explicabilité).

    QUATRIÈME canal quand l'index porte un atlas (build --hybride --atlas, #367) : la
    carte de chaleur de la requête sur l'atlas sémantique, cosinus contre les cartes
    documents — mesuré +2,84 pts R@10 et +3,65 MRR sur Alloprof COMPLET au-dessus du
    trio (les erreurs de la carte SOM sont décorrélées de celles de la grille plate).
    Même règle d'écartement sans signal (requête sans token mappé)."""
    if idx.bm25 is None:
        raise ValueError(
            "index sans bm25.msbm : reconstruire avec `mosaic build --hybride` "
            "pour utiliser la fusion"
        )
    if idx.rerank_vecs is None:
        raise ValueError(
            "index sans rerank.msrv : reconstruire avec `mosaic build --hybride` "
            "pour utiliser la fusion"
        )
    if not rerank_module.available():
        raise ValueError(
            'la fusion nécessite model2vec — pip install "model2vec==0.8.2"'
        )
    n = len(idx.ids)
    if n == 0:
        return []
    tokens = merge(
        merge(canonicalize(tokenize(text), idx._compiled), idx.colloc), idx.colloc
    )
    canaux: list[tuple[str, np.ndarray]] = []
    cos = _cos_all(idx, text)
    if np.any(cos):
        canaux.append(("grille", cos))
    bm = idx.bm25.scores(tokens)
    if np.any(bm):
        canaux.append(("bm25", bm))
    emb = idx.rerank_vecs @ rerank_module.encode_query(text)
    if np.any(emb):
        canaux.append(("embed", emb))
    if (
        idx.atlas_positions is not None
        and idx.atlas_mat is not None
        and idx.atlas_norms is not None
    ):
        carte_q = atlas_module.carte(
            tokens, idx.profiles.rows, idx.atlas_positions, idx.profiles.idf
        )
        nq = float(np.linalg.norm(carte_q))
        if nq > 0.0:
            denom = idx.atlas_norms * np.float32(nq)
            denom = np.where(denom == 0, np.float32(1.0), denom)
            cos_a = (idx.atlas_mat.astype(np.float32) @ carte_q).astype(
                np.float32
            ) / denom
            if np.any(cos_a):
                canaux.append(("atlas", cos_a))
    if not canaux:
        return []
    rrf = np.zeros(n, dtype=np.float64)
    contrib = 1.0 / (K_RRF_DEFAULT + np.arange(1, n + 1, dtype=np.float64))
    rangs: dict[str, np.ndarray] = {}
    for nom, scores in canaux:
        ordre = np.argsort(-scores, kind="stable")
        rrf[ordre] += contrib
        r = np.empty(n, dtype=np.int64)
        r[ordre] = np.arange(1, n + 1)
        rangs[nom] = r
    top = np.argsort(-rrf, kind="stable")[:k]
    return [
        {
            "id": idx.ids[i],
            "score": round(float(rrf[i]), 6),
            "rangs": {nom: int(rangs[nom][i]) for nom, _ in canaux},
        }
        for i in top
    ]


def search_connecteurs(
    idx: "Index", text: str, k: int = 10, lam: float = 0.7
) -> list[dict]:
    """Recherche à ALGÈBRE de connecteurs (« A sans B », « A mais pas B ») : ce qu'on exclut
    fait DESCENDRE le score. score = cos(doc, positif) − λ·cos(doc, négatif). Sans terme négatif,
    retombe exactement sur la recherche normale. Déterministe et explicable (positif/négatif
    exposés par résultat)."""
    positif, negatif = decouper(text)
    if not negatif:
        return search(idx, positif or text, k=k)
    cos_pos = _cos_all(idx, positif)
    cos_neg = _cos_all(idx, negatif)
    combine = cos_pos - np.float32(lam) * cos_neg
    order = np.argsort(-combine)[:k]
    return [
        {
            "id": idx.ids[i],
            "score": round(float(combine[i]), 6),
            "positif": round(float(cos_pos[i]), 4),
            "negatif": round(float(cos_neg[i]), 4),
        }
        for i in order
    ]


def _rank_and_rerank(
    idx: "Index",
    q: np.ndarray,
    qnorm: float,
    k: int,
    rerank: bool,
    rerank_lambda: float,
    rerank_depth: int,
    rerank_qvec: np.ndarray | None = None,
    exclude_rows: frozenset[int] = frozenset(),
) -> list[dict]:
    """Cœur commun à `search()` et `search_like()` : classement cosinus + repêcheur
    optionnel, à partir d'un vecteur de requête DÉJÀ construit (`q`/`qnorm`, et pour le
    repêcheur `rerank_qvec`, déjà résolus par l'appelant — texte encodé pour une requête
    lexicale, ligne stockée ou mélange pour `search_like`). `exclude_rows` retire des
    lignes du classement AVANT toute troncature top-k/profondeur (jamais après) — c'est
    ce qui permet à `search_like` d'exclure le(s) document(s) source(s) d'un id interne."""
    if qnorm == 0.0 or not len(idx.ids):
        return []
    scores = idx.mat_recherche @ q.astype(np.float32)
    denom = idx.norms * np.float32(qnorm)
    denom[denom == 0] = 1.0
    cos = scores.astype(np.float32) / denom
    order = np.argsort(-cos)
    if exclude_rows:
        order = order[~np.isin(order, np.fromiter(exclude_rows, dtype=order.dtype))]
    if not rerank:
        top = order[:k]
        return [{"id": idx.ids[i], "score": round(float(cos[i]), 6)} for i in top]

    if idx.rerank_vecs is None:
        raise ValueError(
            "index sans rerank.msrv : reconstruire avec `mosaic build --rerank-vectors` "
            "pour utiliser --rerank"
        )
    if rerank_qvec is None:
        raise ValueError(
            "rerank demandé sans vecteur de requête rerank (garde interne)"
        )

    depth = min(rerank_depth, len(order))
    depth_idx = order[:depth]
    cos_m2v = idx.rerank_vecs[depth_idx] @ rerank_qvec
    blended = (
        np.float32(rerank_lambda) * cos[depth_idx]
        + np.float32(1.0 - rerank_lambda) * cos_m2v
    )
    local_order = np.argsort(-blended)
    resorted_idx = depth_idx[local_order]
    blended_sorted = blended[local_order]

    results = [
        {
            "id": idx.ids[i],
            "score": round(float(cos[i]), 6),
            "score_rerank": round(float(b), 6),
        }
        for i, b in zip(resorted_idx[:k], blended_sorted[:k], strict=True)
    ]
    if len(results) < k:
        for i in order[depth : depth + (k - len(results))]:
            results.append({"id": idx.ids[i], "score": round(float(cos[i]), 6)})
    return results


def _read_query_document(
    idx: "Index", path: Path, ingest_cache_dir: Path | None
) -> str:
    """Lit un fichier EXTERNE comme document-requête (spec v1.6 §A) : lecture directe pour
    .md/.txt, prisme markitdown pour un convertible (même OCR que celui de l'index — cohérent
    avec le pipeline qui a servi à construire le corpus). Aucune autre extension n'est
    acceptée (jamais un binaire lu à l'aveugle comme du texte).

    Revue (Important, reproduit) : markitdown absent sur un convertible levait un
    « document illisible ou vide » trompeur (indistinguable d'une conversion qui a
    vraiment échoué) — la garde `ingest.available()` est vérifiée EN PREMIER pour lever un
    message actionnable (installer l'extra), avant toute tentative de conversion."""
    suffix = path.suffix.lower()
    if suffix in _EXTS:
        return _read_text(path)
    if suffix in ingest.CONVERTIBLE_EXTS:
        if not ingest.available():
            raise ValueError(
                f'lecture de {suffix} nécessite markitdown — pip install "mosaic-index[ingest]"'
            )
        text = _read_text_convertible(path, ingest_cache_dir, ocr=idx.ocr)
        if text is None:
            raise ValueError(f"document illisible ou vide : {path}")
        return text
    raise ValueError(
        f"extension non prise en charge pour une requête-document : {path.suffix!r} ({path})"
    )


def search_like(
    idx: "Index",
    docs: str | Path | list[str | Path],
    k: int = 10,
    rerank: bool = False,
    rerank_lambda: float = 0.70,
    rerank_depth: int = 50,
    ingest_cache_dir: Path | None = None,
) -> list[dict]:
    """Chiffrage par similarité : la requête est un document ENTIER (spec v1.6 §A), pas une
    phrase. `docs` : un seul id déjà indexé ou chemin de fichier externe, OU une liste de
    2+ (mélange — « ce qui est entre A et B », moyenne des vecteurs unitaires puis
    renormalisation puis re-quantification comme une requête). Id interne : réutilise la
    ligne int8 stockée telle quelle (et son empreinte rerank si présente, --rerank alors
    sans besoin de modèle). Chemin externe : pipeline complet de `search()` (tokens de
    chemin JAMAIS injectés — c'est une requête, pas un document du corpus). Un id interne
    est TOUJOURS exclu des résultats (spec).

    Les ids sont stockés en POSIX (`self.ids`, cf. `Index.build`/`Index.add`) : une saisie
    Windows naturelle avec des antislashs (`2025\\11.2025\\devis.md`) est normalisée AVANT
    comparaison (`\\` → `/`) — sans quoi elle ratait le match d'id, retombait par erreur
    sur la branche fichier externe, et un id interne pouvait alors réapparaître dans ses
    propres résultats (revue, Critical, reproduit). Un id connu est TOUJOURS prioritaire
    sur un chemin de fichier (même chaîne correspondant aux deux : c'est un id)."""
    doc_list = list(docs) if isinstance(docs, list) else [docs]
    if not doc_list:
        raise ValueError(
            "search_like requiert au moins un document (id indexé ou chemin de fichier)"
        )
    doc_norms_seen = [str(d).replace("\\", "/") for d in doc_list]
    if len(doc_norms_seen) != len(set(doc_norms_seen)):
        raise ValueError(
            "search_like : document répété dans le mélange — chaque document ne doit "
            "apparaître qu'une fois"
        )

    dim = idx.mat.shape[1]
    solo = len(doc_list) == 1
    exclude_rows: set[int] = set()
    unit_vecs: list[np.ndarray] = []
    rerank_parts: list[np.ndarray] = []
    quantized_solo: tuple[np.ndarray, float] | None = None

    for doc in doc_list:
        doc_str = str(doc)
        doc_norm = doc_str.replace("\\", "/")
        if doc_norm in idx.ids:
            row = idx.ids.index(doc_norm)
            exclude_rows.add(row)
            norm = float(idx.norms[row])
            unit = (
                idx.mat[row].astype(np.float32) / np.float32(norm)
                if norm != 0.0
                else np.zeros(dim, dtype=np.float32)
            )
            unit_vecs.append(unit)
            if solo:
                quantized_solo = (idx.mat[row], norm)
            if rerank:
                if idx.rerank_vecs is None:
                    raise ValueError(
                        "index sans rerank.msrv : reconstruire avec `mosaic build "
                        "--rerank-vectors` pour utiliser --rerank"
                    )
                rerank_parts.append(idx.rerank_vecs[row])
        else:
            path = Path(doc)
            if not path.is_file():
                raise ValueError(
                    f"document introuvable (ni id indexé, ni chemin de fichier existant) : {doc_str}"
                )
            text = _read_query_document(idx, path, ingest_cache_dir)
            tokens = merge(
                merge(canonicalize(tokenize(text), idx._compiled), idx.colloc),
                idx.colloc,
            )
            q_ext, qnorm_ext = encode(
                tokens,
                idx.profiles,
                embeddings=idx.embeddings,
                weights=idx.weights,
                doc_weight=idx.doc_weight,
            )
            unit = (
                q_ext.astype(np.float32) / np.float32(qnorm_ext)
                if qnorm_ext != 0.0
                else np.zeros(dim, dtype=np.float32)
            )
            unit_vecs.append(unit)
            if solo:
                quantized_solo = (q_ext, qnorm_ext)
            if rerank:
                if idx.rerank_vecs is None:
                    raise ValueError(
                        "index sans rerank.msrv : reconstruire avec `mosaic build "
                        "--rerank-vectors` pour utiliser --rerank"
                    )
                if not rerank_module.available():
                    raise ValueError(
                        '--rerank nécessite model2vec — pip install "model2vec==0.8.2"'
                    )
                rerank_parts.append(rerank_module.encode_query(text))

    if solo:
        assert quantized_solo is not None
        q, qnorm = quantized_solo
    else:
        mean_vec = np.mean(np.stack(unit_vecs), axis=0)
        mean_norm = float(np.linalg.norm(mean_vec))
        unit_mixed = mean_vec / np.float32(mean_norm) if mean_norm > 0.0 else mean_vec
        q, qnorm = quantize(unit_mixed)

    rerank_qvec = None
    if rerank:
        merged = np.mean(np.stack(rerank_parts), axis=0)
        merged_norm = float(np.linalg.norm(merged))
        rerank_qvec = merged / np.float32(merged_norm) if merged_norm > 0.0 else merged

    return _rank_and_rerank(
        idx,
        q,
        qnorm,
        k,
        rerank,
        rerank_lambda,
        rerank_depth,
        rerank_qvec=rerank_qvec,
        exclude_rows=frozenset(exclude_rows),
    )


def related(
    idx: "Index", entite: str, k: int = 10, role: str | None = None
) -> list[dict]:
    """Interrogation du canal de relations (spec v2.0 §Interrogation) : construit la
    requête de liaison — `bind(role, entite)` si `role` est donné, sinon la moyenne de
    `bind(r, entite)` sur les rôles connus pour cette entité (manifeste) — puis produit
    scalaire contre la matrice des canaux, top-k `[{"id", "score"}]`.

    `entite` est normalisée comme au build (`mosaic.relations.normalize_entity`).
    Entité inconnue du manifeste -> [] (pas d'erreur, cf. spec). Index sans
    relations.msrel (jamais construit avec `--relations`) -> ValueError claire."""
    if idx.relations_mat is None:
        raise ValueError(
            "index sans relations.msrel : reconstruire avec `mosaic build --relations` "
            "pour utiliser related()"
        )
    assert idx.relations_norms is not None  # invariant : jamais l'un sans l'autre
    entite_norm = normalize_entity(entite)
    roles = idx.relations_manifest.get(entite_norm)
    if not roles or not len(idx.ids):
        return []
    dim = idx.relations_mat.shape[1]
    if role is not None:
        q = bind(role, entite_norm, dim).astype(np.float32)
    else:
        q = np.mean(
            [bind(r, entite_norm, dim).astype(np.float32) for r in roles], axis=0
        )
    qnorm = float(np.linalg.norm(q))
    if qnorm == 0.0:
        return []
    scores = idx.relations_mat.astype(np.float32) @ q
    denom = idx.relations_norms * np.float32(qnorm)
    denom[denom == 0] = 1.0
    cos = scores / denom
    order = np.argsort(-cos)[:k]
    return [{"id": idx.ids[i], "score": round(float(cos[i]), 6)} for i in order]


def chemin(
    idx: "Index", doc_id: str, k: int = 10, role: str | None = None
) -> list[dict]:
    """Parcours MULTI-SAUTS (v3) : doc -> ses entités (déliage vectoriel du canal, saut 1)
    -> les autres documents de chaque entité (produit scalaire, saut 2). « Les autres
    chantiers de la même année », « les documents du même dossier » — en deux sauts dans
    l'espace des valeurs, avec cleanup à chaque saut (recette validée, multisauts_valeurs).

    `role` restreint le saut 1 à un rôle (dossier/annee/mois). Rend, par entité traversée :
    {"role", "entite", "confiance" (cos du déliage), "documents": [{"id", "score"}]} —
    le document de départ est exclu des frères. Index sans relations.msrel -> ValueError."""
    if idx.relations_mat is None:
        raise ValueError(
            "index sans relations.msrel : reconstruire avec `mosaic build --relations` "
            "pour utiliser chemin()"
        )
    try:
        row = idx.ids.index(doc_id)
    except ValueError:
        raise ValueError(f"document inconnu : {doc_id}") from None
    dim = idx.relations_mat.shape[1]
    # Saut 1 — déliage vectoriel du canal du document (aucune lecture du chemin).
    entites = entites_du_canal(idx.relations_mat[row], idx.relations_manifest, dim)
    if role is not None:
        entites = [e for e in entites if e[0] == role]
    # Saut 2 — pour chaque entité retrouvée, les documents frères (related), départ exclu.
    # Seuil de significativité : related() rend le top-k complet, bruit inclus (un document
    # hors du groupe score ~0) — un frère n'en est un que si son appartenance est nette.
    out: list[dict] = []
    for r, entite, cos in entites:
        freres = [
            h
            for h in related(idx, entite, k=k + 1, role=r)
            if h["id"] != doc_id and h["score"] >= 0.1
        ][:k]
        if freres:
            out.append(
                {"role": r, "entite": entite, "confiance": cos, "documents": freres}
            )
    return out


def _doc_unit_vector(idx: "Index", doc_id: str) -> np.ndarray:
    try:
        row = idx.ids.index(doc_id)
    except ValueError:
        raise ValueError(f"document inconnu : {doc_id}") from None
    norm = idx.norms[row]
    if norm == 0:
        return np.zeros(idx.mat.shape[1], dtype=np.float32)
    return idx.mat[row].astype(np.float32) / np.float32(norm)


def explain(idx: "Index", doc_id: str, k: int = 20) -> list[dict]:
    """Décompose le document en ses tokens du vocabulaire les plus proches (démélange)."""
    doc_vec = _doc_unit_vector(idx, doc_id)
    results = []
    for token in idx.profiles.rows:
        embed = (
            idx.embeddings.vector_grid(token, idx.profiles.dim)
            if idx.embeddings is not None
            else None
        )
        vec = idx.profiles.word_vector(
            token, sig_sign=1, embed=embed, weights=idx.weights
        )
        cos = float(np.dot(vec, doc_vec))
        results.append({"token": token, "poids": round(cos, 4)})
    results.sort(key=lambda r: r["poids"], reverse=True)
    return results[:k]


def explain_match(idx: "Index", query: str, doc_id: str, k: int = 10) -> list[dict]:
    """Contribution de chaque token de la requête (pipeline complet) au score du document."""
    doc_vec = _doc_unit_vector(idx, doc_id)
    tokens = merge(
        merge(canonicalize(tokenize(query), idx._compiled), idx.colloc),
        idx.colloc,
    )
    results = []
    for (token, negated), tf in _signed_counts(tokens).items():
        weight = (1.0 + math.log(tf)) * idx.profiles.idf(token)
        embed = (
            idx.embeddings.vector_grid(token, idx.profiles.dim)
            if idx.embeddings is not None
            else None
        )
        vec = idx.profiles.word_vector(
            token, sig_sign=-1 if negated else 1, embed=embed, weights=idx.weights
        )
        contribution = weight * float(np.dot(vec, doc_vec))
        results.append({"token": token, "poids": round(contribution, 4)})
    results.sort(key=lambda r: r["poids"], reverse=True)
    return results[:k]
