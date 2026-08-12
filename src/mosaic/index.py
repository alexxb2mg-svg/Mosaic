"""Index Mosaic : construction, recherche, ajout incrémental."""

from pathlib import Path

import numpy as np

from mosaic import GRID_DEFAULT, ingest, queries
from mosaic import rerank as rerank_module
from mosaic import facettes as facettes_module
from mosaic import profil as profil_module
from mosaic.bm25 import Bm25
from mosaic import typage as typage_module
from mosaic import atlas as atlas_module
from mosaic import grammaire as grammaire_module
from mosaic.typage import GrilleTypee
from mosaic.collocations import detect, merge
from mosaic.docio import _EXTS, _path_tokens, _read_text, _read_text_convertible
from mosaic.embeddings import Embeddings
from mosaic.encoder import WEIGHTS_DEFAULT, encode, quantize
from mosaic.lexicon import canonicalize, compile_lexicon, load_lexicon
from mosaic.profiles import Profiles
from mosaic.relations import (
    document_channel,
    entities_from_path,
    entities_from_path_profil,
)
from mosaic.smoothing import smooth
from mosaic.store import (
    load_atlas,
    load_bm25,
    load_docs,
    load_relations,
    load_rerank,
    load_vocab,
    save_atlas,
    save_bm25,
    save_docs,
    save_relations,
    save_rerank,
    save_vocab,
)
from mosaic.tokenize import tokenize

# Défauts de construction (bench v1.2, corpus réel 2100 docs) : ppmi domine brut sur
# recall@10/MRR sur la quasi-totalité des configurations testées, et un lissage de rang
# 300 apporte un gain net supplémentaire sans dégrader les requêtes lexicales. brut/0
# restent atteignables explicitement (Index.build(profile_weighting="brut",
# smoothing_rank=0) ou CLI --profile-weighting brut --smoothing-rank 0).
PROFILE_WEIGHTING_DEFAULT = "ppmi"
SMOOTHING_RANK_DEFAULT = 300

# Répertoires jamais indexés, même s'ils contiennent des extensions par ailleurs
# indexables (ex. _MOSAIC/cartes.html). Sans cette exclusion, `mosaic carte`
# s'auto-indexerait à chaque rebuild : la sortie d'un run polluerait le corpus
# du suivant. v1.5 : _backups/_corbeille/_cimetiere/poubelleClaude ajoutés —
# leçon des 50 tests réels (les sauvegardes/corbeilles polluaient le top des
# résultats). Tout composant de chemin correspondant exclut le fichier, à
# n'importe quelle profondeur (cf. le filtre `EXCLUDED_DIRS & set(parts)` plus bas).
EXCLUDED_DIRS = {"_MOSAIC", "_backups", "_corbeille", "_cimetiere", "poubelleClaude"}


def _round_f16(mat: np.ndarray) -> np.ndarray:
    """Arrondit au format de persistance de rerank.msrv (float16) puis re-upcast en float32 —
    garantit que la précision en mémoire (juste après build()/add()) est identique à celle
    d'une ré-ouverture (Index.open() lit rerank.msrv, forcément float16 sur disque)."""
    return mat.astype(np.float16).astype(np.float32)


def _apply_smoothing(profiles: Profiles, rank: int) -> None:
    """Remplace les lignes vivantes de `profiles.acc` par leur approximation de rang faible.

    Ne touche que les V = len(profiles.rows) premières lignes (acc peut être
    sur-alloué par des blocs de croissance futurs). No-op si rank <= 0.
    """
    if rank <= 0:
        return
    v = len(profiles.rows)
    # out=même tranche : évite la copie float32 intermédiaire n×d (plafond RAM du
    # build — cf. smoothing.smooth, bit-identique garanti)
    smooth(profiles.acc[:v], rank, out=profiles.acc[:v])


class Index:
    def __init__(
        self,
        index_dir: Path,
        profiles: Profiles,
        colloc: set[tuple[str, str]],
        mat: np.ndarray,
        norms: np.ndarray,
        ids: list[str],
        grid: tuple[int, int, int],
        lexicon: dict[str, str],
        embeddings: Embeddings | None = None,
        embed_path: Path | None = None,
        weights: tuple[float, float, float] = WEIGHTS_DEFAULT,
        profile_weighting: str = PROFILE_WEIGHTING_DEFAULT,
        legacy: bool = False,
        smoothing_rank: int = SMOOTHING_RANK_DEFAULT,
        abtt: int = 0,
        ignores: int = 0,
        doc_weight: float = 0.0,
        rerank_vecs: np.ndarray | None = None,
        rerank_model: str | None = None,
        index_paths: bool = True,
        ocr: bool = False,
        relations_mat: np.ndarray | None = None,
        relations_norms: np.ndarray | None = None,
        relations_manifest: dict[str, set[str]] | None = None,
        facettes: dict[str, dict[str, str]] | None = None,
        profil: dict | None = None,
        bm25: Bm25 | None = None,
        grilles: dict[str, GrilleTypee] | None = None,
        atlas_positions: np.ndarray | None = None,
        atlas_mat: np.ndarray | None = None,
        atlas_norms: np.ndarray | None = None,
        gram_mat: np.ndarray | None = None,
        gram_norms: np.ndarray | None = None,
    ) -> None:
        self.index_dir = index_dir
        self.profiles = profiles
        self.colloc = colloc
        self.mat = mat
        self.norms = norms
        self.ids = ids
        self.grid = grid
        self.lexicon = lexicon
        self._compiled = compile_lexicon(lexicon)
        self.embeddings = embeddings
        self.embed_path = embed_path
        self.weights = weights
        self.profile_weighting = profile_weighting
        self.legacy = legacy
        self.smoothing_rank = smoothing_rank
        self.abtt = abtt
        self.ignores = ignores
        self.doc_weight = doc_weight
        self.rerank_vecs = rerank_vecs
        self.rerank_model = rerank_model
        self.index_paths = index_paths
        self.ocr = ocr
        # Canal de relations (v2.0, opt-in --relations) : relations_mat est None quand
        # l'index n'a pas de relations.msrel (défaut, comportement v1.6 inchangé). Non
        # None -> matrice int8 N×dim + normes alignées sur `ids`, manifeste
        # {entite: {roles}} pour related(role=None) et l'énumération.
        self.relations_mat = relations_mat
        self.relations_norms = relations_norms
        self.relations_manifest: dict[str, set[str]] = (
            relations_manifest if relations_manifest is not None else {}
        )
        # Facettes (métadonnées exactes par document : type, date) — {} pour un index
        # antérieur sans facettes.json (dégradation propre : filtre/récence refusés loud).
        self.facettes: dict[str, dict[str, str]] = (
            facettes if facettes is not None else {}
        )
        # Profil d'index (paramétrage déclaratif métier) — None = défauts historiques.
        # Persisté dans le meta : build/add/search relisent LE MÊME profil (règle 1).
        self.profil: dict | None = profil
        # Canal BM25 (fusion hybride, opt-in --hybride) : None quand l'index n'a pas de
        # bm25.msbm — search(fusion=True) refuse loud (cf. queries.search_fusion).
        self.bm25: Bm25 | None = bm25
        # Grilles typées (v4, opt-in --grilles-typees) : None = index historique à
        # grille unique (comportement inchangé). Non-None = dict des grilles EXTRA
        # (ref/chemin/custom) — la grille « sens » est la grille principale
        # (mat/profiles ci-dessus). search() route et synthétise (queries.search_typee).
        self.grilles: dict[str, GrilleTypee] | None = grilles
        # Canal atlas (#367, opt-in --atlas) : cellule de chaque token du vocab
        # (positions, ordre profiles.rows) + cartes de chaleur documents int8 alignées
        # sur `ids`. None = index sans atlas.msat — la fusion reste à trois canaux.
        self.atlas_positions: np.ndarray | None = atlas_positions
        self.atlas_mat: np.ndarray | None = atlas_mat
        self.atlas_norms: np.ndarray | None = atlas_norms
        # Canal grammatical (opt-in --grammatical) : traces de rôles liées par
        # bind/np.roll, vecteur SÉPARÉ par document — None = index sans canal
        # (comportement bit-identique). Cf. mosaic.grammaire.
        self.gram_mat: np.ndarray | None = gram_mat
        self.gram_norms: np.ndarray | None = gram_norms
        # Cache float32 OPTIONNEL de la matrice documents (chemin chaud). Le produit
        # int8 @ float upcaste toute la matrice À CHAQUE requête (mesuré : 577 ms sur
        # 18k docs contre 51 ms pré-converti — 11x). Trade-off explicite RAM vs latence :
        # None par défaut (CLI froide, RAM minimale) ; activer via chauffer_recherche()
        # (le serveur MCP le fait — c'est lui le chemin chaud). Invalidé par add().
        self._mat32: np.ndarray | None = None

    def chauffer_recherche(self) -> None:
        """Pré-convertit la matrice documents en float32 (RAM x4 sur ce bloc, recherche
        ~11x plus rapide sur les gros index). À réserver aux process longs (serveur)."""
        if self._mat32 is None:
            self._mat32 = self.mat.astype(np.float32)

    @property
    def mat_recherche(self) -> np.ndarray:
        """La matrice à utiliser pour le produit scalaire de recherche : le cache float32
        s'il a été chauffé, sinon la matrice int8 (sobre, upcast payé par requête)."""
        return self._mat32 if self._mat32 is not None else self.mat

    # -- construction -------------------------------------------------------

    # ------------------------------------------------------------------
    # build : phases nommées (chirurgie 12/08 — l'orchestrateur reste plat,
    # chaque phase est un pur DÉPLACEMENT de code : mêmes opérations, même
    # ordre, bit-identité vérifiée par sha256 d'index complets, défauts et
    # tous canaux). Une phase = testable seule ; le flux de données entre
    # phases est explicite dans les signatures, plus aucun état partagé.
    # ------------------------------------------------------------------

    @staticmethod
    def _build_valider(
        corpus_dir: Path,
        atlas: bool,
        hybride: bool,
        grilles_typees: bool,
        grammatical: bool,
        rerank_vectors: bool,
        profil: dict | None,
        embeddings: Embeddings | None,
        embeddings_path: Path | None,
        abtt: int,
        weights: tuple[float, float, float] | None,
    ) -> tuple[
        bool,
        dict | None,
        Embeddings | None,
        Path | None,
        tuple[float, float, float],
    ]:
        """Phase 1 — refus forts et tôt : gardes de composition des canaux,
        validation du profil, disponibilité model2vec, cohérence embeddings/abtt.
        Rend (rerank_vectors, profil, embeddings, embed_path, weights) résolus."""
        if not corpus_dir.is_dir():
            raise ValueError(f"corpus introuvable : {corpus_dir}")
        if atlas and not hybride:
            # Le canal atlas n'est validé qu'en QUATUOR (grille+BM25+embeddings+atlas,
            # +2,84 pts R@10 plein corpus, #367) — jamais un sous-ensemble silencieux,
            # même règle que le refus du duo par la fusion.
            raise ValueError(
                "--atlas exige --hybride : le canal atlas n'est validé qu'en quatuor "
                "de fusion (grille + BM25 + embeddings + atlas)"
            )
        if atlas and grilles_typees:
            raise ValueError(
                "--atlas et --grilles-typees ne sont pas composés (v1) : le flux typé "
                "fragmenterait la carte — à mesurer avant de coder"
            )
        if grammatical and grilles_typees:
            # Découvert par l'A/B du 12/08 : la combinaison construisait un index
            # INOUVRABLE (canal grammatical stocké en dimension d'origine, en-tête
            # écrit avec la grille typée redimensionnée — reshape impossible à
            # l'ouverture). Refus net plutôt qu'un artefact corrompu silencieux.
            raise ValueError(
                "--grammatical et --grilles-typees ne sont pas composés (v1) : "
                "l'index produit serait inouvrable (géométries divergentes) — "
                "à composer et mesurer avant d'autoriser"
            )
        if hybride:
            # Un index hybride = les TROIS canaux de la fusion (grille + BM25 + embeddings) :
            # le trio est ce que la mesure a validé, le duo grille+BM25 a été mesuré nuisible
            # (cf. mosaic.bm25). --hybride implique donc les vecteurs rerank.
            rerank_vectors = True
        if profil is not None:
            profil = profil_module.valider(profil)  # casse fort et tôt (règle 2)
        if rerank_vectors and not rerank_module.available():
            raise ValueError(
                "model2vec non installé — --rerank-vectors nécessite "
                'pip install "model2vec==0.8.2"'
            )
        embed_path = (
            Path(embeddings_path).resolve() if embeddings_path is not None else None
        )
        if embeddings is not None and embed_path is None:
            raise ValueError(
                "embeddings fourni sans embeddings_path : le meta ne pourrait pas être "
                "persisté pour une ré-ouverture symétrique"
            )
        if embeddings is None and embed_path is not None:
            embeddings = Embeddings.load(embed_path, abtt=abtt)
        elif embeddings is not None and embeddings.abtt != abtt:
            raise ValueError(
                f"embeddings.abtt ({embeddings.abtt}) != abtt demandé ({abtt}) : "
                "la table fournie n'a pas été chargée avec le même paramètre all-but-the-top"
            )
        weights = WEIGHTS_DEFAULT if weights is None else weights
        return rerank_vectors, profil, embeddings, embed_path, weights

    @staticmethod
    def _build_lire_corpus(
        corpus_dir: Path,
        dim_grille: int,
        profil: dict | None,
        ingest_cache_dir: Path | None,
        ocr: bool,
        grilles_typees: bool,
        index_paths: bool,
        type_doc: bool,
        rerank_vectors: bool,
        grammatical: bool,
    ) -> tuple[
        list[tuple[str, list[str]]],
        list[list[str]],
        list[str],
        list[tuple[np.ndarray, float]],
        dict[str, dict[str, str]],
        int,
    ]:
        """Phase 2 — la seule passe qui LIT les documents : tokens de contenu,
        chemins (flux à part si typé), facettes, textes rerank et rôles
        grammaticaux (sur texte brut, DANS la boucle — le texte n'est pas
        conservé). Un document illisible est ignoré ET compté, jamais silencieux."""
        files = sorted(
            p
            for p in corpus_dir.rglob("*")
            if (
                p.suffix.lower() in _EXTS
                or p.suffix.lower() in ingest.CONVERTIBLE_EXTS
                or p.suffix.lower() in ingest.IMAGE_EXTS
            )
            and not (EXCLUDED_DIRS & set(p.relative_to(corpus_dir).parts))
        )
        raw: list[tuple[str, list[str]]] = []
        chemins_bruts: list[
            list[str]
        ] = []  # grilles typées : le chemin est un FLUX à part
        rerank_texts: list[str] = []
        # Canal grammatical (opt-in) : les rôles s'analysent sur le TEXTE BRUT (ordre
        # des mots + ponctuation), jamais sur le flux canonicalisé — calcul DANS la
        # boucle de lecture, le texte n'est pas conservé.
        gram_rows: list[tuple[np.ndarray, float]] = []
        ext_gram = grammaire_module.extension_depuis_profil(profil)
        facettes: dict[str, dict[str, str]] = {}
        ignores = 0
        for p in files:
            doc_id = p.relative_to(corpus_dir).as_posix()
            if p.suffix.lower() in _EXTS:
                text = _read_text(p)
            else:
                text = _read_text_convertible(p, ingest_cache_dir, ocr=ocr)
                if text is None:
                    ignores += 1
                    continue
            content_tokens = tokenize(text)
            if grilles_typees:
                # provenance séparée : le chemin ira dans SA grille (jamais superposé
                # au contenu — c'est le principe même du tri à l'écriture)
                tokens = content_tokens
                chemins_bruts.append(_path_tokens(doc_id) if index_paths else [])
            else:
                tokens = (
                    _path_tokens(doc_id) + content_tokens
                    if index_paths
                    else content_tokens
                )
            facettes[doc_id] = facettes_module.extraire(p, doc_id, text, profil=profil)
            if type_doc:  # facette « type de document » aussi DANS la grille (opt-in)
                tokens = tokenize(facettes[doc_id]["type"]) + tokens
            raw.append((doc_id, tokens))
            if rerank_vectors:
                rerank_texts.append(text)
            if grammatical:
                gram_rows.append(
                    grammaire_module.canal_document(text, dim_grille, ext_gram)
                )
        return raw, chemins_bruts, rerank_texts, gram_rows, facettes, ignores

    @staticmethod
    def _build_router_typee(
        docs: list[tuple[str, list[str]]],
        chemins_bruts: list[list[str]],
        compiled,
        profil: dict | None,
        dim: int,
        grid: tuple[int, int, int],
    ) -> tuple[
        list[tuple[str, list[str]]],
        list[dict[str, list[str]]],
        dict[str, dict],
        int,
        tuple[int, int, int],
    ]:
        """Phase 4 (opt-in v4) — routage à l'écriture : chaque document est réparti
        entre les grilles, le flux « sens » devient la grille principale,
        dimensionnée au vocabulaire réel."""
        config_grilles = typage_module.config_grilles(profil)
        regles_refs = (profil or {}).get("refs")
        chemins_canon = [canonicalize(c, compiled) for c in chemins_bruts]
        flux_par_doc = [
            typage_module.router_flux(
                tokens, chemins_canon[i], config_grilles, regles_refs
            )
            for i, (_doc_id, tokens) in enumerate(docs)
        ]
        docs = [
            (doc_id, flux["sens"])
            for (doc_id, _t), flux in zip(docs, flux_par_doc, strict=True)
        ]
        vocab_sens = len({t for f in flux_par_doc for t in f["sens"]})
        dim = typage_module.dim_effective(
            int(config_grilles["sens"]["dim"]), vocab_sens
        )
        grid = typage_module.grille_de_dim(dim)
        return docs, flux_par_doc, config_grilles, dim, grid

    @staticmethod
    def _build_encoder_grille(
        docs: list[tuple[str, list[str]]],
        dim: int,
        profile_weighting: str,
        smoothing_rank: int,
        embeddings: Embeddings | None,
        weights: tuple[float, float, float],
        doc_weight: float,
    ) -> tuple[Profiles, np.ndarray, np.ndarray, list[str]]:
        """Phase 5 — le cœur : apprentissage des profils de cooccurrence
        (learn/finalize/lissage) puis encodage int8 de chaque document."""
        profiles = Profiles(dim)
        for _, tokens in docs:
            profiles.learn(tokens)
        profiles.finalize(profile_weighting)
        _apply_smoothing(profiles, smoothing_rank)
        mat = np.zeros((len(docs), dim), dtype=np.int8)
        norms = np.zeros(len(docs), dtype=np.float32)
        ids: list[str] = []
        for row, (doc_id, tokens) in enumerate(docs):
            q, n = encode(
                tokens,
                profiles,
                embeddings=embeddings,
                weights=weights,
                doc_weight=doc_weight,
            )
            mat[row], norms[row] = q, n
            ids.append(doc_id)
        return profiles, mat, norms, ids

    @staticmethod
    def _build_canal_bm25(
        docs: list[tuple[str, list[str]]],
        flux_par_doc: list[dict[str, list[str]]],
        hybride: bool,
        grilles_typees: bool,
    ) -> Bm25 | None:
        """Phase 6a — postings BM25 sur le MÊME flux de tokens que la grille
        (canonical + collocations) : les deux canaux de la fusion voient le même
        monde. Index typé : le flux complet est la réunion des flux par grille."""
        if hybride and grilles_typees:
            return Bm25.from_docs(
                [
                    (doc_id, [t for typ in sorted(flux) for t in flux[typ]])
                    for (doc_id, _s), flux in zip(docs, flux_par_doc, strict=True)
                ]
            )
        return Bm25.from_docs(docs) if hybride else None

    @staticmethod
    def _build_canal_grammatical(
        gram_rows: list[tuple[np.ndarray, float]], dim_grille: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Phase 6b — empile les rôles analysés en phase 2 (int8 + normes)."""
        gram_mat = (
            np.stack([v for v, _n in gram_rows]).astype(np.int8)
            if gram_rows
            else np.zeros((0, dim_grille), dtype=np.int8)
        )
        gram_norms = np.array([n for _v, n in gram_rows], dtype=np.float32)
        return gram_mat, gram_norms

    @staticmethod
    def _build_canal_atlas(
        docs: list[tuple[str, list[str]]], profiles: Profiles
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Phase 6c (#367) — SOM déterministe sur les profils du vocabulaire ->
        cellule par token, puis carte de chaleur tf×idf par document (même flux
        de tokens que la grille et BM25), quantifiée int8 comme les grilles."""
        atlas_positions = atlas_module.construire_mapping(profiles)
        atlas_mat = np.zeros(
            (len(docs), atlas_module.COTE * atlas_module.COTE), dtype=np.int8
        )
        atlas_norms = np.zeros(len(docs), dtype=np.float32)
        for row, (_doc_id, toks) in enumerate(docs):
            m = atlas_module.carte(toks, profiles.rows, atlas_positions, profiles.idf)
            atlas_mat[row], atlas_norms[row] = quantize(m)
        return atlas_positions, atlas_mat, atlas_norms

    @staticmethod
    def _build_grilles_extra(
        config_grilles: dict[str, dict],
        flux_par_doc: list[dict[str, list[str]]],
        profile_weighting: str,
        embeddings: Embeddings | None,
        weights: tuple[float, float, float],
    ) -> dict[str, "GrilleTypee"]:
        """Phase 6d (v4) — grilles typées EXTRA (ref/chemin/custom), la grille sens
        étant déjà la principale. Chaque grille : sa dimension (taillée au vocab),
        ses poids, son lissage (JAMAIS sur des identifiants), ses embeddings."""
        grilles: dict[str, GrilleTypee] = {}
        for t, cfg in config_grilles.items():
            if t == "sens":
                continue
            flux_t = [f[t] for f in flux_par_doc]
            vocab_t = len({tok for toks in flux_t for tok in toks})
            dim_t = typage_module.dim_effective(
                int(cfg["dim"]), vocab_t, cfg.get("poids")
            )
            prof_t = Profiles(dim_t)
            for toks in flux_t:
                prof_t.learn(toks)
            # finalize INCONDITIONNEL : rows peut être vide alors que la grille
            # sert (une réf SEULE par document ne forme aucune paire de
            # cooccurrence — la signature porte tout le signal ; état normal).
            prof_t.finalize(profile_weighting)
            if prof_t.rows:
                _apply_smoothing(prof_t, int(cfg.get("lissage") or 0))
            mat_t = np.zeros((len(flux_t), dim_t), dtype=np.int8)
            norms_t = np.zeros(len(flux_t), dtype=np.float32)
            emb_t = embeddings if cfg.get("embeddings") else None
            poids_t = tuple(cfg["poids"]) if cfg.get("poids") else weights
            for row, toks in enumerate(flux_t):
                q_t, n_t = encode(toks, prof_t, embeddings=emb_t, weights=poids_t)
                mat_t[row], norms_t[row] = q_t, n_t
            grilles[t] = GrilleTypee(prof_t, mat_t, norms_t, dict(cfg))
        return grilles

    @classmethod
    def build(
        cls,
        corpus_dir: Path,
        index_dir: Path,
        grid: tuple[int, int, int] = GRID_DEFAULT,
        lexicon: dict[str, str] | None = None,
        embeddings: Embeddings | None = None,
        embeddings_path: Path | None = None,
        weights: tuple[float, float, float] | None = None,
        profile_weighting: str = PROFILE_WEIGHTING_DEFAULT,
        smoothing_rank: int = SMOOTHING_RANK_DEFAULT,
        abtt: int = 0,
        ingest_cache_dir: Path | None = None,
        doc_weight: float = 0.0,
        rerank_vectors: bool = False,
        index_paths: bool = True,
        ocr: bool = False,
        relations: bool = False,
        type_doc: bool = False,
        profil: dict | None = None,
        hybride: bool = False,
        grilles_typees: bool = False,
        atlas: bool = False,
        grammatical: bool = False,
    ) -> "Index":
        """Construit un index — orchestrateur PLAT, une ligne par phase :

        1. valider (refus forts et tôt)         _build_valider
        2. lire le corpus (seule passe I/O)     _build_lire_corpus
        3. canonicaliser + collocations         (inline : trois lignes)
        4. router vers les grilles typées       _build_router_typee (opt-in)
        5. apprendre + encoder la grille        _build_encoder_grille
        6. canaux annexes                       _build_canal_* / _build_grilles_extra
        7. assembler + persister                cls(...) + _save()
        """
        corpus_dir, index_dir = Path(corpus_dir), Path(index_dir)
        rerank_vectors, profil, embeddings, embed_path, weights = cls._build_valider(
            corpus_dir,
            atlas,
            hybride,
            grilles_typees,
            grammatical,
            rerank_vectors,
            profil,
            embeddings,
            embeddings_path,
            abtt,
            weights,
        )
        dim = grid[0] * grid[1] * grid[2]
        dim_grille = dim  # le canal grammatical reste sur la géométrie d'origine
        raw, chemins_bruts, rerank_texts, gram_rows, facettes, ignores = (
            cls._build_lire_corpus(
                corpus_dir,
                dim_grille,
                profil,
                ingest_cache_dir,
                ocr,
                grilles_typees,
                index_paths,
                type_doc,
                rerank_vectors,
                grammatical,
            )
        )
        if lexicon is None:
            lexicon = load_lexicon()
        compiled = compile_lexicon(lexicon)
        canon = [(doc_id, canonicalize(tokens, compiled)) for doc_id, tokens in raw]
        colloc = detect([t for _, t in canon])
        docs = [
            (doc_id, merge(merge(tokens, colloc), colloc)) for doc_id, tokens in canon
        ]
        config_grilles: dict[str, dict] = {}
        flux_par_doc: list[dict[str, list[str]]] = []
        if grilles_typees:
            docs, flux_par_doc, config_grilles, dim, grid = cls._build_router_typee(
                docs, chemins_bruts, compiled, profil, dim, grid
            )
        profiles, mat, norms, ids = cls._build_encoder_grille(
            docs,
            dim,
            profile_weighting,
            smoothing_rank,
            embeddings,
            weights,
            doc_weight,
        )
        rerank_vecs = None
        rerank_model = None
        if rerank_vectors:
            # Arrondi float16 immédiat : rerank.msrv persiste en float16 (cf. save_rerank),
            # donc un search() juste après build() doit voir exactement la même précision
            # qu'un search() après Index.open() — jamais deux comportements pour le même index.
            rerank_vecs = _round_f16(rerank_module.encode_texts(rerank_texts))
            # Le nom du modèle RÉELLEMENT chargé (MOSAIC_POTION_MODEL_DIR compris) —
            # la constante mentait quand l'env pointait un autre modèle (bug 12/08).
            rerank_model = rerank_module.model_name_effectif()
        bm25 = cls._build_canal_bm25(docs, flux_par_doc, hybride, grilles_typees)
        gram_mat = gram_norms = None
        if grammatical:
            gram_mat, gram_norms = cls._build_canal_grammatical(gram_rows, dim_grille)
        atlas_positions = atlas_mat = atlas_norms = None
        if atlas:
            atlas_positions, atlas_mat, atlas_norms = cls._build_canal_atlas(
                docs, profiles
            )
        grilles: dict[str, GrilleTypee] | None = None
        if grilles_typees:
            grilles = cls._build_grilles_extra(
                config_grilles, flux_par_doc, profile_weighting, embeddings, weights
            )
        index_dir.mkdir(parents=True, exist_ok=True)
        idx = cls(
            index_dir,
            profiles,
            colloc,
            mat,
            norms,
            ids,
            grid,
            lexicon,
            embeddings=embeddings,
            embed_path=embed_path,
            weights=weights,
            profile_weighting=profile_weighting,
            legacy=False,
            smoothing_rank=smoothing_rank,
            abtt=abtt,
            ignores=ignores,
            doc_weight=doc_weight,
            rerank_vecs=rerank_vecs,
            rerank_model=rerank_model,
            index_paths=index_paths,
            ocr=ocr,
            facettes=facettes,
            profil=profil,
            bm25=bm25,
            grilles=grilles,
            atlas_positions=atlas_positions,
            atlas_mat=atlas_mat,
            atlas_norms=atlas_norms,
            gram_mat=gram_mat,
            gram_norms=gram_norms,
        )
        if relations:
            idx._build_relations()
        idx._save()
        return idx

    @classmethod
    def open(
        cls,
        index_dir: Path,
        embeddings_path: Path | None = None,
        lazy: bool = True,
        verify_embeddings: bool = True,
    ) -> "Index":
        """`lazy=True` (défaut v1.5) : le bloc acc du vocabulaire (float32, V×dim — le plus
        gros bloc d'un index, 3 Go sur le plus gros index de prod) est ouvert en np.memmap
        au lieu d'être copié en RAM, et le bloc épars (jamais lu par search()) n'est pas
        analysé. search() ne lit que les quelques lignes d'acc correspondant aux tokens de
        la requête. add()/finalize() continuent de fonctionner : le premier learn() déclenche
        la lecture complète du bloc épars, et finalize() réalloue acc en un tableau réel
        (voir Profiles._materialize_sparse/finalize — coupe toute référence au memmap avant
        que _save() ne ré-écrive vocab.msev, piège Windows où un memmap actif verrouille le
        fichier contre l'écrasement).

        `verify_embeddings=False` (v1.5, réservé à la CLI `search`) : `Embeddings.load(...,
        verify=False)` — table d'embeddings ouverte en memmap (pas de lecture complète), sha
        non calculé donc non comparé à `embed_sha` du meta (`stats()["embeddings_verifies"]`
        le signale). Toujours True par défaut : build()/add() n'ont jamais de raison de faire
        confiance sans vérifier, seule une recherche peut se permettre de ne pas relire les
        dizaines de Mo de la table à chaque ouverture."""
        index_dir = Path(index_dir)
        mat, norms, ids, grid = load_docs(index_dir)
        profiles, colloc, lexicon, meta = load_vocab(index_dir, lazy=lazy)
        # Garde de cohérence inter-fichiers : _save() écrit docs.msei puis vocab.msev en
        # deux gestes atomiques indépendants. Si le 2e a échoué (verrou Windows, crash),
        # docs.msei porte N+1 documents et vocab.msev n'en connaît que N — index déchiré.
        # On refuse loud plutôt que de servir en silence des scores incohérents (revue v1.6).
        _n_docs_vocab = meta.get("n_docs")
        if _n_docs_vocab is not None and mat.shape[0] != _n_docs_vocab:
            raise ValueError(
                f"index incohérent : {mat.shape[0]} documents dans docs.msei mais "
                f"{_n_docs_vocab} dans vocab.msev — écriture interrompue, reconstruire l'index"
            )
        embeddings = None
        embed_path = None
        weights = WEIGHTS_DEFAULT
        abtt = int(meta.get("abtt", 0))
        if "embed_path" in meta:
            weights = tuple(meta.get("weights", weights))
            embed_path = (
                Path(embeddings_path).resolve()
                if embeddings_path is not None
                else Path(meta["embed_path"])
            )
            if not embed_path.is_file():
                raise ValueError(f"table d'embeddings introuvable : {embed_path}")
            embeddings = Embeddings.load(
                embed_path, abtt=abtt, verify=verify_embeddings
            )
            if verify_embeddings and embeddings.sha != meta["embed_sha"]:
                raise ValueError(
                    f"table d'embeddings altérée (sha différent) : {embed_path}"
                )
        # Fallback "brut"/0 intentionnellement figé (≠ PROFILE_WEIGHTING_DEFAULT/
        # SMOOTHING_RANK_DEFAULT) : un index sur disque sans ces clés en meta a été
        # construit avant leur existence, donc avec le comportement v1 d'origine — pas
        # avec les nouveaux défauts de build.
        profile_weighting = meta.get("profile_weighting", "brut")
        legacy = bool(meta.get("legacy", False))
        smoothing_rank = int(meta.get("smoothing_rank", 0))
        ignores = int(meta.get("ignores", 0))
        # défaut 0.0 : un index sur disque sans cette clé a été construit avant l'existence
        # du canal document (v1.2 ou antérieur) -> comportement v1.2 exact à la réouverture.
        doc_weight = float(meta.get("doc_weight", 0.0))
        # défaut False pour les deux (v1.5) : un index sur disque sans ces clés a été
        # construit avant leur existence -> comportement v1.4 exact à la réouverture, et
        # surtout à un add() ultérieur (jamais d'injection/OCR qu'un build n'avait pas).
        index_paths = bool(meta.get("index_paths", False))
        ocr = bool(meta.get("ocr", False))
        rerank_vecs = None
        rerank_model = None
        loaded_rerank = load_rerank(index_dir)
        if loaded_rerank is not None:
            rerank_mat, rerank_model = loaded_rerank
            if rerank_mat.shape[0] != len(ids):
                raise ValueError(
                    "rerank.msrv incohérent avec l'index (nombre de documents différent) — "
                    "reconstruire avec `mosaic build --rerank-vectors`"
                )
            rerank_vecs = rerank_mat.astype(np.float32)
        relations_mat = None
        relations_norms = None
        relations_manifest: dict[str, set[str]] = {}
        loaded_relations = load_relations(index_dir)
        if loaded_relations is not None:
            relations_mat, relations_norms, manifest_lists, _rel_grid = loaded_relations
            if relations_mat.shape[0] != len(ids):
                raise ValueError(
                    "relations.msrel incohérent avec l'index (nombre de documents différent) — "
                    "reconstruire avec `mosaic build --relations`"
                )
            relations_manifest = {k: set(v) for k, v in manifest_lists.items()}
        bm25 = load_bm25(index_dir)
        if bm25 is not None and bm25.n_docs != len(ids):
            raise ValueError(
                "bm25.msbm incohérent avec l'index (nombre de documents différent) — "
                "reconstruire avec `mosaic build --hybride`"
            )
        gram_mat = gram_norms = None
        if (index_dir / "docs_grammatical.msei").is_file():
            gram_mat, gram_norms, ids_gram, _grid_g = load_docs(
                index_dir, suffixe="_grammatical"
            )
            if ids_gram != ids:
                raise ValueError(
                    "docs_grammatical.msei incohérent avec l'index (documents "
                    "différents) — reconstruire avec `mosaic build --grammatical`"
                )
        atlas_positions = atlas_mat = atlas_norms = None
        atlas_charge = load_atlas(index_dir)
        if atlas_charge is not None:
            atlas_positions, _cote = atlas_charge
            atlas_mat, atlas_norms, ids_atlas, _grid_a = load_docs(
                index_dir, suffixe="_atlas"
            )
            if ids_atlas != ids:
                raise ValueError(
                    "docs_atlas.msei incohérent avec l'index (documents différents) — "
                    "reconstruire avec `mosaic build --hybride --atlas`"
                )
        grilles: dict[str, GrilleTypee] | None = None
        gt_meta = meta.get("grilles_typees")
        if gt_meta:
            grilles = {}
            for t, cfg in gt_meta.items():
                if t == "sens":  # la grille sens EST la grille principale déjà chargée
                    continue
                mat_t, norms_t, _ids_t, _grid_t = load_docs(index_dir, suffixe=f"_{t}")
                if mat_t.shape[0] != len(ids):
                    raise ValueError(
                        f"docs_{t}.msei incohérent avec l'index (nombre de documents "
                        "différent) — reconstruire avec `mosaic build --grilles-typees`"
                    )
                prof_t, _c, _l, _m = load_vocab(index_dir, lazy=lazy, suffixe=f"_{t}")
                grilles[t] = GrilleTypee(prof_t, mat_t, norms_t, dict(cfg))
        # acc est déjà lissé sur disque (build/add l'ont matérialisé) : on ne
        # réapplique PAS smooth() ici, seulement au prochain add() (re-finalize).
        return cls(
            index_dir,
            profiles,
            colloc,
            mat,
            norms,
            ids,
            grid,
            lexicon,
            embeddings=embeddings,
            embed_path=embed_path,
            weights=weights,
            profile_weighting=profile_weighting,
            legacy=legacy,
            smoothing_rank=smoothing_rank,
            abtt=abtt,
            ignores=ignores,
            doc_weight=doc_weight,
            rerank_vecs=rerank_vecs,
            rerank_model=rerank_model,
            index_paths=index_paths,
            ocr=ocr,
            relations_mat=relations_mat,
            relations_norms=relations_norms,
            relations_manifest=relations_manifest,
            facettes=facettes_module.charger(index_dir),
            profil=meta.get("profil"),
            bm25=bm25,
            grilles=grilles,
            atlas_positions=atlas_positions,
            atlas_mat=atlas_mat,
            atlas_norms=atlas_norms,
            gram_mat=gram_mat,
            gram_norms=gram_norms,
        )

    def _save(self) -> None:
        save_docs(self.index_dir, self.mat, self.norms, self.ids, self.grid)
        if self.facettes:
            facettes_module.sauver(self.index_dir, self.facettes)
        extra_meta: dict[str, object] = {
            "profile_weighting": self.profile_weighting,
            "smoothing_rank": self.smoothing_rank,
            "abtt": self.abtt,
            "ignores": self.ignores,
            "doc_weight": self.doc_weight,
            "index_paths": self.index_paths,
            "ocr": self.ocr,
        }
        if self.profil is not None:
            extra_meta["profil"] = self.profil
        if self.embeddings is not None and self.embed_path is not None:
            extra_meta.update(
                {
                    "embed_path": str(self.embed_path),
                    "embed_sha": self.embeddings.sha,
                    "weights": list(self.weights),
                }
            )
        if self.rerank_vecs is not None:
            assert (
                self.rerank_model is not None
            )  # invariant : jamais de vecteurs sans nom de modèle
            extra_meta["rerank_model"] = self.rerank_model
            save_rerank(self.index_dir, self.rerank_vecs, self.rerank_model)
        if self.bm25 is not None:
            save_bm25(self.index_dir, self.bm25)
        if self.gram_mat is not None:
            assert self.gram_norms is not None
            save_docs(
                self.index_dir,
                self.gram_mat,
                self.gram_norms,
                self.ids,
                self.grid,
                suffixe="_grammatical",
            )
        if self.atlas_positions is not None:
            assert (
                self.atlas_mat is not None and self.atlas_norms is not None
            )  # invariant : jamais de mapping sans cartes
            save_atlas(self.index_dir, self.atlas_positions, atlas_module.COTE)
            save_docs(
                self.index_dir,
                self.atlas_mat,
                self.atlas_norms,
                self.ids,
                (atlas_module.COTE, atlas_module.COTE, 1),
                suffixe="_atlas",
            )
        if self.grilles is not None:
            # meta : la carte des grilles (sens = grille principale, dims EFFECTIVES) ;
            # fichiers : un couple docs_<t>.msei / vocab_<t>.msev par grille extra,
            # formats historiques réutilisés tels quels.
            gt_meta: dict[str, dict] = {"sens": {"dim": int(self.mat.shape[1])}}
            for t, g in self.grilles.items():
                gt_meta[t] = {
                    "dim": g.dim,
                    "poids": list(g.config["poids"]) if g.config.get("poids") else None,
                    "lissage": int(g.config.get("lissage") or 0),
                    "embeddings": bool(g.config.get("embeddings")),
                }
                grid_t = typage_module.grille_de_dim(g.dim)
                save_docs(
                    self.index_dir, g.mat, g.norms, self.ids, grid_t, suffixe=f"_{t}"
                )
                save_vocab(
                    self.index_dir, g.profiles, set(), grid_t, {}, suffixe=f"_{t}"
                )
            extra_meta["grilles_typees"] = gt_meta
        if self.relations_mat is not None:
            assert (
                self.relations_norms is not None
            )  # invariant : jamais de matrice sans normes
            extra_meta["relations"] = True
            save_relations(
                self.index_dir,
                self.relations_mat,
                self.relations_norms,
                self.relations_manifest,
                self.grid,
            )
        save_vocab(
            self.index_dir,
            self.profiles,
            self.colloc,
            self.grid,
            self.lexicon,
            extra_meta=extra_meta,
        )

    def _build_relations(self) -> None:
        """Construit le canal de relations pour TOUS les documents courants (`self.ids`) à
        partir de leur chemin (spec v2.0 §Relations tirées du chemin). Appelé uniquement au
        build (`relations=True`) — jamais un no-op silencieux ensuite : une fois actif,
        `add()` maintient la matrice/manifeste (cf. `add()`)."""
        dim = self.mat.shape[1]
        n = len(self.ids)
        rel_mat = np.zeros((n, dim), dtype=np.int8)
        rel_norms = np.zeros(n, dtype=np.float32)
        manifest: dict[str, set[str]] = {}
        regles = profil_module.roles_du_profil(self.profil)
        for row, doc_id in enumerate(self.ids):
            rels = (
                entities_from_path_profil(doc_id, regles)
                if regles is not None
                else entities_from_path(doc_id)
            )
            for role, entite in rels:
                manifest.setdefault(entite, set()).add(role)
            q, doc_norm = document_channel(rels, dim)
            rel_mat[row] = q
            rel_norms[row] = doc_norm
        self.relations_mat = rel_mat
        self.relations_norms = rel_norms
        self.relations_manifest = manifest

    # -- opérations ---------------------------------------------------------

    def _add_valider(self) -> None:
        """Phase 1 de add() — refus forts AVANT toute mutation de self : un échec
        ici laisse l'index bit-identique à avant l'appel (jamais d'état partiel)."""
        if self.legacy:
            raise ValueError(
                "index legacy v1.0 (sans comptages épars) : add() refusé — "
                "reconstruire l'index via Index.build pour ajouter des documents"
            )
        # Revue finale v1.5 (important, reproduit) : Embeddings.load(..., verify=False) ne
        # calcule jamais de sha (.sha == ""). Un add() ici écrirait embed_sha="" via _save(),
        # rendant l'index inouvrable en mode vérifié (défaut) à la prochaine Index.open().
        if self.embeddings is not None and not self.embeddings.verified:
            raise ValueError(
                "index ouvert sans vérification des embeddings : rouvrir en mode vérifié "
                "avant add()"
            )
        # rerank.msrv présent -> le nouveau document doit être encodé lui aussi.
        if self.rerank_vecs is not None and not rerank_module.available():
            raise ValueError(
                "model2vec non installé — impossible de mettre à jour rerank.msrv "
                '(pip install "model2vec==0.8.2")'
            )

    def _add_flux(
        self, file: Path, text: str
    ) -> tuple[list[str], dict[str, list[str]] | None]:
        """Phase 2 — du texte aux flux de tokens : chemin (file.name seulement —
        jamais le chemin absolu de l'appelant, revue v1.5), canonicalisation,
        collocations, routage typé si l'index l'est."""
        raw_tokens = tokenize(text)
        # Revue finale v1.5 (critique, reproduit) : tokeniser file.as_posix() (le chemin
        # TEL QUE PASSÉ par l'appelant, souvent absolu) injectait les répertoires parents
        # (temp, nom d'utilisateur…) dans le vocabulaire PERSISTÉ de l'index — build()
        # n'a jamais ce problème (doc_id relatif au corpus). add() n'a pas de notion de
        # corpus_dir : le seul chemin cohérent avec ce qui est réellement stocké est
        # file.name, exactement ce que reçoit self.ids.append() plus bas.
        chemin_brut = _path_tokens(file.name) if self.index_paths else []
        if self.grilles is None and self.index_paths:
            raw_tokens = chemin_brut + raw_tokens
        tokens = merge(
            merge(canonicalize(raw_tokens, self._compiled), self.colloc), self.colloc
        )
        flux_add: dict[str, list[str]] | None = None
        if self.grilles is not None:
            # index typé : même routage qu'au build — le chemin va dans SA grille,
            # la grille principale ne reçoit que le flux « sens »
            config_add = typage_module.config_grilles(self.profil)
            flux_add = typage_module.router_flux(
                tokens,
                canonicalize(chemin_brut, self._compiled),
                config_add,
                (self.profil or {}).get("refs"),
            )
            tokens = flux_add["sens"]
        return tokens, flux_add

    def _add_grilles(self, flux_add: dict[str, list[str]]) -> None:
        """Phase 4a — maintient chaque grille typée (learn/finalize/lissage/encode)."""
        for t, g in self.grilles.items():
            toks_t = flux_add.get(t, [])
            g.profiles.learn(toks_t)
            # finalize INCONDITIONNEL — deux raisons architecturales : (1) rows
            # peut être vide alors que la grille sert (réfs sans paires de
            # cooccurrence : la signature porte tout) ; (2) c'est finalize qui
            # réalloue acc et COUPE le memmap du chargement paresseux — le sauter
            # laisse un handle Windows ouvert et _save() échoue (mesuré).
            g.profiles.finalize(self.profile_weighting)
            if g.profiles.rows:
                _apply_smoothing(g.profiles, int(g.config.get("lissage") or 0))
            emb_t = self.embeddings if g.config.get("embeddings") else None
            poids_t = (
                tuple(g.config["poids"]) if g.config.get("poids") else self.weights
            )
            q_t, n_t = encode(toks_t, g.profiles, embeddings=emb_t, weights=poids_t)
            g.mat = np.vstack([g.mat, q_t[np.newaxis, :]])
            g.norms = np.append(g.norms, np.float32(n_t))

    def _add_bm25(
        self, tokens: list[str], flux_add: dict[str, list[str]] | None
    ) -> None:
        """Phase 4b — même flux de tokens que les grilles (invariant du canal : les
        deux voient le même monde, cf. build) — index typé : le flux COMPLET."""
        complet = (
            tokens
            if flux_add is None
            else [t for typ in sorted(flux_add) for t in flux_add[typ]]
        )
        self.bm25.add_doc(complet)

    def _add_grammatical(self, text: str) -> None:
        """Phase 4c — canal grammatical du nouveau document, analysé sur son TEXTE
        brut (extension métier relue depuis le profil persisté)."""
        assert self.gram_norms is not None
        v_g, n_g = grammaire_module.canal_document(
            text,
            self.grid[0] * self.grid[1] * self.grid[2],
            grammaire_module.extension_depuis_profil(self.profil),
        )
        self.gram_mat = np.vstack([self.gram_mat, v_g[None, :]])
        self.gram_norms = np.append(self.gram_norms, np.float32(n_g))

    def _add_atlas(self, tokens: list[str]) -> None:
        """Phase 4d — carte du nouveau document via le mapping FIGÉ au build : la
        SOM n'est pas incrémentale, les tokens inconnus du build ne contribuent
        pas (dérive observable via stats(), re-placée au rebuild nightly)."""
        assert self.atlas_mat is not None and self.atlas_norms is not None
        m = atlas_module.carte(
            tokens, self.profiles.rows, self.atlas_positions, self.profiles.idf
        )
        q_a, n_a = quantize(m)
        self.atlas_mat = np.vstack([self.atlas_mat, q_a[None, :]])
        self.atlas_norms = np.append(self.atlas_norms, np.float32(n_a))

    def _add_relations(self, file: Path) -> None:
        """Phase 4e — add() n'a pas de notion de corpus_dir : file.name n'a jamais
        de segment de dossier -> canal vide (zéros), jamais d'erreur (spec
        §Relations tirées du chemin)."""
        regles = profil_module.roles_du_profil(self.profil)
        rels = (
            entities_from_path_profil(file.name, regles)
            if regles is not None
            else entities_from_path(file.name)
        )
        for role, entite in rels:
            self.relations_manifest.setdefault(entite, set()).add(role)
        assert self.relations_norms is not None  # apparié à relations_mat
        rel_q, rel_norm = document_channel(rels, self.relations_mat.shape[1])
        self.relations_mat = np.vstack([self.relations_mat, rel_q[np.newaxis, :]])
        self.relations_norms = np.append(self.relations_norms, np.float32(rel_norm))

    def add(self, file: Path, ingest_cache_dir: Path | None = None) -> None:
        """Ajoute un document — orchestrateur plat, même contrat que build (invariant
        testé : add ≡ rebuild). Ordre des mutations INCHANGÉ depuis la version
        monolithique (découpage 12/08, déplacements purs) : tout ce qui peut
        échouer est calculé AVANT la première mutation de self."""
        self._add_valider()
        file = Path(file)
        if (
            file.suffix.lower() in ingest.CONVERTIBLE_EXTS
            or file.suffix.lower() in ingest.IMAGE_EXTS
        ):
            text = _read_text_convertible(file, ingest_cache_dir, ocr=self.ocr)
            if text is None:
                self.ignores += 1
                self._save()
                return
        else:
            text = _read_text(file)
        tokens, flux_add = self._add_flux(file, text)

        # Encodage rerank calculé ICI, AVANT toute mutation de self (profiles/mat/norms/ids) :
        # available() ne garantit que l'import, pas qu'un encode() runtime réussisse (poids
        # corrompus, erreur du modèle...). Si ça échoue, self doit rester bit-identique à avant
        # l'appel — jamais mat/ids avancés de 1 pendant que rerank_vecs resterait en retard.
        new_rerank_vec = None
        if self.rerank_vecs is not None:
            new_rerank_vec = _round_f16(rerank_module.encode_texts([text]))

        self.profiles.learn(tokens)
        self.profiles.finalize(self.profile_weighting)
        _apply_smoothing(self.profiles, self.smoothing_rank)
        q, n = encode(
            tokens,
            self.profiles,
            embeddings=self.embeddings,
            weights=self.weights,
            doc_weight=self.doc_weight,
        )
        self._mat32 = None  # le cache chaud ne doit jamais servir un index périmé
        self.mat = np.vstack([self.mat, q[np.newaxis, :]])
        self.norms = np.append(self.norms, np.float32(n))
        self.ids.append(file.name)
        if self.facettes:  # maintenir facettes.json (même clé que self.ids : file.name)
            self.facettes[file.name] = facettes_module.extraire(
                file, file.name, text, profil=self.profil
            )
        if new_rerank_vec is not None:
            assert self.rerank_vecs is not None  # non-None dès que new_rerank_vec l'est
            self.rerank_vecs = np.vstack([self.rerank_vecs, new_rerank_vec])
        if self.grilles is not None:
            assert flux_add is not None  # posé plus haut dès que grilles l'est
            self._add_grilles(flux_add)
        if self.bm25 is not None:
            self._add_bm25(tokens, flux_add)
        if self.gram_mat is not None:
            self._add_grammatical(text)
        if self.atlas_positions is not None:
            self._add_atlas(tokens)
        if self.relations_mat is not None:
            self._add_relations(file)
        self._save()

    def search(
        self,
        text: str,
        k: int = 10,
        rerank: bool = False,
        rerank_lambda: float = 0.70,
        rerank_depth: int = 50,
        type_filtre: str | None = None,
        recence: float = 0.0,
        fusion: bool = False,
        grammatical: bool = False,
    ) -> list[dict]:
        """`fusion=True` : fusion RRF à trois canaux (grille + BM25 + embeddings), validée
        par la mesure — nécessite un index construit avec --hybride (cf. queries.search_fusion).
        Incompatible avec `rerank` (le canal embeddings est DÉJÀ un canal plein de la fusion).

        `type_filtre` / `recence` (facettes, opt-in) : filtre exact par type de document et
        fusion sémantique/fraîcheur — voir mosaic.facettes. Sans facettes.json (index antérieur),
        ces options sont refusées loud (reconstruire l'index les active).

        Boost réf AUTOMATIQUE : si la requête contient un code-like (réf produit, numéro de
        devis/commande — >= 5 caractères mixtes ou >= 6 chiffres), les documents qui portent ce
        code EXACTEMENT prennent la tête (`ref_exacte`). Se désactive silencieusement sans
        facettes.json (rien n'a été demandé explicitement)."""
        if fusion and rerank:
            raise ValueError(
                "fusion et rerank sont exclusifs : la fusion intègre déjà le canal "
                "embeddings comme canal plein"
            )
        if grammatical:
            if self.gram_mat is None:
                raise ValueError(
                    "index sans canal grammatical : reconstruire avec "
                    "`mosaic build --grammatical` pour utiliser --grammatical"
                )
            if fusion or rerank or type_filtre or recence:
                raise ValueError(
                    "--grammatical est exclusif (v1) avec fusion/rerank/type/recence — "
                    "le canal structural se mesure seul avant de se composer"
                )
            return queries.search_grammatical(self, text, k)
        refs_requete = (
            facettes_module.refs_du_texte(text, (self.profil or {}).get("refs"))
            if self.facettes
            else []
        )
        if type_filtre is None and recence == 0.0 and not refs_requete:
            if fusion:
                return queries.search_fusion(self, text, k)
            if self.grilles is not None:
                return queries.search_typee(
                    self, text, k, rerank, rerank_lambda, rerank_depth
                )
            return queries.search(self, text, k, rerank, rerank_lambda, rerank_depth)
        if not self.facettes:
            raise ValueError(
                "index sans facettes.json — reconstruire l'index (mosaic build) pour "
                "utiliser type_filtre/recence"
            )
        # Pool élargi : le filtre par type écarte des candidats, la fusion récence re-trie,
        # le porteur d'une réf exacte peut être hors du top-50 sémantique — tous ont besoin
        # de voir plus loin que k pour rendre k bons résultats. Avec type rare ou réf, le
        # pool doit être PROFOND (5 PDF sur 579 docs entièrement hors top-50, mesuré). Le
        # surcoût est négligeable : le produit matriciel complet domine, indépendant de k.
        profond = bool(type_filtre) or bool(refs_requete)
        pool = min(len(self.ids), max(k * 40, 400)) if profond else max(k * 5, 50)
        if fusion:
            hits = queries.search_fusion(self, text, pool)
        elif self.grilles is not None:
            hits = queries.search_typee(
                self, text, pool, rerank, rerank_lambda, rerank_depth
            )
        else:
            hits = queries.search(self, text, pool, rerank, rerank_lambda, rerank_depth)
        return facettes_module.appliquer(
            hits,
            self.facettes,
            type_filtre=type_filtre,
            recence=recence,
            k=k,
            refs_requete=refs_requete,
        )

    def search_connecteurs(
        self, text: str, k: int = 10, lam: float = 0.7
    ) -> list[dict]:
        return queries.search_connecteurs(self, text, k, lam)

    def chemin(self, doc_id: str, k: int = 10, role: str | None = None) -> list[dict]:
        """Parcours multi-sauts v3 : doc -> ses entités (déliage vectoriel) -> documents
        frères par entité. Voir queries.chemin."""
        return queries.chemin(self, doc_id, k=k, role=role)

    def proches(
        self, mot: str, k: int = 10, dico: bool = False
    ) -> list[tuple[str, float]] | None:
        """Voisinage sémantique métier d'un mot, appris de CE corpus (profils de cooccurrence).
        Exclut les tokens injectés depuis les chemins des documents (noms de fichiers, dates,
        références) — sinon ils polluent le voisinage. None si le mot est absent/sans cooccurrence.

        `dico=True` : ne garde que les vrais mots français, via le vocabulaire de la table potion
        embarquée (déjà chargée avec l'index) — écarte le boilerplate de contenu (collocations,
        codes, dates), au prix des codes-métier hors dictionnaire. Sans effet si l'index n'a pas
        d'embeddings."""
        # Rejoue le MÊME pipeline que le build (canonicalize + fusion de collocations) sur les
        # tokens de chemin, pour retrouver leur forme réelle dans le vocabulaire (« acme corp »
        # -> « acme_corp »), et les exclure du voisinage.
        path_tokens: set[str] = set()
        for doc_id in self.ids:
            merged = merge(
                merge(canonicalize(_path_tokens(doc_id), self._compiled), self.colloc),
                self.colloc,
            )
            path_tokens.update(merged)
        dico_set = (
            frozenset(self.embeddings.rows)
            if dico and self.embeddings is not None
            else None
        )
        return self.profiles.proches(
            mot, k, exclus=frozenset(path_tokens), dico=dico_set
        )

    def search_like(
        self,
        docs: str | Path | list[str | Path],
        k: int = 10,
        rerank: bool = False,
        rerank_lambda: float = 0.70,
        rerank_depth: int = 50,
        ingest_cache_dir: Path | None = None,
    ) -> list[dict]:
        return queries.search_like(
            self, docs, k, rerank, rerank_lambda, rerank_depth, ingest_cache_dir
        )

    def related(self, entite: str, k: int = 10, role: str | None = None) -> list[dict]:
        return queries.related(self, entite, k, role)

    def explain(self, doc_id: str, k: int = 20) -> list[dict]:
        """Décompose le document en ses tokens du vocabulaire les plus proches (démélange)."""
        return queries.explain(self, doc_id, k)

    def explain_match(self, query: str, doc_id: str, k: int = 10) -> list[dict]:
        """Contribution de chaque token de la requête (pipeline complet) au score du document."""
        return queries.explain_match(self, query, doc_id, k)

    def stats(self) -> dict:
        s: dict[str, object] = {
            "docs": len(self.ids),
            "vocab": len(self.profiles.rows),
            "grid": list(self.grid),
            "collocations": len(self.colloc),
            "ignores": self.ignores,
            "index_paths": self.index_paths,
            "ocr": self.ocr,
        }
        # δ (poids du thème) demandé mais pas de table d'embeddings : le canal document est
        # inactif (nécessite l'artefact .msee, cf. mosaic.encoder.doc_channel), le moteur
        # reste pur v1.2 — signalé explicitement plutôt que silencieusement ignoré.
        if self.doc_weight > 0.0 and self.embeddings is None:
            s["canal_document"] = "inactif (pas de table)"
        if self.rerank_vecs is not None:
            s["rerank_model"] = self.rerank_model
        if (
            self.bm25 is not None
        ):  # index hybride : la fusion trois canaux est disponible
            s["hybride"] = True
        if self.gram_mat is not None and self.gram_norms is not None:
            s["grammatical"] = {
                "docs_avec_roles": int((self.gram_norms > 0).sum()),
                "docs": int(len(self.gram_norms)),
            }
        if self.atlas_positions is not None:  # canal atlas (#367)
            s["atlas"] = {
                "cote": atlas_module.COTE,
                "tokens_mappes": int(len(self.atlas_positions)),
                "cellules_occupees": int(len(np.unique(self.atlas_positions))),
            }
        if self.grilles is not None:  # v4 : dims EFFECTIVES par grille (auto visibles)
            s["grilles_typees"] = {
                "sens": int(self.mat.shape[1]),
                **{t: g.dim for t, g in self.grilles.items()},
            }
        if self.relations_mat is not None:
            s["relations"] = True
        if (
            self.profil is not None
        ):  # découverte agent : le paramétrage métier de l'index
            s["profil"] = self.profil
        # v1.5 : Index.open(..., verify_embeddings=False) — table d'embeddings chargée en
        # confiance (memmap, sha non vérifié contre embed_sha). Signalé explicitement plutôt
        # que silencieusement absent, jamais présent quand la table est absente ou vérifiée.
        if self.embeddings is not None and not self.embeddings.verified:
            s["embeddings_verifies"] = False
        return s
