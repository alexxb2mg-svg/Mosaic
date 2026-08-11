"""Index Mosaic : construction, recherche, ajout incrémental."""

from pathlib import Path

import numpy as np

from mosaic import GRID_DEFAULT, ingest, queries
from mosaic import rerank as rerank_module
from mosaic import facettes as facettes_module
from mosaic import profil as profil_module
from mosaic.bm25 import Bm25
from mosaic.collocations import detect, merge
from mosaic.docio import _EXTS, _path_tokens, _read_text, _read_text_convertible
from mosaic.embeddings import Embeddings
from mosaic.encoder import WEIGHTS_DEFAULT, encode
from mosaic.lexicon import canonicalize, compile_lexicon, load_lexicon
from mosaic.profiles import Profiles
from mosaic.relations import (
    document_channel,
    entities_from_path,
    entities_from_path_profil,
)
from mosaic.smoothing import smooth
from mosaic.store import (
    load_bm25,
    load_docs,
    load_relations,
    load_rerank,
    load_vocab,
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
    profiles.acc[:v] = smooth(profiles.acc[:v], rank)


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
    ) -> "Index":
        corpus_dir, index_dir = Path(corpus_dir), Path(index_dir)
        if not corpus_dir.is_dir():
            raise ValueError(f"corpus introuvable : {corpus_dir}")
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
        dim = grid[0] * grid[1] * grid[2]
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
        rerank_texts: list[str] = []
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
            tokens = (
                _path_tokens(doc_id) + content_tokens if index_paths else content_tokens
            )
            facettes[doc_id] = facettes_module.extraire(p, doc_id, text, profil=profil)
            if type_doc:  # facette « type de document » aussi DANS la grille (opt-in)
                tokens = tokenize(facettes[doc_id]["type"]) + tokens
            raw.append((doc_id, tokens))
            if rerank_vectors:
                rerank_texts.append(text)
        if lexicon is None:
            lexicon = load_lexicon()
        compiled = compile_lexicon(lexicon)
        canon = [(doc_id, canonicalize(tokens, compiled)) for doc_id, tokens in raw]
        colloc = detect([t for _, t in canon])
        docs = [
            (doc_id, merge(merge(tokens, colloc), colloc)) for doc_id, tokens in canon
        ]
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
        rerank_vecs = None
        rerank_model = None
        if rerank_vectors:
            # Arrondi float16 immédiat : rerank.msrv persiste en float16 (cf. save_rerank),
            # donc un search() juste après build() doit voir exactement la même précision
            # qu'un search() après Index.open() — jamais deux comportements pour le même index.
            rerank_vecs = _round_f16(rerank_module.encode_texts(rerank_texts))
            rerank_model = rerank_module.MODEL_NAME
        # Postings BM25 sur le MÊME flux de tokens que la grille (canonical + collocations,
        # chemins selon index_paths) : les deux canaux de la fusion voient le même monde.
        bm25 = Bm25.from_docs(docs) if hybride else None
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

    def add(self, file: Path, ingest_cache_dir: Path | None = None) -> None:
        if self.legacy:
            raise ValueError(
                "index legacy v1.0 (sans comptages épars) : add() refusé — "
                "reconstruire l'index via Index.build pour ajouter des documents"
            )
        # Revue finale v1.5 (important, reproduit) : Embeddings.load(..., verify=False) ne
        # calcule jamais de sha (.sha == ""). Un add() ici écrirait embed_sha="" via _save(),
        # rendant l'index inouvrable en mode vérifié (défaut) à la prochaine Index.open().
        # Refus net AVANT toute mutation de self, comme les autres garde-fous de add().
        if self.embeddings is not None and not self.embeddings.verified:
            raise ValueError(
                "index ouvert sans vérification des embeddings : rouvrir en mode vérifié "
                "avant add()"
            )
        file = Path(file)
        # rerank.msrv présent -> le nouveau document doit être encodé lui aussi. Vérifié
        # AVANT toute mutation de self (mat/norms/ids/profiles) : un échec ici laisse
        # l'index inchangé plutôt qu'à moitié mis à jour (fail-fast, jamais d'état partiel).
        if self.rerank_vecs is not None and not rerank_module.available():
            raise ValueError(
                "model2vec non installé — impossible de mettre à jour rerank.msrv "
                '(pip install "model2vec==0.8.2")'
            )
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
        raw_tokens = tokenize(text)
        if self.index_paths:
            # Revue finale v1.5 (critique, reproduit) : tokeniser file.as_posix() (le chemin
            # TEL QUE PASSÉ par l'appelant, souvent absolu) injectait les répertoires parents
            # (temp, nom d'utilisateur…) dans le vocabulaire PERSISTÉ de l'index — build()
            # n'a jamais ce problème (doc_id relatif au corpus). add() n'a pas de notion de
            # corpus_dir : le seul chemin cohérent avec ce qui est réellement stocké est
            # file.name, exactement ce que reçoit self.ids.append() plus bas.
            raw_tokens = _path_tokens(file.name) + raw_tokens
        tokens = merge(
            merge(canonicalize(raw_tokens, self._compiled), self.colloc), self.colloc
        )

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
        if self.bm25 is not None:
            # même flux de tokens que la ligne ajoutée à la matrice grille (invariant du
            # canal : les deux voient le même monde, cf. build)
            self.bm25.add_doc(tokens)
        if self.relations_mat is not None:
            # add() n'a pas de notion de corpus_dir (cf. commentaire index_paths plus haut) :
            # entities_from_path(file.name) n'a donc jamais de segment de dossier -> canal
            # vide (zéros), jamais d'erreur — comportement documenté (spec §Relations tirées
            # du chemin, « un doc sans relation extractible -> canal vide »).
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
        refs_requete = (
            facettes_module.refs_du_texte(text, (self.profil or {}).get("refs"))
            if self.facettes
            else []
        )
        if type_filtre is None and recence == 0.0 and not refs_requete:
            if fusion:
                return queries.search_fusion(self, text, k)
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
        hits = (
            queries.search_fusion(self, text, pool)
            if fusion
            else queries.search(self, text, pool, rerank, rerank_lambda, rerank_depth)
        )
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
