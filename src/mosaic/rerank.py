"""Le repêcheur : reclassement du top-`depth` par similarité document model2vec (potion).

model2vec est une dépendance OPTIONNELLE (`pip install model2vec==0.8.2`) : sans lui, Mosaic
continue de fonctionner à l'identique — `--rerank-vectors` (build) et `--rerank` (search)
échouent avec un message clair, jamais un crash silencieux. Import sous garde try/except,
même principe que `mosaic.ingest` pour markitdown : c'est le seul autre module du cœur
Mosaic à connaître ce paquet.

Le TEXTE des documents n'est jamais persisté (principe du prisme) — seule son empreinte
model2vec (vecteur unitaire 256d) l'est, dans le fichier annexe `rerank.msrv`
(cf. mosaic.store.save_rerank/load_rerank).
"""

import os
from pathlib import Path

import numpy as np

try:
    from model2vec import StaticModel  # ty: ignore[unresolved-import]
except ImportError:
    StaticModel = None  # ty: ignore[invalid-assignment]  # garde dépendance optionnelle

MODEL_NAME = "minishlab/potion-multilingual-128M"
DIM = 256

# v1.5 : répertoire local du modèle (généré par `scripts/prepare_potion.py --save-model`).
# model2vec.persistence._resolve_folder retourne un chemin qui `.exists()` TEL QUEL, sans
# jamais appeler huggingface_hub (ni snapshot_download, ni vérification d'etag/metadata) —
# c'est ce bypass complet qui coupe le coût mesuré (~4.4 s : 500 Mo safetensors + vérifs hub)
# même quand le modèle est déjà en cache HF (le cache hub, lui, est toujours revérifié en
# ligne par from_pretrained(MODEL_NAME) puisque force_download vaut True par défaut côté
# model2vec). Chemin RELATIF au CWD (même convention que cli._WIKDICT_LEXICON_EXTERNAL) :
# usage documenté = lancer mosaic depuis la racine du dépôt.
_LOCAL_MODEL_DIR_ENV = "MOSAIC_POTION_MODEL_DIR"
_LOCAL_MODEL_DIR_DEFAULT = "data_externes/potion_model"

_model = None  # singleton paresseux, construit une seule fois par process


def available() -> bool:
    """True si model2vec est importable (`pip install model2vec==0.8.2`)."""
    return StaticModel is not None


def _model_source() -> str:
    """Résout la source à passer à `StaticModel.from_pretrained` : le répertoire local si
    présent (chargement local, aucun appel réseau/hub), sinon le nom du modèle HF (chargé
    via le cache hub, avec vérification hub à chaque appel)."""
    local = os.environ.get(_LOCAL_MODEL_DIR_ENV, _LOCAL_MODEL_DIR_DEFAULT)
    if Path(local).is_dir():
        return local
    return MODEL_NAME


def model_name_effectif() -> str:
    """Le nom du modèle RÉELLEMENT chargé — celui que les métadonnées doivent porter.

    Bug débusqué en mesure (12/08) : build écrivait la constante MODEL_NAME dans
    rerank.msrv même quand MOSAIC_POTION_MODEL_DIR chargeait un AUTRE modèle — un
    index construit avec un modèle 512d se déclarait « potion-multilingual-128M ».
    Une métadonnée qui ment est pire qu'une métadonnée absente (« déterministe ou
    explicite »). Règle : le répertoire local PAR DÉFAUT (data_externes/potion_model)
    est par contrat un miroir du modèle canonique — seule la SURCHARGE env signale un
    modèle différent."""
    env = os.environ.get(_LOCAL_MODEL_DIR_ENV)
    if env and Path(env).is_dir():
        return f"local:{Path(env).name}"
    return MODEL_NAME


def _get_model():
    global _model
    if _model is None:
        if StaticModel is None:
            raise RuntimeError(
                "model2vec non installé — appeler available() avant _get_model()"
            )
        _model = StaticModel.from_pretrained(_model_source())
    return _model


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def encode_texts(texts: list[str]) -> np.ndarray:
    """Encode une liste de textes en vecteurs UNITAIRES (n, DIM) float32.

    Normalisés ici (une fois) pour que la similarité cosinus au moment du rerank se
    réduise à un simple produit scalaire — même convention que
    mosaic.embeddings.Embeddings.vector_grid. Liste vide -> (0, DIM), le modèle n'est
    même pas chargé (jamais de coût pour un batch vide).
    """
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    vecs = _get_model().encode(texts).astype(np.float32)
    return _normalize_rows(vecs)


def encode_query(text: str) -> np.ndarray:
    """Encode une requête en vecteur unitaire (DIM,) float32."""
    return encode_texts([text])[0]
