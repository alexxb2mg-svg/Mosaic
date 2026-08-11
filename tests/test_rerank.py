"""Tests du module mosaic.rerank : import sous garde (model2vec optionnel), encodage."""

import numpy as np
import pytest

from mosaic import rerank


def test_available_reflects_model2vec_import(monkeypatch):
    monkeypatch.setattr(rerank, "StaticModel", None)
    assert rerank.available() is False


def test_available_true_quand_model2vec_installe():
    pytest.importorskip("model2vec")
    assert rerank.available() is True


def test_encode_texts_sans_model2vec_leve_runtimeerror(monkeypatch):
    monkeypatch.setattr(rerank, "StaticModel", None)
    monkeypatch.setattr(rerank, "_model", None)
    with pytest.raises(RuntimeError):
        rerank.encode_texts(["bonjour"])


def test_encode_texts_vecteurs_unitaires(monkeypatch):
    class _FakeModel:
        def encode(self, texts):
            rng = np.random.default_rng(1)
            return rng.normal(size=(len(texts), rerank.DIM)).astype(np.float32)

    monkeypatch.setattr(rerank, "_get_model", lambda: _FakeModel())
    vecs = rerank.encode_texts(["a", "b", "c"])
    assert vecs.shape == (3, rerank.DIM)
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_encode_texts_vide_ne_charge_pas_le_modele(monkeypatch):
    def _boom():
        raise AssertionError("le modèle ne doit pas être chargé pour une liste vide")

    monkeypatch.setattr(rerank, "_get_model", _boom)
    vecs = rerank.encode_texts([])
    assert vecs.shape == (0, rerank.DIM)


def test_encode_query_vecteur_unitaire_dim_correcte(monkeypatch):
    class _FakeModel:
        def encode(self, texts):
            rng = np.random.default_rng(2)
            return rng.normal(size=(len(texts), rerank.DIM)).astype(np.float32)

    monkeypatch.setattr(rerank, "_get_model", lambda: _FakeModel())
    q = rerank.encode_query("interrupteur différentiel")
    assert q.shape == (rerank.DIM,)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-5


def test_encode_texts_deterministe_meme_modele(monkeypatch):
    """Un même texte encodé deux fois via le même modèle donne le même vecteur."""

    class _FakeModel:
        def encode(self, texts):
            out = np.zeros((len(texts), rerank.DIM), dtype=np.float32)
            for i, t in enumerate(texts):
                out[i] = np.random.default_rng(hash(t) & 0xFFFFFFFF).normal(
                    size=rerank.DIM
                )
            return out

    monkeypatch.setattr(rerank, "_get_model", lambda: _FakeModel())
    v1 = rerank.encode_texts(["cablage tableau"])[0]
    v2 = rerank.encode_texts(["cablage tableau"])[0]
    assert np.array_equal(v1, v2)


def test_encode_texts_modele_reel():
    """Intégration réelle : potion-multilingual-128M est en cache local (env de dev)."""
    pytest.importorskip("model2vec")
    vecs = rerank.encode_texts(["interrupteur différentiel", "carrelage cuisine"])
    assert vecs.shape == (2, rerank.DIM)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)


# -- v1.5 : répertoire local du modèle (bypass des vérifications hub) ----------------------


def test_model_source_utilise_le_repertoire_local_si_present(tmp_path, monkeypatch):
    local_dir = tmp_path / "potion_model"
    local_dir.mkdir()
    monkeypatch.setenv("MOSAIC_POTION_MODEL_DIR", str(local_dir))
    assert rerank._model_source() == str(local_dir)


def test_model_source_repli_sur_le_nom_hub_si_repertoire_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("MOSAIC_POTION_MODEL_DIR", str(tmp_path / "n_existe_pas"))
    assert rerank._model_source() == rerank.MODEL_NAME


def test_model_source_defaut_sans_variable_env(monkeypatch, tmp_path):
    """Bug revue finale (mineur) : l'assertion d'origine (`in (MODEL_NAME,
    _LOCAL_MODEL_DIR_DEFAULT)`) était tautologique — vraie quelle que soit la branche prise,
    puisque ce sont les deux SEULES valeurs que la fonction peut jamais retourner. Le cwd est
    fixé sur un répertoire vide (sans `data_externes/potion_model/`) pour rendre le défaut
    déterministe et vérifier la VRAIE valeur attendue, indépendamment de ce qui traîne sur le
    poste de dev (répertoire local généré par `prepare_potion.py --save-model`, gitignored)."""
    monkeypatch.delenv("MOSAIC_POTION_MODEL_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert rerank._model_source() == rerank.MODEL_NAME


def test_get_model_charge_depuis_le_repertoire_local(monkeypatch, tmp_path):
    local_dir = tmp_path / "potion_model"
    local_dir.mkdir()
    monkeypatch.setenv("MOSAIC_POTION_MODEL_DIR", str(local_dir))
    monkeypatch.setattr(rerank, "_model", None)
    captured = {}

    class _FakeStaticModel:
        @staticmethod
        def from_pretrained(path):
            captured["path"] = path
            return object()

    monkeypatch.setattr(rerank, "StaticModel", _FakeStaticModel)
    rerank._get_model()
    assert captured["path"] == str(local_dir)


def test_get_model_repli_sur_le_hub_sans_repertoire_local(monkeypatch, tmp_path):
    monkeypatch.setenv("MOSAIC_POTION_MODEL_DIR", str(tmp_path / "n_existe_pas"))
    monkeypatch.setattr(rerank, "_model", None)
    captured = {}

    class _FakeStaticModel:
        @staticmethod
        def from_pretrained(path):
            captured["path"] = path
            return object()

    monkeypatch.setattr(rerank, "StaticModel", _FakeStaticModel)
    rerank._get_model()
    assert captured["path"] == rerank.MODEL_NAME
