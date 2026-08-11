"""Tests — profil d'index (src/mosaic/profil.py) : validation stricte, application aux trois
canaux (rôles/types/réfs), persistance build->open->add, explication, suggestion."""

import pytest

from mosaic.index import Index
from mosaic.profil import expliquer, suggerer, valider

GRID = (32, 32, 3)

PROFIL_CABINET = {
    "nom": "cabinet",
    "description": "Cabinet : Clients/<CLIENT>/Affaires/<AFF-####>/...",
    "roles": [
        {"role": "affaire", "motif": r"^aff-(\d+)$", "valeur": "{1}"},
        {"role": "client", "motif": r"^[a-z]+$"},
    ],
    "types": {".dwg": "plan"},
    "refs": {"min_mixte": 4, "min_chiffres": 4},
}


def test_validation_stricte():
    valider(PROFIL_CABINET)  # profil sain : passe
    with pytest.raises(ValueError, match="clés inconnues"):
        valider({"role": []})  # faute de frappe (role vs roles) -> loud
    with pytest.raises(ValueError, match="regex invalide"):
        valider({"roles": [{"role": "x", "motif": "["}]})
    with pytest.raises(ValueError, match="min_mixte"):
        valider({"refs": {"min_mixte": 0}})
    with pytest.raises(ValueError, match="extension"):
        valider({"types": {"dwg": "plan"}})  # extension sans point -> loud


def test_profil_roles_pilotent_le_graphe(tmp_path):
    """Les rôles déclarés remplacent dossier/annee/mois : un chemin Clients/durand/AFF-042
    produit des entités client/affaire — et `chemin` traverse avec CES rôles."""
    c = tmp_path / "corpus"
    for chemin, texte in [
        ("durand/aff-042/note_audience.md", "audience tribunal durand"),
        ("durand/aff-042/note_conclusions.md", "conclusions ecrites durand"),
        ("durand/aff-051/note_bail.md", "bail commercial durand"),
        ("lemaire/aff-042/note_autre.md", "affaire homonyme lemaire"),
    ]:
        f = c / chemin
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(texte, encoding="utf-8")
    idx = Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        index_paths=False,
        smoothing_rank=0,
        relations=True,
        profil=PROFIL_CABINET,
    )
    groupes = idx.chemin("durand/aff-042/note_audience.md", k=5)
    par_cle = {(g["role"], g["entite"]): g for g in groupes}
    # rôle « client » : les autres notes de durand (dont l'affaire 051)
    ids_client = {d["id"] for d in par_cle[("client", "durand")]["documents"]}
    assert "durand/aff-051/note_bail.md" in ids_client
    assert "lemaire/aff-042/note_autre.md" not in ids_client
    # rôle « affaire » (valeur = groupe capturé « 042 ») : l'homonyme de lemaire est dedans
    ids_affaire = {d["id"] for d in par_cle[("affaire", "042")]["documents"]}
    assert "durand/aff-042/note_conclusions.md" in ids_affaire
    assert "lemaire/aff-042/note_autre.md" in ids_affaire


def test_profil_types_et_refs_custom(tmp_path):
    """Types custom (extension inconnue -> libellé métier) et critère de réfs abaissé
    (min 4) : SKU courts reconnus, et le profil est relu à la RECHERCHE (boost)."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "catalogue.md").write_text("article sku A123 stock douze", encoding="utf-8")
    (c / "notes.md").write_text(
        "inventaire general stock articles rayonnage", encoding="utf-8"
    )
    profil = {"nom": "boutique", "refs": {"min_mixte": 4, "min_chiffres": 4}}
    idx = Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        index_paths=False,
        smoothing_rank=0,
        profil=profil,
    )
    # A123 (4 mixtes) est une réf avec ce profil (refusée par le défaut min 5)
    r = idx.search("stock A123", k=2)
    assert r[0]["id"] == "catalogue.md"
    assert r[0].get("ref_exacte") == ["A123"]


def test_profil_persiste_build_open_add(tmp_path):
    """Règle 1 : le profil survit au cycle build->open, et add() l'applique (jamais de
    divergence silencieuse entre build et add)."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "doc.md").write_text("premier document", encoding="utf-8")
    profil = {"nom": "boutique", "refs": {"min_mixte": 4, "min_chiffres": 4}}
    Index.build(
        c,
        tmp_path / "idx",
        grid=GRID,
        index_paths=False,
        smoothing_rank=0,
        profil=profil,
    )
    idx = Index.open(tmp_path / "idx")
    assert idx.profil is not None and idx.profil["nom"] == "boutique"
    assert idx.stats()["profil"]["nom"] == "boutique"  # découverte agent
    nouveau = tmp_path / "ajout.md"
    nouveau.write_text("ajout sku B777 stock", encoding="utf-8")
    idx.add(nouveau)
    assert (
        "B777" in idx.facettes["ajout.md"]["refs"]
    )  # add applique le critère du profil


def test_expliquer_mode_humain():
    txt = expliquer(PROFIL_CABINET)
    assert "cabinet" in txt
    assert "affaire" in txt and "client" in txt  # chaque règle racontée
    assert "plan" in txt  # le type custom
    assert "défaut" in expliquer(None)  # sans profil : les défauts sont racontés aussi


def test_expliquer_version_anglaise():
    """La version anglaise raconte les mêmes règles (repo public : le mode humain n'est pas
    enfermé dans le français). Les types canoniques restent des identifiants, glosés en en."""
    txt = expliquer(PROFIL_CABINET, langue="en")
    assert "Profile" in txt and "entity" in txt  # racontée en anglais
    assert "affaire" in txt and "client" in txt  # les MÊMES règles
    assert "Effect" in txt
    defauts = expliquer(None, langue="en")
    assert "spreadsheet" in defauts  # le type canonique « tableur » est glosé
    with pytest.raises(ValueError, match="langue"):
        expliquer(None, langue="de")


def test_suggerer_calibration(tmp_path):
    """Mode agent : le scan d'un corpus arborescent daté propose année + attrape-tout, et
    liste les extensions inconnues à mapper — un profil VALIDE prêt à ajuster."""
    c = tmp_path / "corpus"
    (c / "CHANTIER_A/2026").mkdir(parents=True)
    (c / "CHANTIER_A/2026/note.md").write_text("x", encoding="utf-8")
    (c / "CHANTIER_A/plan.dwg").write_bytes(b"\x00")
    suggestion = suggerer(c)
    roles = {r["role"] for r in suggestion["roles"]}
    assert "annee" in roles and "dossier" in roles
    assert suggestion["types"][".dwg"] == "?"  # extension inconnue signalée à mapper
    # la suggestion est un profil valide (une fois les « ? » remplacés)
    suggestion["types"][".dwg"] = "plan"
    valider(suggestion)
