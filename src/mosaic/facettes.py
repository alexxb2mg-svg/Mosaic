"""Facettes de la carte d'identité — métadonnées EXACTES par document, hors grille.

La grille porte le sens (matching flou) ; les facettes portent les attributs exacts qu'un vecteur
représente mal : le TYPE de document (tableur, pdf scanné/numérique, photo…) et la DATE (lue dans
le nom de fichier). Extraites à l'indexation (gratuit, déterministe), persistées dans
`facettes.json`, appliquées à la recherche comme filtre (type) et comme signal de fusion
(récence, fusion de rangs façon RRF — le motif validé de la méta-recherche).

Complémentaires, jamais concurrentes : le sens présélectionne, la facette précise. Un index sans
`facettes.json` (antérieur) dégrade proprement : filtre/récence indisponibles, erreur explicite.
"""

import json
import re
from pathlib import Path

from mosaic import exif
from mosaic.ingest import type_semantique
from mosaic.temporel import SANS_DATE, date_du_chemin

FICHIER = "facettes.json"
K_RRF = 60  # constante de fusion de rangs (même valeur que la méta-recherche)

# Réf/code PROBABLE : >= 5 caractères MIXTES lettres+chiffres (MFN710, D26050018) ou >= 6
# chiffres purs (086027, 4050499010). Volontairement strict — « 16A » (calibre) ou « 2026 »
# (année) ne sont PAS des réfs : un faux positif détournerait le boost exact.
_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9./-]*")
_REFS_MAX = (
    300  # plafond par document (un catalogue entier ne doit pas gonfler le JSON)
)


def _est_ref(tok: str, regles: dict | None = None) -> bool:
    n = re.sub(r"[^A-Za-z0-9]", "", tok)
    if regles and "motif" in regles:  # motif custom du profil : il REMPLACE le critère
        return bool(re.fullmatch(regles["motif"], n))
    min_mixte = (regles or {}).get("min_mixte", 5)
    min_chiffres = (regles or {}).get("min_chiffres", 6)
    if len(n) >= min_mixte and re.search(r"[A-Za-z]", n) and re.search(r"[0-9]", n):
        return True
    return len(n) >= min_chiffres and n.isdigit()


def refs_du_texte(texte: str, regles: dict | None = None) -> list[str]:
    """Tokens code-like d'un texte (réfs produit, numéros de devis/commande, EAN…), normalisés
    MAJUSCULES sans séparateurs, dédupliqués, ordre d'apparition, plafonnés à _REFS_MAX.
    `regles` (profil d'index, clé `refs`) adapte le critère au métier."""
    vus: dict[str, None] = {}
    for tok in _REF.findall(texte):
        if _est_ref(tok, regles):
            vus.setdefault(re.sub(r"[^A-Za-z0-9]", "", tok).upper(), None)
            if len(vus) >= _REFS_MAX:
                break
    return list(vus)


def extraire(
    path: Path, doc_id: str, texte: str | None, profil: dict | None = None
) -> dict:
    """Les facettes d'un document : type sémantique + date du chemin + réfs code-like. Étendre
    ici quand une nouvelle facette universelle apparaît (persistée et applicable sans autre
    geste). Les réfs viennent du texte ET du nom de fichier (un devis porte sa réf en nom).
    `profil` (optionnel) adapte types custom et critère de réfs à l'environnement."""
    profil = profil or {}
    refs = refs_du_texte(f"{doc_id} {texte or ''}", profil.get("refs"))
    # Le chemin fait foi quand il porte une date : c'est un classement VOULU. L'EXIF
    # ne prend le relais que sur les documents autrement sans date — typiquement une
    # photo de chantier (`IMG_2043.jpg`), qui tombait sinon à « 0000-00-00 » et
    # restait hors de portée de `--recence` et de `mosaic actuel`. Mesuré sur le
    # corpus chantiers : 183 des 1 009 JPEG récupèrent ainsi une date exacte, et le
    # parseur maison est équivalent à Pillow sur ce corpus (183 accords, 0 écart).
    date = date_du_chemin(doc_id)
    if date == SANS_DATE:
        date = exif.date_de_prise_de_vue(path) or SANS_DATE
    fac: dict = {
        "type": type_semantique(path, texte, profil.get("types")),
        "date": date,
    }
    if refs:
        fac["refs"] = refs
    return fac


def sauver(index_dir: Path, facettes: dict[str, dict[str, str]]) -> None:
    """Écriture atomique de facettes.json (doc_id -> {type, date}), triée (reproductible)."""
    blob = json.dumps(facettes, ensure_ascii=False, sort_keys=True, indent=0)
    tmp = Path(index_dir) / (FICHIER + ".tmp")
    tmp.write_text(blob, encoding="utf-8")
    tmp.replace(Path(index_dir) / FICHIER)


def charger(index_dir: Path) -> dict[str, dict[str, str]] | None:
    """None si facettes.json absent (index antérieur — dégradation propre, jamais un crash)."""
    f = Path(index_dir) / FICHIER
    if not f.is_file():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def appliquer(
    hits: list[dict],
    facettes: dict[str, dict],
    type_filtre: str | None = None,
    recence: float = 0.0,
    k: int = 10,
    refs_requete: list[str] | None = None,
) -> list[dict]:
    """Applique les facettes à une liste de résultats sémantiques (déjà triés par score).

    - `type_filtre` : ne garde que les documents de ce type (filtre EXACT — « je veux le
      tableur ») ; chaque résultat gagne sa facette `type` pour l'explicabilité.
    - `refs_requete` : réfs code-like détectées dans la requête — les documents qui portent une
      de ces réfs EXACTEMENT prennent la tête (champ `ref_exacte` pour l'explicabilité), triés
      entre eux par score sémantique. Un match exact de code est ce que l'utilisateur veut ;
      le flou sémantique n'a pas à le noyer.
    - `recence` ∈ [0, 1] : poids de la fraîcheur. 0 = ordre sémantique pur ; sinon fusion de
      rangs (1-p)/(K+rang_sémantique) + p/(K+rang_date) — invariante d'échelle, le motif RRF.
      Les documents sans date (« 0000-00-00 ») tombent naturellement en queue du rang date.
      Appliquée après la partition réf (un match exact reste devant).
    """
    if not 0.0 <= recence <= 1.0:
        raise ValueError(f"recence doit être dans [0, 1] : {recence!r}")
    cherchees = set(refs_requete or [])
    porteurs: list[dict] = []
    enrichis: list[dict] = []
    for h in hits:
        fac = facettes.get(h["id"], {})
        h = {**h, "type": fac.get("type", "?"), "date": fac.get("date", SANS_DATE)}
        if type_filtre is not None and h["type"] != type_filtre:
            continue
        communes = cherchees & set(fac.get("refs", ())) if cherchees else set()
        if communes:
            h["ref_exacte"] = sorted(communes)
            porteurs.append(h)
        else:
            enrichis.append(h)
    if recence > 0.0 and enrichis:
        rang_date = {
            h["id"]: r
            for r, h in enumerate(
                sorted(enrichis, key=lambda x: str(x["date"]), reverse=True), start=1
            )
        }
        fusion: dict[str, float] = {}
        for rang_sem, h in enumerate(enrichis, start=1):
            fusion[h["id"]] = (1.0 - recence) / (K_RRF + rang_sem) + recence / (
                K_RRF + rang_date[h["id"]]
            )
            h["score_fusion"] = round(fusion[h["id"]], 6)
        enrichis.sort(key=lambda x: -fusion[x["id"]])
    # partition finale : les matchs exacts de réf devant (triés par sémantique entre eux),
    # le reste derrière (ordre sémantique ou fusionné récence).
    return (porteurs + enrichis)[:k]
