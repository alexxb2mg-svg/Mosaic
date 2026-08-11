"""Canal de relations (v2.0) : liaison hyperdimensionnelle par permutation circulaire
déterministe, et extraction gratuite de relations depuis l'arborescence du corpus.

`bind(role, entite, dim) = np.roll(signature(entite, dim), decalage)` où le décalage est
un hash déterministe du rôle — même famille d'opération que les signatures ternaires
(cf. `mosaic.signatures`), réservée dans l'en-tête du format depuis la v1 (docs.msei).
Capacité mesurée non limitante aux échelles testées (80+ relations/document).
"""

import hashlib
import re

import numpy as np

from mosaic.encoder import quantize
from mosaic.signatures import signature
from mosaic.tokenize import tokenize

# Préfixe numérique d'ordre d'un segment de dossier : "03_", "08-", "08." — un ou
# plusieurs chiffres suivis d'au moins un séparateur `-_.`.
_ORDER_PREFIX_RE = re.compile(r"^\d+[-_.]+")
_YEAR_RE = re.compile(r"^20\d\d$")
_MONTH_LABEL_RE = re.compile(r"^(0[1-9]|1[0-2])-(.+)$")
_MONTH_YEAR_RE = re.compile(r"^(0[1-9]|1[0-2])\.(\d{4})$")


def decalage(role: str, dim: int) -> int:
    """Décalage de permutation déterministe par rôle, dans [1, dim)."""
    h = hashlib.sha256(("role:" + role).encode("utf-8")).digest()[:4]
    return int.from_bytes(h, "big") % (dim - 1) + 1


def bind(role: str, entite: str, dim: int) -> np.ndarray:
    """Liaison hyperdimensionnelle : la signature ternaire de l'entité, permutée
    circulairement d'un décalage déterministe par rôle. int32 (comme signature())."""
    return np.roll(signature(entite, dim), decalage(role, dim))


def document_channel(
    relations: list[tuple[str, str]], dim: int
) -> tuple[np.ndarray, float]:
    """Canal de relations d'un document = normaliser(Σ bind(role, entite)) sur ses
    relations, quantifié int8 par le même schéma pic-127 que la mosaïque
    (`mosaic.encoder.quantize`). Aucune relation -> canal vide (zéros, norme 0),
    jamais d'erreur."""
    if not relations:
        return np.zeros(dim, dtype=np.int8), 0.0
    acc = np.zeros(dim, dtype=np.float32)
    for role, entite in relations:
        acc += bind(role, entite, dim).astype(np.float32)
    return quantize(acc)


def entites_du_canal(
    canal: np.ndarray,
    manifest: dict[str, set[str]],
    dim: int,
    seuil: float = 0.15,
) -> list[tuple[str, str, float]]:
    """DÉLIAGE VECTORIEL du canal d'un document : retrouve ses (rôle, entité) sans lire son
    chemin — le saut 1 du parcours multi-sauts, fait entièrement dans les valeurs.

    Pour chaque rôle du manifeste : dé-permute le canal (`roll` inverse du décalage du rôle)
    puis CLEANUP contre les signatures des entités connues pour ce rôle (produit scalaire +
    seuil de cosinus). C'est la recette validée par la recherche (multisauts_valeurs.py) :
    un nettoyage vectoriel à chaque saut, jamais de chaînage brut. Rend [(role, entite, cos)]
    trié par cosinus décroissant ; canal vide -> []."""
    norme = float(np.linalg.norm(canal.astype(np.float32)))
    if norme == 0.0 or not manifest:
        return []
    roles: dict[str, list[str]] = {}
    for entite, rs in manifest.items():
        for r in rs:
            roles.setdefault(r, []).append(entite)
    out: list[tuple[str, str, float]] = []
    for role, entites in roles.items():
        depermute = np.roll(canal.astype(np.float32), -decalage(role, dim))
        codebook = np.stack([signature(e, dim).astype(np.float32) for e in entites])
        normes_sig = np.linalg.norm(codebook, axis=1)
        cos = (codebook @ depermute) / (normes_sig * norme)
        for i in np.argsort(-cos):
            if float(cos[i]) < seuil:
                break
            out.append((role, entites[int(i)], round(float(cos[i]), 4)))
    out.sort(key=lambda x: -x[2])
    return out


def normalize_entity(raw: str) -> str:
    """Normalisation d'une entité (dossier, valeur libre) : minuscules, tokenisation
    (accents conservés, `mosaic.tokenize`), tokens rejoints par `_` — un segment comme
    "ATLAS_NORD" ou une requête `related()` en clair normalisent au même point fixe,
    exactement ce qu'exige la spec (« entité normalisée comme au build »)."""
    return "_".join(tokenize(raw.lower()))


def _normalize_segment(raw_segment: str) -> str:
    """Normalise un segment de CHEMIN (dossier) : retire un préfixe numérique d'ordre
    (`03_`, `08-`, `08.`) puis normalise le libellé restant (`normalize_entity`)."""
    stripped = _ORDER_PREFIX_RE.sub("", raw_segment.lower(), count=1)
    return normalize_entity(stripped)


def entities_from_path_profil(doc_id: str, regles) -> list[tuple[str, str]]:
    """Extraction PILOTÉE PAR PROFIL (déclaratif pur) : pour chaque segment de dossier, la
    PREMIÈRE règle (motif_compilé, role, valeur_template) qui correspond gagne. `valeur` est
    un gabarit sur les groupes du motif (« {2}-{1} ») ; sans gabarit, le segment normalisé
    (préfixe d'ordre retiré). Dédupliqué, ordre d'apparition — même contrat que la logique
    historique, mais chaque règle est déclarée et visible (mosaic profil --explique)."""
    relations: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_seg in doc_id.split("/")[:-1]:
        for motif, role, valeur in regles:
            m = motif.match(raw_seg.lower())
            if not m:
                continue
            if valeur:
                entite = valeur
                for gi, gv in enumerate(m.groups(), start=1):
                    entite = entite.replace("{%d}" % gi, gv or "")
            else:
                entite = _normalize_segment(raw_seg)
            key = (role, entite)
            if entite and key not in seen:
                seen.add(key)
                relations.append(key)
            break  # première règle gagnante : on passe au segment suivant
    return relations


def entities_from_path(doc_id: str) -> list[tuple[str, str]]:
    """Relations tirées des segments de dossier d'un id (spec v2.0 §Relations tirées du
    chemin) — le dernier segment (nom de fichier) est ignoré. Chaque segment produit
    `(dossier, <segment normalisé>)` ; un segment `20\\d\\d` ajoute aussi `(annee, YYYY)`
    (et devient le contexte d'année pour les segments mois suivants) ; un segment
    `MM-Libellé` (avec une année déjà vue plus haut dans le chemin) ou `MM.AAAA`
    (autonome) ajoute `(mois, YYYY-MM)`. Entités dédupliquées par document, ordre
    d'apparition préservé. Aucun segment de dossier -> [] (canal vide, jamais d'erreur)."""
    segments = doc_id.split("/")[:-1]
    relations: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    annee_courante: str | None = None

    def _add(role: str, entite: str) -> None:
        if not entite:
            return
        key = (role, entite)
        if key not in seen:
            seen.add(key)
            relations.append(key)

    for raw_seg in segments:
        seg_lower = raw_seg.lower()

        # Purs marqueurs de date : relation temporelle SEULE, pas de (dossier, ...) —
        # sinon « 2026 » et « 08.2026 » collapseraient tous deux vers (dossier, "2026")
        # (revue v2.0). Les mois LIBELLÉS (« 08-Août ») restent des dossiers cherchables.
        if _YEAR_RE.fullmatch(seg_lower):
            annee_courante = seg_lower
            _add("annee", seg_lower)
            continue

        m_ma = _MONTH_YEAR_RE.fullmatch(seg_lower)
        if m_ma:
            mois, annee = m_ma.group(1), m_ma.group(2)
            annee_courante = annee
            _add("mois", f"{annee}-{mois}")
            continue

        _add("dossier", _normalize_segment(raw_seg))
        m_ml = _MONTH_LABEL_RE.match(seg_lower)
        if m_ml and annee_courante is not None:
            mois = m_ml.group(1)
            _add("mois", f"{annee_courante}-{mois}")

    return relations
