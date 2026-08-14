"""Aiguilleur de requête — envoyer la question au circuit qui sait y répondre.

Le moteur a plusieurs circuits, et ils ne répondent pas aux mêmes questions :
la recherche classe par ressemblance, le magasin structurel COMPTE et ORDONNE,
le boost de référence retrouve un code à l'identique. Aujourd'hui c'est à
l'appelant de choisir, et il choisit mal — mesuré le 14/08 sur un corpus réel :
la même question posée en trois mots discriminants sort au rang 2, décorée de
six mots plausibles au rang 231. Le facteur limitant n'est ni le classement ni
la robustesse, c'est la FORMULATION.

PRINCIPE, hérité de `requete.py` : les marqueurs de comptage et d'ordre
appartiennent à des CLASSES FERMÉES du français (« combien », « nombre de »,
« le plus récent », « dernier »). On les reconnaît par liste, jamais par
jugement, et chaque décision NOMME le motif qui l'a déclenchée.

LE PIÈGE, déclaré avant d'écrire ce module (P-C3, file de recherche) : « le
piège sera la détection. Un routage naïf enverrait tout à la lecture exacte et
casserait la prose. » D'où la règle de prudence qui gouverne tout ici :

    en cas de doute, SÉMANTIQUE.

C'est le seul circuit qui ne se trompe jamais de NATURE de réponse — il rend des
documents, ce que toute question accepte. Un comptage manqué coûte une reprise ;
une question de sens envoyée au comptage rend un nombre absurde. Les deux
erreurs ne se valent pas, l'aiguilleur penche donc toujours du même côté.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Circuit(Enum):
    """Où envoyer la question. L'ordre de priorité est celui du code, pas celui-ci."""

    COMPTAGE = "comptage"  # magasin structurel : compter(), repartition_par_mois()
    ORDRE = "ordre"  # magasin structurel : plus_recents()
    REFERENCE = "reference"  # recherche + boost de référence exacte
    SEMANTIQUE = "semantique"  # recherche en fusion — le défaut, et le refuge


@dataclass(frozen=True)
class Route:
    """Décision d'aiguillage, toujours JUSTIFIÉE : `motif` nomme ce qui l'a
    déclenchée, pour qu'un humain puisse contester la décision sans lire le code."""

    circuit: Circuit
    motif: str
    refs: tuple[str, ...] = ()
    type_doc: str | None = None
    k: int | None = None


# --- classes fermées ------------------------------------------------------------

# « combien de X », « nombre de X » : la question attend un NOMBRE. Le « de » est
# exigé — « combien coûte » est une question de prix, pas de dénombrement.
_COMPTAGE = re.compile(
    r"\bcombien\s+(?:de|d')\b"
    r"|\bcombien\s+y\s+a-t-il\b"
    r"|\bnombre\s+(?:de|d')\b"
    r"|\bquantité\s+(?:de|d')\b",
    re.IGNORECASE,
)
# Faux amis du comptage, CALIBRÉS SUR LE RÉEL (2 316 questions, 14/08) :
# « un grand nombre de » est une tournure de prose ; « combien de temps » demande
# une DURÉE et « combien de fois » une FRÉQUENCE — ni l'une ni l'autre ne se
# comptent dans un corpus de documents.
_COMPTAGE_FAUX_AMIS = re.compile(
    r"\b(?:un|le)\s+(?:grand|certain|petit)\s+nombre\s+(?:de|d')\b"
    r"|\bcombien\s+(?:de\s+)?(?:temps|fois)\b",
    re.IGNORECASE,
)

# « le plus récent », « le dernier X » : la question attend UN document, celui qui
# vient en tête d'un ordre — pas les plus ressemblants.
_ORDRE = re.compile(
    r"\b(?:le|la|les)\s+plus\s+récent(?:e|s|es)?\b"
    r"|\bplus\s+récent(?:e|s|es)?\b"
    r"|\b(?:le|la|les)\s+dernier(?:s)?\b|\b(?:la|les)\s+dernière(?:s)?\b"
    r"|\b\d+\s+dernier(?:s|es)?\b|\b\d+\s+dernières\b",
    re.IGNORECASE,
)
# « le dernier étage », « le dernier mot » : « dernier » est spatial ou figuré.
_ORDRE_FAUX_AMIS = re.compile(
    r"\bdernier(?:s)?\s+(?:étage|niveau|mot|recours|ressort|cri)\b"
    r"|\bdernière(?:s)?\s+(?:main|minute|chance)\b",
    re.IGNORECASE,
)

# Une question de MÉTHODE (« comment calculer le nombre de… ») demande un procédé,
# pas un dénombrement du corpus : « combien de BL en juin » compte des documents,
# « comment compter les BL » demande une explication. La distinction vaut hors de
# tout corpus particulier, et elle emportait à elle seule la moitié des fausses
# alarmes restantes (mesuré le 14/08).
_METHODE = re.compile(
    r"\bcomment\b|\bcalculer\b|\bconvertir\b|\bexpliqu\w+\b|\bméthode\b|\bformule\b",
    re.IGNORECASE,
)

# Types de documents nommés dans la question — un FILTRE, jamais un circuit.
_TYPES = {
    "photo": ("photo", "photos", "cliché", "clichés", "image", "images"),
    "tableur": ("tableur", "tableau de prix", "liste de prix", "excel", "xlsx"),
    "pdf scanné": ("scan", "scanné", "scannée", "scannés"),
    "note texte": ("note", "notes"),
}

# Une RÉFÉRENCE : code alphanumérique mixte d'au moins 5 caractères, ou nombre
# d'au moins 6 chiffres. Le seuil est celui du boost de facettes déjà en place
# (facettes.py) : deux circuits qui divergeraient sur « qu'est-ce qu'une réf »
# seraient pires qu'un seul qui se trompe.
_REF_MIXTE = re.compile(r"\b(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[A-Za-z0-9-]{5,}\b")
_REF_NUM = re.compile(r"\b\d{6,}\b")
# Une année seule (19xx/20xx) n'est pas une référence : « les devis de 2026 ».
_ANNEE = re.compile(r"^(?:19|20)\d{2}$")
# Une mesure suivie d'une unité non plus : « 240 mm », « 3000K », « 16W ».
# Une PLAGE de mesures non plus (« 16-18W », « 20-40mm ») : mesuré sur nos propres
# requêtes (journal des recherches, 14/08), « encastré LED 240 mm 3000K 16-18W »
# partait au circuit référence à cause de « 16-18W ».
_MESURE = re.compile(
    r"^\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?)?(?:mm|cm|m|k|w|v|a|va|kw|ma|lm|°|%)$",
    re.IGNORECASE,
)


def _est_code(mot: str) -> bool:
    """Un code métier s'écrit en MAJUSCULES (DE26040008, F24069901, A1760620).

    Garde calibrée sur le réel (14/08, 2 316 questions) : sans elle, `5iemes`,
    `tan-1`, `x10000` et les pseudonymes en CamelCase passaient pour des
    références et déroutaient une question de sens sur vingt. Un mot dont les
    lettres ne sont pas toutes en capitales n'est pas un code — c'est du texte
    qui contient un chiffre."""
    lettres = [c for c in mot if c.isalpha()]
    return not lettres or all(c.isupper() for c in lettres)


def _refs(question: str) -> tuple[str, ...]:
    """Codes présents dans la question, sans les années, mesures ni mots courants."""
    trouves = []
    for m in list(_REF_MIXTE.finditer(question)) + list(_REF_NUM.finditer(question)):
        mot = m.group(0)
        if _ANNEE.match(mot) or _MESURE.match(mot) or not _est_code(mot):
            continue
        if mot not in trouves:
            trouves.append(mot)
    return tuple(trouves)


def _type_doc(question: str) -> str | None:
    bas = question.lower()
    for type_canonique, marqueurs in _TYPES.items():
        for mot in marqueurs:
            if re.search(rf"\b{re.escape(mot)}\b", bas):
                return type_canonique
    return None


def _combien(question: str) -> int | None:
    """« les 3 derniers » -> 3. Rend None si la question ne chiffre rien."""
    m = re.search(r"\b(\d+)\s+dernier", question, re.IGNORECASE)
    return int(m.group(1)) if m else None


def aiguiller(question: str) -> Route:
    """Choisit le circuit pour cette question. Ne lève jamais, ne devine jamais.

    Priorité : comptage > ordre > référence > sémantique. Le comptage passe en
    premier parce qu'une question qui demande un nombre ET porte une référence
    (« combien de BL portent la réf X ») attend d'abord un nombre."""
    q = question.strip()
    if not q:
        return Route(Circuit.SEMANTIQUE, "question vide")

    type_doc = _type_doc(q)

    # Une question de méthode reste sémantique, quels que soient ses marqueurs :
    # elle attend une explication, et aucun compte ne l'y aidera.
    if _METHODE.search(q):
        return Route(
            Circuit.SEMANTIQUE,
            "question de méthode (comment/calculer) — le compte ne répondrait pas",
            type_doc=type_doc,
        )

    m = _COMPTAGE.search(q)
    if m and not _COMPTAGE_FAUX_AMIS.search(q):
        return Route(
            Circuit.COMPTAGE,
            f"marqueur de comptage « {m.group(0)} »",
            refs=_refs(q),
            type_doc=type_doc,
        )

    m = _ORDRE.search(q)
    if m and not _ORDRE_FAUX_AMIS.search(q):
        return Route(
            Circuit.ORDRE,
            f"marqueur d'ordre « {m.group(0)} »",
            refs=_refs(q),
            type_doc=type_doc,
            k=_combien(q),
        )

    refs = _refs(q)
    if refs:
        return Route(
            Circuit.REFERENCE,
            f"référence exacte « {refs[0]} »",
            refs=refs,
            type_doc=type_doc,
        )

    return Route(
        Circuit.SEMANTIQUE, "aucun marqueur — circuit par défaut", type_doc=type_doc
    )
