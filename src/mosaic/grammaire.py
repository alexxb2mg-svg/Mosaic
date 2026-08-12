"""Canal grammatical déterministe — la « graine de sens structural » (brief 12/08).

Le seul reproche de fond au moteur : sac-de-mots. « disjoncteur en amont du
différentiel » et « différentiel en amont du disjoncteur » — mots identiques, sens
opposé — sont indistinguables (mesuré : cosinus intra-paire = 1.0000 exactement sur
25/34 paires plantées). Ce module produit des RÔLES grammaticaux par un analyseur
DÉTERMINISTE à règles et classes fermées du français (aucun réseau, aucune ressource
externe — le Lefff même n'est pas requis : les rôles visés sont portés par les
classes fermées, et un lexique large n'aiderait que les cas ouverts, précisément
ceux où l'ambiguïté impose l'abstention).

Doctrine « trace, pas reconstruction » : on ne reconstruit pas la phrase (capacité de
superposition finie — bancs horizon) ; on dépose 3-4 traces à haute valeur par
document (négation à portée ; amont/aval ordonné ; agent/patient), liées aux
signatures par la permutation circulaire du canal relations (bind/np.roll).

GARDE-FOU FONDATEUR (mesuré, P2) : le bruit coûte plus cher que le silence — un rôle
mal attribué MENT à l'agent qui tranche en aval. L'analyseur S'ABSTIENT sur l'ambigu
(erreur 2,1 %, abstention propre sur tous les pièges plantés du banc
research/banc_grammatical.jsonl). Le non-attribué n'émet RIEN : le canal reste un
vecteur nul, strictement neutre dans toute similarité.

Le canal est SÉPARÉ et opt-in (--grammatical) : désactivé, l'impact est nul au bit
près ; la fusion silencieuse dans le vecteur principal est interdite par la leçon
mesurée des grilles typées (dilution). Validation P1 : 33/34 paires séparées (97 %,
seuil déclaré 80 %) — cf. research/canal_grammatical.py.
"""

import re
from typing import NamedTuple

import numpy as np

from mosaic.relations import document_channel
from mosaic.tokenize import STOPWORDS


_MOT_RE = re.compile(
    r"[a-z0-9àâäæçéèêëîïôöœùûüÿ]+(?:-[a-z0-9àâäæçéèêëîïôöœùûüÿ]+)*|[,;:.()!?]"
)
_PONCT = frozenset(",;:.()!?")

NEGATEURS_NOMINAUX = frozenset({"sans", "aucun", "aucune", "ni"})
IDIOMES_SANS = frozenset({"fin", "objet", "délai", "préavis", "suite", "cesse"})
FERMETURES_NE = frozenset({"pas", "plus", "jamais"})
COPULES = frozenset(
    "est sont sera seront était étaient reste restent demeure demeurent a ont aura".split()
)
EXISTENTIELS = frozenset(
    "dispose comporte possède prévoit inclut intègre présente".split()
)

# délimiteurs de portée de négation : conjonctions, prépositions, dé+défini
DELIM_PORTEE = frozenset(
    """et mais ou puis donc car dans sur sous pour par avec depuis pendant entre vers
    chez à au aux du des hors selon après avant que qui dont où ne n""".split()
)
ARTICLES = frozenset("le la les l un une".split())

# amont/aval : sauts à gauche (copules, participes et verbes de POSE — liste fermée)
_PARTICIPES_POSE = "placé situé installé monté posé raccordé câblé implanté inséré positionné disposé prévu repris"
_VERBES_POSE = (
    "monte montent raccorde raccordent branche branchent câble câblent place placent "
    "situe situent installe installent pose posent insère insèrent implante implantent"
)


def _avec_accords(bases: str) -> frozenset[str]:
    formes = set()
    for b in bases.split():
        formes.update({b, b + "e", b + "s", b + "es"})
    return frozenset(formes)


SAUT_GAUCHE = (
    _avec_accords(_PARTICIPES_POSE)
    | frozenset(_VERBES_POSE.split())
    | COPULES
    | frozenset(
        "se trouve trouvent figure figurent directement immédiatement juste".split()
    )
    | ARTICLES
    | frozenset("ne n".split())
)

VERBES_ACTIFS = frozenset(
    "alimente alimentent commande commandent protège protègent coupe coupent "
    "pilote pilotent dessert desservent contrôle contrôlent déclenche déclenchent".split()
)
PARTICIPES_PASSIFS = _avec_accords(
    "alimenté commandé protégé coupé piloté desservi contrôlé déclenché"
)

DE_NUS = frozenset({"de", "d"})
DE_DEFINIS = frozenset({"du", "des", "au", "aux"})


class Extension(NamedTuple):
    """Listes verbales OUVERTES AU MÉTIER, déclarées par le profil (clé « grammaire »).

    Seules les listes dépendantes du domaine sont extensibles (verbes d'action,
    participes passifs, sauts de pose) — les classes fermées du français (négateurs,
    copules, prépositions, délimiteurs de portée) ne le sont PAS : c'est leur clôture
    qui fonde le déterminisme et le taux d'erreur mesuré de l'analyseur."""

    verbes_actifs: frozenset[str]
    participes_passifs: frozenset[str]
    saut_gauche: frozenset[str]


def extension_depuis_profil(profil: dict | None) -> Extension | None:
    """Extension grammaticale déclarée par un profil d'index, ou None.

    Contrat (doctrine abstention > devinette, donc AUCUNE magie morphologique) :
    - `verbes_actifs` et `saut_gauche` : formes conjuguées EXACTES, telles quelles ;
    - `participes_passifs` : masculin singulier, les accords e/s/es sont mécaniques
      et ajoutés automatiquement (seule flexion sans ambiguïté)."""
    g = (profil or {}).get("grammaire")
    if not g:
        return None
    return Extension(
        verbes_actifs=frozenset(g.get("verbes_actifs", ())),
        participes_passifs=_avec_accords(" ".join(g.get("participes_passifs", ()))),
        saut_gauche=frozenset(g.get("saut_gauche", ())),
    )


def _tokens(clause: str) -> list[str]:
    return _MOT_RE.findall(clause.lower())


def _est_plein(tok: str) -> bool:
    """Token plein = candidat entité : alphabétique (accents/traits d'union), >= 3 chars."""
    return len(tok) >= 3 and not any(ch.isdigit() for ch in tok) and tok not in _PONCT


def _argument_gauche(
    toks: list[str], i: int, saut: frozenset[str] = SAUT_GAUCHE
) -> str | None:
    """Premier token plein à gauche de la position i, en sautant `saut` et les
    stopwords ; ponctuation, début de phrase ou NÉGATEUR -> None (un argument nié
    n'est pas un argument : la relation ne tient pas, on s'abstient)."""
    j = i - 1
    while j >= 0:
        t = toks[j]
        if t in _PONCT or t in NEGATEURS_NOMINAUX:
            return None
        if t in saut or t in STOPWORDS or not _est_plein(t):
            j -= 1
            continue
        return t
    return None


def _argument_droit(toks: list[str], i: int) -> str | None:
    """Premier token plein à droite de la position i (saute articles/stopwords) ;
    ponctuation ou négateur -> None (même principe : jamais un argument nié)."""
    j = i + 1
    while j < len(toks):
        t = toks[j]
        if t in _PONCT or t in NEGATEURS_NOMINAUX:
            return None
        if t in ARTICLES or t in STOPWORDS or not _est_plein(t):
            j += 1
            continue
        return t
    return None


def _portee_negation(
    toks: list[str], i: int, va: frozenset[str] = VERBES_ACTIFS
) -> list[str]:
    """Tokens pleins couverts par un négateur nominal en position i (max 4)."""
    portee: list[str] = []
    j = i + 1
    while j < len(toks) and len(portee) < 4:
        t = toks[j]
        if (
            t in _PONCT
            or t in NEGATEURS_NOMINAUX
            or t in DELIM_PORTEE
            or t in DE_DEFINIS
        ):
            break
        if portee and (t in va or t in COPULES):
            break  # un verbe conjugué (liste fermée) termine le groupe nominal nié —
            # mais la TÊTE du groupe peut être un homographe (« aucun contrôle de… »)
        if t in DE_NUS:
            # « de » nu = qualificateur de type (continue) ; « de la/l' » = référent
            # défini (stop : il existe, il n'est pas absent)
            if j + 1 < len(toks) and toks[j + 1] in ARTICLES:
                break
            j += 1
            continue
        if t in ARTICLES or not _est_plein(t):
            j += 1
            continue
        portee.append(t)
        j += 1
    return portee


def _indices_verbes_nies(toks: list[str]) -> set[int]:
    """Indices des tokens pris dans un cadre « ne/n' … pas|plus|jamais »."""
    nies: set[int] = set()
    for i, t in enumerate(toks):
        if t in ("ne", "n"):
            for j in range(i + 1, min(i + 4, len(toks))):
                if toks[j] in FERMETURES_NE:
                    nies.update(range(i, j + 1))
                    break
    return nies


def analyse(clause: str, extension: Extension | None = None) -> list[tuple[str, str]]:
    """Rôles (role, entite) d'une clause — liste FERMÉE de motifs, abstention par
    défaut. `extension` (clé « grammaire » d'un profil d'index) élargit les seules
    listes ouvertes au métier."""
    va, pp, saut = VERBES_ACTIFS, PARTICIPES_PASSIFS, SAUT_GAUCHE
    if extension is not None:
        va = va | extension.verbes_actifs
        pp = pp | extension.participes_passifs
        saut = saut | extension.saut_gauche
    toks = _tokens(clause)
    rel: list[tuple[str, str]] = []
    vu: set[tuple[str, str]] = set()

    def emet(role: str, ent: str | None) -> None:
        if ent and (role, ent) not in vu:
            vu.add((role, ent))
            rel.append((role, ent))

    nies = _indices_verbes_nies(toks)

    for i, t in enumerate(toks):
        # ---- négation nominale : sans / aucun / aucune / ni --------------------
        if t in NEGATEURS_NOMINAUX:
            if t == "sans" and i + 1 < len(toks) and toks[i + 1] in IDIOMES_SANS:
                continue  # vis sans fin, sans suite… : idiome, abstention
            for ent in _portee_negation(toks, i, va):
                emet("absent", ent)
            continue

        # ---- non + token plein -------------------------------------------------
        if t == "non":
            emet("absent", _argument_droit(toks, i))
            continue

        # ---- cadre verbal ne … pas/plus/jamais ---------------------------------
        if t in FERMETURES_NE and i in nies:
            # remonte au verbe entre ne et la fermeture
            verbe = None
            for j in range(i - 1, -1, -1):
                if toks[j] in ("ne", "n"):
                    break
                if _est_plein(toks[j]) or toks[j] in COPULES:
                    verbe = toks[j]
            if verbe in COPULES:
                emet("absent", _argument_droit(toks, i))  # n'est pas conforme
            elif verbe in EXISTENTIELS and i + 1 < len(toks) and toks[i + 1] in DE_NUS:
                for ent in _portee_negation(toks, i + 1, va):
                    emet("absent", ent)  # ne dispose pas de bypass manuel
            elif verbe is not None:
                emet("absent", verbe)  # ne coupe pas … : l'action est niée
            continue

        # ---- amont / aval : « en amont|aval de … » -----------------------------
        if t in ("amont", "aval") and i > 0 and toks[i - 1] == "en":
            if i + 1 >= len(toks) or toks[i + 1] not in (DE_NUS | DE_DEFINIS):
                continue
            x = _argument_gauche(toks, i - 1, saut)
            y = _argument_droit(toks, i + 1)
            if x is None or y is None:
                continue  # jamais une demi-relation
            if t == "amont":
                emet("amont", x)
                emet("aval", y)
            else:
                emet("aval", x)
                emet("amont", y)
            continue

        # ---- voix active : X V Y (V dans la liste fermée, non nié) -------------
        if t in va and i not in nies:
            x = _argument_gauche(toks, i, saut)
            y = _argument_droit(toks, i)
            if x is not None and y is not None:
                emet("agent", x)
                emet("patient", y)
            continue

        # ---- voix passive : X est PARTICIPE par Y ------------------------------
        if t in pp and i > 0 and toks[i - 1] in COPULES:
            if i + 1 < len(toks) and toks[i + 1] == "par":
                x = _argument_gauche(toks, i - 1, saut)
                y = _argument_droit(toks, i + 1)
                if x is not None and y is not None:
                    emet("patient", x)
                    emet("agent", y)
            continue

    return rel


# --- P2 : mesure contre le banc -----------------------------------------------------


def canal_document(
    texte: str, dim: int, extension: Extension | None = None
) -> tuple[np.ndarray, float]:
    """Canal grammatical d'un document : Σ bind(rôle, entité) sur les rôles analysés
    du TEXTE BRUT (l'analyseur exige l'ordre des mots et la ponctuation — jamais le
    flux canonicalisé), normalisé et quantifié int8 par le schéma du canal relations.
    Aucun rôle -> vecteur nul, norme 0.0 (neutre)."""
    return document_channel(analyse(texte, extension), dim)
