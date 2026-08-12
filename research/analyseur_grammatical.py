"""P2 du chantier « canal grammatical » — fiabilité de l'analyseur déterministe.

LE PARAMÈTRE DÉCISIF DU BRIEF, mesuré AVANT tout code d'intégration : un rôle mal
attribué MENT à l'agent qui tranche en aval, donc l'analyseur doit S'ABSTENIR plutôt
que deviner. Fiabilité d'abord, couverture ensuite.

ANALYSEUR : règles pures sur tokens + listes FERMÉES (aucun réseau, aucun pip install,
aucune ressource externe — le Lefff n'est PAS utilisé : les rôles visés sont portés par
des classes fermées du français (négateurs, prépositions, copules) et par de petites
listes de verbes du domaine ; un lexique morphologique large n'aiderait que les cas
ouverts, précisément ceux où l'ambiguïté impose l'abstention — choix documenté).

RÔLES PRODUITS (étroit, comme demandé) :
- absent  : négation à portée. Négateurs nominaux {sans, aucun, aucune, ni} : la portée
  court sur les tokens pleins suivants (max 4) jusqu'à un délimiteur (ponctuation,
  conjonction, préposition, ou dé+article défini « du/des/de la/de l' » — un complément
  au référent défini EXISTE, on ne le déclare pas absent ; le « de » nu continue :
  qualificateur de type, « conducteur de terre »). Idiomes bloqués après sans :
  {fin, objet, délai, préavis, suite, cesse}. Cadre verbal « ne/n' … pas|plus|jamais » :
  V copule -> absent(attribut après pas) ; V existentiel {dispose, comporte, possède,
  prévoit, inclut, intègre, présente} suivi de « de » -> absent(portée après de) ;
  sinon absent(V) seul — l'action est niée, pas ses arguments. « pas » SANS ne/n' :
  jamais négateur (« le pas de vis ») -> abstention. « non » + token plein suivant.
- amont/aval : motif « X … en amont|aval de/du/des Y ». X = premier token plein à
  gauche de « en » en sautant copules/participes/verbes de pose (liste fermée) ;
  Y = premier token plein après le « de ». X ou Y introuvable (début de phrase,
  ponctuation) -> ABSTENTION TOTALE, jamais une demi-relation. « amont/aval » hors de
  ce motif (réseau amont, l'aval du maître d'ouvrage) -> rien.
- agent/patient : voix active « X V Y » avec V dans une liste fermée de verbes du
  domaine ; voix passive « X est|sont PARTICIPE par Y ». Argument manquant ou verbe
  nié -> abstention (la relation ne tient pas).

CONVENTION D'ARGUMENT (déclarée, appliquée aussi à la vérité du banc) : l'argument est
le token plein le PLUS PROCHE du motif (« transformateur de courant se monte en
amont… » -> courant). Trace, pas reconstruction : un token de tête ou son qualificateur
suffisent à faire remonter le document. Les nombres et tokens < 3 lettres (30, mA) ne
sont pas des entités.

ENCODAGE DU « NON ATTRIBUÉ » : aucune émission -> le canal reste le vecteur nul, donc
strictement NEUTRE dans toute similarité (réponse à la question ouverte du brief : le
silence n'est pas un rôle, c'est l'absence de contribution).

PRÉDICTIONS DÉCLARÉES AVANT MESURE (falsifiables) :
- P2a taux d'erreur de rôle (émissions fausses / émissions) <= 5 % — les erreurs
  attendues sont nommées dans le banc : T07 (ASI, x2) et C05 (verbe de flux, x1).
  GATE du protocole : > 10 % -> STOP, rapport sans intégration.
- P2b couverture (relations vraies retrouvées / relations vraies) >= 85 % — les
  manques attendus sont les abstentions volontaires C01, C04 et C05 partiel.
- P2c abstention propre : 8/8 pièges à vérité vide sans AUCUNE émission.

Usage : python research/analyseur_grammatical.py  (lit research/banc_grammatical.jsonl)
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosaic.tokenize import STOPWORDS

# --- lexique fermé -----------------------------------------------------------------

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


def _tokens(clause: str) -> list[str]:
    return _MOT_RE.findall(clause.lower())


def _est_plein(tok: str) -> bool:
    """Token plein = candidat entité : alphabétique (accents/traits d'union), >= 3 chars."""
    return len(tok) >= 3 and not any(ch.isdigit() for ch in tok) and tok not in _PONCT


def _argument_gauche(toks: list[str], i: int) -> str | None:
    """Premier token plein à gauche de la position i, en sautant SAUT_GAUCHE et les
    stopwords ; ponctuation, début de phrase ou NÉGATEUR -> None (un argument nié
    n'est pas un argument : la relation ne tient pas, on s'abstient)."""
    j = i - 1
    while j >= 0:
        t = toks[j]
        if t in _PONCT or t in NEGATEURS_NOMINAUX:
            return None
        if t in SAUT_GAUCHE or t in STOPWORDS or not _est_plein(t):
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


def _portee_negation(toks: list[str], i: int) -> list[str]:
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
        if portee and (t in VERBES_ACTIFS or t in COPULES):
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


def analyse(clause: str) -> list[tuple[str, str]]:
    """Rôles (role, entite) d'une clause — liste FERMÉE de motifs, abstention par défaut."""
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
            for ent in _portee_negation(toks, i):
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
                for ent in _portee_negation(toks, i + 1):
                    emet("absent", ent)  # ne dispose pas de bypass manuel
            elif verbe is not None:
                emet("absent", verbe)  # ne coupe pas … : l'action est niée
            continue

        # ---- amont / aval : « en amont|aval de … » -----------------------------
        if t in ("amont", "aval") and i > 0 and toks[i - 1] == "en":
            if i + 1 >= len(toks) or toks[i + 1] not in (DE_NUS | DE_DEFINIS):
                continue
            x = _argument_gauche(toks, i - 1)
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
        if t in VERBES_ACTIFS and i not in nies:
            x = _argument_gauche(toks, i)
            y = _argument_droit(toks, i)
            if x is not None and y is not None:
                emet("agent", x)
                emet("patient", y)
            continue

        # ---- voix passive : X est PARTICIPE par Y ------------------------------
        if t in PARTICIPES_PASSIFS and i > 0 and toks[i - 1] in COPULES:
            if i + 1 < len(toks) and toks[i + 1] == "par":
                x = _argument_gauche(toks, i - 1)
                y = _argument_droit(toks, i + 1)
                if x is not None and y is not None:
                    emet("patient", x)
                    emet("agent", y)
            continue

    return rel


# --- P2 : mesure contre le banc -----------------------------------------------------


def _contenu(clause: str) -> dict[str, int]:
    """Multiset des tokens hors stopwords (contrôle « mots identiques » des paires)."""
    compte: dict[str, int] = {}
    for t in _tokens(clause):
        if t not in _PONCT and t not in STOPWORDS:
            compte[t] = compte.get(t, 0) + 1
    return compte


def main() -> int:
    banc = Path(__file__).resolve().parent / "banc_grammatical.jsonl"
    lignes = [
        json.loads(li) for li in banc.read_text(encoding="utf-8").splitlines() if li
    ]

    # contrôle du banc : chaque paire a des mots (pleins) identiques et des rôles distincts
    paires: dict[str, list[dict]] = {}
    for c in lignes:
        if c["paire"]:
            paires.setdefault(c["paire"], []).append(c)
    for pid, (a, b) in sorted((k, v) for k, v in paires.items()):
        assert _contenu(a["clause"]) == _contenu(b["clause"]), (
            f"paire {pid} : mots differents"
        )
        assert a["verite"] != b["verite"], f"paire {pid} : verites identiques"
    print(
        f"banc : {len(lignes)} clauses, {len(paires)} paires (mots identiques verifies)\n"
    )

    emissions = erreurs = vraies = retrouvees = 0
    pieges_vides = pieges_silencieux = 0
    detail_erreurs: list[str] = []
    for c in lignes:
        produit = analyse(c["clause"])
        verite = {(r, e) for r, e in c["verite"]}
        emissions += len(produit)
        vraies += len(verite)
        retrouvees += sum(1 for p in produit if p in verite)
        for p in produit:
            if p not in verite:
                erreurs += 1
                detail_erreurs.append(f"  {c['id']}: {p[0]}:{p[1]}")
        if c["categorie"] == "piege" and not verite:
            pieges_vides += 1
            if not produit:
                pieges_silencieux += 1

    taux_err = erreurs / emissions if emissions else 0.0
    couverture = retrouvees / vraies if vraies else 0.0
    print(f"P2a taux d'erreur de role : {erreurs}/{emissions} = {taux_err:.1%}")
    if detail_erreurs:
        print("\n".join(detail_erreurs))
    print(f"P2b couverture           : {retrouvees}/{vraies} = {couverture:.1%}")
    print(
        f"P2c abstention propre    : {pieges_silencieux}/{pieges_vides} pieges silencieux"
    )
    print()
    if taux_err > 0.10:
        print("GATE : erreur > 10 % -> STOP (rapport, pas d'integration)")
        return 1
    print("GATE : erreur <= 10 % -> P2 PASSE, P1 autorise (canal_grammatical.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
