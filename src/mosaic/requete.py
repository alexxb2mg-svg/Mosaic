"""Prétraitement DÉTERMINISTE de la requête — retirer le bruit avant de chercher.

MESURE QUI JUSTIFIE CE MODULE (13/08) : rapporté au score de BM25, le moteur
atteint **67 %** sur Alloprof (questions d'élèves : « Bonjour! Je voulais d'aide
avec ces question je n'arrive pas a comprendre comment les faire… ») et **86 %**
sur SciFact (affirmations scientifiques nettes). Ce n'est ni la langue ni le
corpus qui pénalise le moteur — c'est le BRUIT de la requête. Marge visée : les
19 points qui séparent les deux régimes.

PRINCIPE, et il est le même que celui du canal grammatical : le bruit d'une
question appartient à des **classes fermées** du français — salutations, formules
de politesse, demandes d'aide, aveux d'incompréhension, méta-discours (« ma
question est… »). On les retire par LISTE, jamais par jugement. Aucun modèle,
aucune heuristique inventée, aucune dépendance.

TROIS GARDE-FOUS, tenus par les tests :
  1. Jamais de requête VIDE : si le nettoyage retire tout, l'originale est
     rendue telle quelle (une requête vide ne trouve rien, ce serait pire que
     du bruit).
  2. Jamais de perte de terme PORTEUR : on ne retire que des motifs déclarés,
     jamais un mot inconnu — l'abstention plutôt que la devinette.
  3. Traçable : `nettoyer` rend AUSSI ce qui a été retiré, pour que la
     transformation soit lisible et vérifiable, jamais une boîte noire.
"""

from __future__ import annotations

import re

# --- classes fermées : ce qui n'est jamais un terme de recherche ------------------

# `_L` = liaison : espace, apostrophe droite ou typographique. Les trois se
# rencontrent dans les mêmes requêtes réelles (« s'il vous plaît », « s il vous
# plait », « s’il vous plaît ») — les traiter séparément était le premier bug.
_L = r"[\s'’]+"

# Salutations et politesses. « hi » est VOLONTAIREMENT absent : dans « Ly6C hi
# monocytes » (SciFact) c'est un terme du domaine (high expression), et le retirer
# détruisait un mot porteur — faux positif trouvé au diagnostic AVANT la mesure.
_CIVILITES = (
    r"bonjour|bonsoir|salut|coucou|hello|bonne journ[ée]e|bonne soir[ée]e"
    rf"|merci{_L}d{_L}?avance|merci beaucoup|merci|svp"
    rf"|s{_L}il vous pla[îi]t|s{_L}il te pla[îi]t"
    r"|d[ée]sol[ée]e?|excusez[- ]moi|pardon|cordialement"
)

# Demandes d'aide et méta-discours : la phrase parle de la question, pas du sujet
_META = (
    rf"j{_L}?(?:aurais |voudrais |aimerais )?(?:besoin d{_L}|voulais )?"
    rf"(?:un peu d{_L})?aide"
    rf"|pouvez[- ]vous m{_L}aider|peux[- ]tu m{_L}aider|aidez[- ]moi"
    rf"|j{_L}ai (?:une |des )?questions?|ma question (?:est|porte sur)"
    rf"|voici ma question|j{_L}aimerais savoir|je voudrais savoir"
    r"|est[- ]ce que quelqu.? un (?:peut|pourrait)"
    rf"|quelqu.? un (?:peut|pourrait) m{_L}(?:aider|expliquer)"
    rf"|pouvez[- ]vous m{_L}expliquer|peux[- ]tu m{_L}expliquer"
    rf"|merci de m{_L}(?:aider|expliquer|r[ée]pondre)"
)

# Aveux d'incompréhension : décrivent l'état de l'élève, pas la matière
_INCOMPREHENSION = (
    rf"je (?:ne |n{_L})?(?:arrive|comprends|comprend|sais|saisis) (?:pas|rien)"
    r"(?: (?:[àa] )?(?:comprendre|faire|le faire|les faire|la faire))?"
    rf"|j{_L}ai (?:du mal|des difficult[ée]s)(?: [àa] (?:comprendre|faire))?"
    r"|je (?:suis )?(?:bloqu[ée]e?|perdue?)"
    rf"|je (?:ne |n{_L})?(?:comprends|comprend) pas (?:comment|pourquoi|ce que)"
)

# Les CIVILITÉS ne sont retirées qu'en TÊTE ou en QUEUE de segment : au milieu
# d'une phrase, un mot de politesse peut être un terme du domaine (« la formule de
# politesse dans une lettre »). Les autres classes sont des locutions verbales
# assez longues pour être sans ambiguïté où qu'elles se trouvent.
_BORDS = r"(?:^|(?<=[\s,;:.!?—–-]))"
_MOTIFS = [
    re.compile(rf"{_BORDS}(?:{_CIVILITES})\b(?=[\s,;:.!?…)]*(?:$|\W))", re.I),
    re.compile(rf"\b(?:{_META})\b", re.I),
    re.compile(rf"\b(?:{_INCOMPREHENSION})\b", re.I),
]

# Ponctuation expressive : « !!! », « ??? » n'apportent rien, mais on garde UN
# point d'interrogation (il peut porter la nature interrogative pour un lecteur).
_PONCT_EXPRESSIVE = re.compile(r"([!?])\1+")
_ESPACES = re.compile(r"\s{2,}")
# Résidus de liaison laissés par une suppression : « , et », « et , », « : » isolés
_RESIDUS = re.compile(r"^[\s,;:.!?…\-–—'\"]+|[\s,;:]+(?=[,;:.])|\s+(?=[,.;:!?])")
# Ponctuation orpheline laissée par une suppression : « éléments. en détails.:) »
# -> la queue « .:) » ne porte rien. On coupe les amas de ponctuation résiduels.
# PAS de nettoyage de la ponctuation : la tokenisation du moteur l'ignore
# déjà, donc il serait SANS EFFET sur les résultats — et il détruisait du
# contenu (« zidovudine (AZT). » -> « zidovudine (AZT »). Retiré après
# diagnostic : ne garder que ce qui change une mesure.


def nettoyer(requete: str) -> tuple[str, list[str]]:
    """Retire le bruit conversationnel d'une requête.

    Rend `(requête_nettoyée, fragments_retirés)`. Les fragments sont rendus pour
    que la transformation soit **vérifiable** — un prétraitement muet serait
    invérifiable, donc inacceptable.

    Si le nettoyage vide la requête (elle n'était QUE de la politesse), l'originale
    est rendue inchangée : mieux vaut chercher du bruit que ne rien chercher.
    """
    if not requete or not requete.strip():
        return requete, []
    retires: list[str] = []
    texte = requete
    for motif in _MOTIFS:

        def _capturer(m: re.Match) -> str:
            retires.append(m.group(0).strip())
            return " "

        texte = motif.sub(_capturer, texte)
    texte = _PONCT_EXPRESSIVE.sub(r"\1", texte)
    texte = _RESIDUS.sub("", texte)
    texte = _ESPACES.sub(" ", texte).strip(" ,;:-–—")
    if not texte.strip():
        return requete, []  # garde-fou 1 : jamais de requête vide
    return texte, retires


def taux_de_bruit(requete: str) -> float:
    """Part des caractères retirés — diagnostic, pour savoir si un corpus de
    requêtes justifie le prétraitement AVANT de l'activer."""
    if not requete.strip():
        return 0.0
    nette, _ = nettoyer(requete)
    return round(1.0 - len(nette) / len(requete), 4)
