"""N-grammes de caractères — donner au moteur la notion de PRESQUE.

Le problème, constaté sur un document réel le 14/08 : l'OCR d'un bon de livraison
lit « Comer » là où le logo imprime « Fournisseur ». Pour une recherche par mot exact,
ces deux mots n'ont RIEN en commun — le fournisseur devient introuvable par son
nom sur toute une famille de scans, et 92 % des pièces comptables sont scannées.

Mesuré avant d'écrire ce module (`research/tolerance_ocr.py`, corpus métier) :
une seule confusion de glyphe sur le mot le plus long d'une requête coûte
**10 points de rappel** en moyenne, et fait tomber une requête sur dix de 1,00 à
**0,00** — effondrement total, pas dégradation.

La réponse est mécanique et sans modèle : découper les mots en fragments de
caractères et les indexer À CÔTÉ des mots entiers. « fournisseur » et « comer »
partagent alors trois fragments sur six, là où l'égalité de mots en partage zéro.

TROIS CHOIX, chacun contre un piège précis :

1. **Les fragments sont MARQUÉS (`§`)** — sans marqueur, le trigramme « cha »
   entrerait en collision avec le mot « cha » s'il existait, et deux mondes se
   mélangeraient dans la même grille.
2. **Les bornes de mot comptent (`^`, `$`)** — sinon « tableau » et « tabIeau »
   se ressembleraient autant que « tableau » et « tableaux », et un début de
   mot ne vaudrait pas plus qu'un milieu.
3. **Les mots COURTS ne sont pas fragmentés** — « de », « sur », « le » n'ont
   aucun pouvoir discriminant, et les fragmenter noierait le flux sous du bruit.
   Le seuil est la longueur du n-gramme plus un : en dessous, un mot ne produit
   qu'un fragment ou deux, qui ne disent rien de plus que lui.

Les fragments s'AJOUTENT au flux, ils ne remplacent jamais les mots : on ne troque
pas la précision contre la tolérance, on ajoute une seconde chance.

VERDICT DE SON BANC (14/08, `research/tolerance_ocr.py`, corpus métier —
40 requêtes, confusions de glyphes injectées) :

    stratégie                     chute due à l'OCR   dilution (requêtes saines)
    sans n-grammes                    −10,0 pts                  —
    fragmenter TOUS les mots            0,0 pt              −10,0 pts
    repli sur les mots inconnus        −7,5 pts               0,0 pt

**ÉCARTÉ au regard du critère déclaré** (diviser la chute par deux), mais le
résultat est plus intéressant que le verdict, parce qu'il isole exactement le
compromis :

1. **Fragmenter systématiquement rend le moteur INSENSIBLE aux fautes d'OCR** —
   la chute tombe à zéro. Et coûte dix points de rappel sur toutes les autres
   requêtes : on perdrait sur cent pour cent des recherches afin de gagner sur
   dix. Mauvais échange, sans appel.
2. **Un second banc a isolé l'origine de cette dilution : elle vient ENTIÈREMENT
   de la requête**, jamais de l'index. Un index enrichi interrogé sans fragmenter
   rend exactement le rappel d'un index normal (0,8000 contre 0,8000). Autrement
   dit, porter les fragments côté documents ne coûte RIEN en qualité — seulement
   en taille de vocabulaire.
3. D'où le REPLI implémenté dans `queries.flux_requete` : ne fragmenter que les
   mots absents du vocabulaire. Dilution nulle, et 2,5 points de tolérance
   regagnés. Le gain est réel mais partiel — les fragments d'un mot isolé sont
   peu discriminants, et une requête d'un seul mot abîmé reste floue.

**Défaut : DÉSACTIVÉ** (`ngrammes=0`). L'option existe, mesurée et documentée ;
elle n'entre pas en production sur un gain de 2,5 points. Ce qui reste ouvert :
pondérer les fragments plus faiblement que les mots, ou n'enrichir que les
documents SCANNÉS (là où l'OCR fait des fautes) plutôt que tout le corpus — deux
variantes non mesurées, donc deux hypothèses, pas deux solutions.
"""

from __future__ import annotations

MARQUEUR = "§"


def ngrammes_du_mot(mot: str, n: int) -> list[str]:
    """Fragments marqués d'un mot, bornes comprises. Vide si le mot est trop court.

    « chat » en trigrammes -> ['§^ch', '§cha', '§hat', '§at$']"""
    if n <= 0 or len(mot) < n + 1:
        return []
    borne = f"^{mot}$"
    return [MARQUEUR + borne[i : i + n] for i in range(len(borne) - n + 1)]


def enrichir(tokens: list[str], n: int) -> list[str]:
    """Le flux d'origine SUIVI de ses fragments — l'ordre garde les mots devant.

    Déterministe : même entrée, même sortie, toujours. Un `n` nul ou négatif rend
    le flux intact plutôt que de le corrompre en silence."""
    if n <= 0 or not tokens:
        return list(tokens)
    fragments: list[str] = []
    for mot in tokens:
        fragments.extend(ngrammes_du_mot(mot, n))
    return list(tokens) + fragments
