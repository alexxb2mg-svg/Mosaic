"""Confiance de requête — le moteur a-t-il TROUVÉ, ou a-t-il comblé ?

Une recherche rend toujours dix documents. Elle ne dit jamais si le premier est
une réponse ou le moins mauvais d'un corpus qui n'en contenait pas. C'est le trou
qu'a exposé la mesure du 14/08 : une même question, décorée de six mots plausibles
mais absents des documents, passe du rang 2 au rang 231 — et rien, dans la
réponse, ne signalait que la seconde formulation était mal servie.

La prédiction de performance de requête (QPP) comble ce trou SANS RIEN AJOUTER au
calcul : elle lit la FORME de la distribution des scores que le moteur vient de
produire. L'intuition tient en une phrase — quand le haut du classement se détache
nettement du fond, la requête est bien servie ; quand tout se vaut, le moteur a
ordonné du bruit.

`nqc` implémente **Normalized Query Commitment** (Shtok, Kurland, Carmel, Raiber,
Markovits, *Predicting Query Performance by Query-Drift Estimation*, TOIS 30(2),
2012, DOI 10.1145/2180868.2180873) : écart-type des scores du top-k, normalisé par
le score moyen du corpus entier.

DEUX LIMITES, à garder sous les yeux :

1. **C'est un signal de TRI, jamais une mesure.** Les corrélations publiées sont
   modérées (Kendall autour de 0,4) et varient fortement selon la collection. Il
   sert à comparer deux requêtes sur un même corpus, à déclencher une alerte ou à
   conditionner un traitement — jamais à annoncer un taux de réussite.
2. **Sur requêtes courtes, il peut mesurer la LONGUEUR plutôt que la difficulté**
   (rupture nommée par la littérature). C'est au banc de le vérifier sur nos
   corpus, pas à ce module de le supposer : `research/qpp_precision.py`.

VERDICT DE SON BANC (14/08, `research/qpp_precision.py`, 2 316 requêtes
annotées) : **ÉCARTÉ — il ne prédit pas la difficulté sur ce moteur.**

    canal      corr. succès   corr. longueur
    grille        +0,0412        −0,0802
    bm25          +0,1470        −0,1599
    embed         +0,0335        −0,1434

Le meilleur canal reste sous le seuil déclaré (0,15) et, surtout, **sa
corrélation avec la LONGUEUR de la requête dépasse celle avec le succès** : la
rupture annoncée par la littérature s'est réalisée telle quelle. Ce que NQC
mesure ici, c'est qu'une requête courte produit des scores plus dispersés — pas
qu'elle est mieux servie.

LEÇON DE MÉTHODE, plus utile que le verdict : sur un échantillon de 150 requêtes,
le même banc donnait +0,1764 sur la grille et concluait « adopté ». Sur les
2 316, la grille tombe à +0,0412 — l'échantillon était trompeur d'un facteur
quatre. Un banc court sert à vérifier qu'un protocole tourne, jamais à trancher.

Le module reste dans le dépôt : il est correct, c'est son pouvoir prédictif qui
est nul SUR CE CORPUS. Une variante non testée demeure ouverte — normaliser par
le nombre de termes actifs de la requête, correction que la littérature nomme —
mais elle exige de rejouer le banc entier, et rien ne dit qu'elle suffira.

Ce module ne décide rien. Il rend un nombre, et c'est l'appelant — l'aiguilleur,
le serveur, un banc — qui choisit quoi en faire.
"""

from __future__ import annotations

import numpy as np


def nqc(scores: np.ndarray, k: int = 10) -> float:
    """Confiance NQC pour une requête, d'après TOUS les scores du corpus.

    `scores` : le score de CHAQUE document pour cette requête (c'est ce que
    produit déjà chaque canal). `k` : profondeur du haut de classement examiné.

    Rend un réel positif — grand = le haut se détache, petit = tout se vaut.
    Invariant d'échelle : multiplier tous les scores par dix ne change rien, sinon
    deux canaux aux échelles différentes seraient incomparables.

    Rend 0.0 sur un corpus vide ou des scores tous nuls : l'absence de signal est
    une confiance nulle, jamais une division par zéro."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        return 0.0
    moyenne_corpus = float(np.mean(s))
    if moyenne_corpus == 0.0:
        return 0.0
    haut = np.sort(s)[::-1][: max(1, min(k, s.size))]
    return float(np.std(haut) / abs(moyenne_corpus))
