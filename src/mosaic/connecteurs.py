"""Algèbre de requête par connecteurs — découpe une requête en mots VOULUS (+) et EXCLUS (−).

« et / ou / avec » = renforcer (garder le signe), « sans / pas / ni » = soustraire, « mais pas »
= contraste négatif, « mais » seul = re-positive. Le score de recherche devient alors
cos(doc, positif) − λ·cos(doc, négatif) : ce qu'on exclut fait DESCENDRE le classement.

Module de FONDATION : stdlib uniquement, déterministe, n'importe aucun autre module Mosaic.
La portée (quoi est nié) est réglée par des règles simples — robuste sur une requête courte et
explicite (l'usage réel : un agent ou l'utilisateur écrit les connecteurs), pas sur de la prose.
"""

NEG_MARQUEURS = {"sans", "pas", "ni", "sauf", "hormis", "excepté", "excepte"}
POS_MARQUEURS = {"et", "ou", "avec", "plus"}


def analyser(text: str) -> list[tuple[str, int]]:
    """Renvoie [(mot, signe)] pour chaque mot plein : signe +1 (voulu) ou −1 (exclu)."""
    toks = text.lower().replace(",", " ").split()
    signe = 1
    out: list[tuple[str, int]] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "mais":
            if i + 1 < len(toks) and toks[i + 1] in NEG_MARQUEURS:
                signe = -1
                i += 2
                continue
            signe = 1
        elif t in NEG_MARQUEURS:
            signe = -1
        elif t in POS_MARQUEURS:
            pass  # conjonction : garde le signe courant
        else:
            out.append((t, signe))
        i += 1
    return out


def decouper(text: str) -> tuple[str, str]:
    """Renvoie (texte_positif, texte_négatif) — les mots voulus et les mots exclus, réassemblés
    en deux sous-requêtes que le pipeline de recherche encode indépendamment."""
    pairs = analyser(text)
    positif = " ".join(mot for mot, signe in pairs if signe > 0)
    negatif = " ".join(mot for mot, signe in pairs if signe < 0)
    return positif, negatif
