"""Tokenisation française : minuscules, accents conservés, références techniques intactes."""

import re

STOPWORDS = frozenset(
    """au aux avec ce ces cette dans de des du elle en et eux il ils je la le les
    leur lui ma mais me même mes moi mon ne nos notre nous on ou où par pas plus
    pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous
    y d l j n s t c est sont être avoir a ont très aussi comme si
    à ça cela entre vers chez depuis pendant""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9àâäæçéèêëîïôöœùûüÿ]+(?:-[a-z0-9àâäæçéèêëîïôöœùûüÿ]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
