"""Vérité temporelle : ne jamais rendre une version PÉRIMÉE comme canonique.

Une recherche plate classe par le SENS. Sur un dossier qui évolue en plusieurs sessions, les
versions successives d'un même aspect (une spéc révisée trois fois) remontent donc MÊLÉES — et
rien ne dit laquelle est à jour ; la plus ancienne peut même être la mieux classée. Un agent qui
saisit le premier résultat prend alors une donnée obsolète pour vérité.

`versions_actuelles` corrige ça sans aucune connaissance métier : on prend le top-k sémantique, on
REGROUPE les documents qui sont des versions d'un même aspect (forte similarité mutuelle entre
eux), puis dans chaque groupe la version la plus RÉCENTE (date extraite du nom de fichier) est
CANONIQUE ; les autres sont marquées PÉRIMÉES. Déterministe, souverain.

Limites assumées : la date doit être lisible dans le chemin (préfixe AAAA-MM-JJ, motif courant des
dossiers chantier) — sinon le document forme son propre groupe (pas de supersession). Le seuil de
regroupement est corpus-dépendant : trop bas, deux aspects proches fusionnent ; trop haut, des
versions ne se rejoignent pas. Défaut 0.55, réglable.
"""

import re
from collections.abc import Callable

import numpy as np

SEUIL_VERSION_DEFAUT = 0.55
# Sentinelle « aucune date connue ». Choisie pour trier naturellement en queue d'un
# classement décroissant par chaîne — les documents sans date tombent derrière, sans
# cas particulier nulle part. Nommée ici parce que c'est ici qu'elle est produite.
SANS_DATE = "0000-00-00"
_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def date_du_chemin(doc_id: str) -> str:
    """Date AAAA-MM-JJ trouvée dans le chemin/nom du document ; SANS_DATE si absente."""
    m = _DATE.search(doc_id.replace("\\", "/").rsplit("/", 1)[-1]) or _DATE.search(
        doc_id
    )
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else SANS_DATE


def versions_actuelles(
    idx,
    texte: str,
    k: int = 10,
    seuil_version: float = SEUIL_VERSION_DEFAUT,
    date_de: Callable[[str], str] = date_du_chemin,
    **opts_recherche,
) -> list[dict]:
    """Renvoie les groupes de versions du top-k, chacun avec sa version CANONIQUE (la plus récente)
    et ses versions PÉRIMÉES. Trié par score du document canonique.

    `opts_recherche` est relayé tel quel à `idx.search` (rerank, type_filtre, recence,
    fusion…) : `actuel` n'est plus un search amputé — un agent qui filtre par type en
    recherche garde son filtre en vérité temporelle (audit CLI 12/08, finding 4)."""
    hits = idx.search(texte, k=k, **opts_recherche)
    if not hits:
        return []
    ids = [h["id"] for h in hits]
    score = {h["id"]: float(h["score"]) for h in hits}
    pos = [idx.ids.index(i) for i in ids]
    mat = idx.mat.astype(np.float64)
    norms = np.linalg.norm(mat, axis=1)

    parent = list(range(len(ids)))

    def trouve(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            cos = float(mat[pos[a]] @ mat[pos[b]]) / (
                norms[pos[a]] * norms[pos[b]] + 1e-9
            )
            if cos >= seuil_version:
                parent[trouve(a)] = trouve(b)

    groupes: dict[int, list[int]] = {}
    for i in range(len(ids)):
        groupes.setdefault(trouve(i), []).append(i)

    out: list[tuple[float, dict]] = []
    for membres in groupes.values():
        membres.sort(key=lambda i: date_de(ids[i]), reverse=True)
        canon = ids[membres[0]]
        sc = float(score[canon])
        out.append(
            (
                sc,
                {
                    "canonique": canon,
                    "date": date_de(canon),
                    "score": round(sc, 4),
                    "perimees": [
                        {"id": ids[i], "date": date_de(ids[i])} for i in membres[1:]
                    ],
                },
            )
        )
    out.sort(key=lambda t: -t[0])
    return [g for _, g in out]
