"""Journal des recherches — la vérité terrain que personne n'a le courage d'annoter.

Le problème. Calibrer un index demande de savoir, pour quelques dizaines de questions,
quel document était le bon. Personne n'a ces exemples le jour de l'installation, et
personne n'a envie de les écrire. Résultat : un moteur tourne des mois sur ses réglages
par défaut sans que quiconque puisse dire s'ils lui conviennent — mesuré le 13/08 sur
ArguAna, où les défauts rendent 0.28 là où un autre équilibre de canaux rend 0.61.

L'idée. L'usage produit ce jugement tout seul, à condition de l'écrire. Une ligne par
recherche, et quatre signaux s'en déduisent PLUS TARD, par simple relecture :

  reformulation  deux recherches proches dans le temps, même processus, mots qui se
                 recoupent -> la première a échoué. Et le couple « requête faible ->
                 requête qui a marché » est exactement la matière du prétraitement.
  abandon        une recherche sans reformulation ni suite -> échec silencieux.
  redondance     les rangs par canal sont consignés : si un canal ne remonte jamais
                 rien que les autres n'aient déjà, il ne sert à rien SUR CE CORPUS.
                 Ce signal-là ne demande AUCUNE annotation.
  distribution   la part réelle des questions lexicales et sémantiques. Le banc du
                 13/08 donne 1.00 de rappel sur les premières et 0.33 sur les
                 secondes : savoir laquelle domine décide de ce qu'il faut réparer.

CE MODULE N'ÉCRIT RIEN PAR DÉFAUT. Il faut poser `MOSAIC_JOURNAL=<chemin>` — le dépôt
est public et personne ne doit découvrir un journal qu'il n'a pas demandé. Le fichier
contient les requêtes en clair, donc des données métier : il reste local, hors du
dépôt et hors de tout export.

Une écriture qui échoue n'échoue JAMAIS la recherche : un journal est un observateur,
il n'a pas le droit d'avoir un avis sur le déroulement de ce qu'il observe.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

VARIABLE = "MOSAIC_JOURNAL"
# Le PID sépare les appelants : un serveur MCP, une session CLI et un banc sont trois
# processus distincts. Sans lui, deux agents qui cherchent en même temps auraient l'air
# de se reformuler l'un l'autre — et le signal le plus utile deviendrait le plus faux.
_PID = os.getpid()


def actif() -> Path | None:
    """Le chemin du journal, ou None si la variable n'est pas posée (cas par défaut)."""
    chemin = os.environ.get(VARIABLE, "").strip()
    return Path(chemin) if chemin else None


def _ligne(index: str, requete: str, k: int, options: dict, hits: list[dict]) -> dict:
    # On consigne les RANGS par canal, jamais les scores ni le contenu : les rangs
    # suffisent à mesurer la redondance entre canaux, ils sont comparables d'une
    # requête à l'autre, et ils ne font pas grossir le fichier.
    resultats = [
        {"id": h.get("id"), **({"rangs": h["rangs"]} if "rangs" in h else {})}
        for h in hits
    ]
    return {
        "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pid": _PID,
        "index": index,
        "q": requete,
        "k": k,
        "opts": options,
        "hits": resultats,
    }


def consigner(
    index: str, requete: str, k: int, options: dict, hits: list[dict]
) -> None:
    """Ajoute une ligne au journal si `MOSAIC_JOURNAL` est posée. Sans effet sinon.

    `options` ne doit contenir que les options NON par défaut : une ligne de journal se
    relit à l'œil, et vingt champs à `false` la rendraient illisible."""
    chemin = actif()
    if chemin is None:
        return
    try:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        with chemin.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(_ligne(index, requete, k, options, hits), ensure_ascii=False)
                + "\n"
            )
    except (OSError, TypeError, ValueError):
        # Disque plein, chemin invalide, objet non sérialisable : la recherche a déjà
        # produit son résultat, l'utilisateur doit le recevoir. On perd une ligne de
        # journal, on ne perd pas une réponse.
        return


def lire(chemin: Path) -> list[dict]:
    """Les lignes du journal, celles qui sont lisibles. Une ligne tronquée (écriture
    interrompue) est ignorée plutôt que de faire échouer toute l'analyse."""
    lignes = []
    for li in Path(chemin).read_text(encoding="utf-8").splitlines():
        if not li.strip():
            continue
        try:
            lignes.append(json.loads(li))
        except json.JSONDecodeError:
            continue
    return lignes
