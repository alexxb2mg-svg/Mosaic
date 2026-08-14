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

import gzip
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

VARIABLE = "MOSAIC_JOURNAL"
# Le PID sépare les appelants : un serveur MCP, une session CLI et un banc sont trois
# processus distincts. Sans lui, deux agents qui cherchent en même temps auraient l'air
# de se reformuler l'un l'autre — et le signal le plus utile deviendrait le plus faux.
_PID = os.getpid()
# Le PID du PARENT dit autre chose, et c'est ce qui manquait : il identifie la SESSION.
# Un agent ne lance pas un processus, il en lance plusieurs — ce moteur de recherche et,
# à côté, l'outil qui ouvrira le document trouvé. Ces deux-là ont des PID différents mais
# le même parent. Le noter ici est ce qui permet, plus tard et sans rien coordonner, de
# rapprocher « cette question a été posée » de « ce document a ensuite été ouvert » : le
# jugement de pertinence que personne n'a eu à écrire.
_PPID = os.getppid()


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
        "ppid": _PPID,
        "index": index,
        "q": requete,
        "k": k,
        "opts": options,
        "hits": resultats,
    }


VARIABLE_MAX_MO = "MOSAIC_JOURNAL_MAX_MO"
MAX_MO_DEFAUT = 5.0


def _pivoter(chemin: Path) -> None:
    """Archive le journal COMPRESSÉ et le laisse repartir à vide, au-delà du seuil.

    Rotation par TAILLE, pas par date : c'est le volume qui gêne, et une semaine
    chargée ne pèse pas comme une semaine calme. Le seuil (`MOSAIC_JOURNAL_MAX_MO`,
    5 Mo par défaut) laisse plusieurs semaines d'usage soutenu dans le fichier
    actif — une ligne pèse ~1,2 ko.

    ARCHIVER, jamais SUPPRIMER. Mesuré sur un journal réel : la compression rend
    un facteur **3,7** (pas dix — les lignes portent des chemins longs et variés,
    peu redondants). Soit, à 200 recherches par jour, 79 Mo par an bruts et
    21 Mo archivés, avec une rotation toutes les trois semaines environ. C'est le
    prix d'une vérité terrain que personne n'a eu à annoter : la jeter pour gagner
    vingt mégaoctets serait un mauvais échange."""
    try:
        seuil = float(os.environ.get(VARIABLE_MAX_MO, MAX_MO_DEFAUT))
    except ValueError:
        return  # seuil illisible : on ne pivote pas plutôt que de deviner
    if seuil <= 0 or not chemin.exists() or chemin.stat().st_size < seuil * 1024 * 1024:
        return
    horodatage = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    archive = chemin.with_name(f"{chemin.stem}-{horodatage}{chemin.suffix}.gz")
    try:
        with (
            chemin.open("rb") as src,
            gzip.open(archive, "wb") as dst,
        ):
            shutil.copyfileobj(src, dst)
        chemin.unlink()
    except OSError:
        # Disque plein, fichier verrouillé : on garde le journal tel quel. Un
        # journal trop gros est un inconvénient ; une recherche cassée est un bug.
        return


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
        _pivoter(chemin)
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
