"""Boucle de résolution : quand une croyance est INCERTAINE, aller chercher le contexte qui tranche.

C'est le point de jonction des deux moteurs de Mosaic. La mémoire de croyance (VSA, structurée)
sait dire « je ne sais pas » — `courant()` lève `a_preciser` quand deux valeurs se disputent ou
que les canaux divergent. Cette abstention DÉCLENCHE une recherche sémantique (le moteur Mosaic)
dans le corpus qui fait autorité, pour chaque valeur candidate. La valeur dont la PREUVE est la
plus forte l'emporte et est ré-assérée (plus récente → elle domine).

Garde-fou décisif : si les preuves sont trop proches (marge < seuil), la boucle NE TRANCHE PAS —
elle rend « non tranché », à escalader vers un humain ou un agent. On ne remplace pas une
incertitude par un pari silencieux (cf. docs/criteres_pertinence.md, « cas ambigu → ne pas
deviner »). Le scaffolding est déterministe et sans LLM ; un agent doté d'un LLM n'intervient
qu'au point de JUGEMENT — affiner la formulation d'une valeur, lire une nuance dans le document
de preuve — jamais pour fabriquer la réponse.
"""

from typing import Protocol

from mosaic.croyance import MemoireCroyance

SEUIL_PREUVE_DEFAUT = (
    0.05  # écart de score minimal entre les deux meilleures preuves pour trancher
)


class Chercheur(Protocol):
    """Tout objet exposant `search(text, k) -> [{'id', 'score', ...}]` (un Index Mosaic)."""

    def search(self, text: str, k: int = ...) -> list[dict]: ...


def resoudre(
    mem: MemoireCroyance,
    entite: str,
    attribut: str,
    index: Chercheur,
    formulations: dict[str, str] | None = None,
    seuil_preuve: float = SEUIL_PREUVE_DEFAUT,
    profondeur_preuve: int = 3,
    t: float | None = None,
) -> dict:
    """Tente de résoudre la croyance (entite, attribut) par la preuve documentaire de `index`.

    `formulations` : pour une valeur candidate, la phrase de recherche à employer (défaut : la
    valeur elle-même). Ex. {"termine": "réception PV signé livré"} — traduit un code d'état en
    langage du corpus. `profondeur_preuve` : l'évidence d'une valeur = SOMME des scores de ses
    `profondeur_preuve` meilleurs documents (largeur de corroboration) — un état ancré dans
    plusieurs documents du corpus l'emporte sur un simple phrase-match court et isolé. Retourne un
    rapport `statut` ∈ {inconnu, deja_net, resolu, non_tranche}."""
    c = mem.courant(entite, attribut)
    if c is None:
        return {"statut": "inconnu", "entite": entite, "attribut": attribut}
    if not c["a_preciser"]:
        return {
            "statut": "deja_net",
            "valeur": c["valeur"],
            "confiance": c["confiance"],
        }

    formulations = formulations or {}
    candidats = c.get("candidats") or sorted(mem._vocab.get(attribut, set()))
    # Scores et docs typés séparément du rapport (évite l'inférence union du dict de sortie).
    scores: dict[str, float] = {}
    docs: dict[str, str | None] = {}
    for val in candidats:
        res = index.search(
            f"{entite} {formulations.get(val, val)}", k=profondeur_preuve
        )
        scores[val] = round(
            sum(float(r["score"]) for r in res), 4
        )  # largeur de corroboration
        docs[val] = str(res[0]["id"]) if res else None
    ordre = sorted(candidats, key=lambda v: -scores[v])
    preuves = [{"valeur": v, "score_preuve": scores[v], "doc": docs[v]} for v in ordre]

    meilleur_val = ordre[0]
    score_meilleur = scores[meilleur_val]
    second = scores[ordre[1]] if len(ordre) > 1 else 0.0
    marge = score_meilleur - second
    if score_meilleur <= 0.0 or marge < seuil_preuve:
        return {
            "statut": "non_tranche",
            "raison": "preuves trop proches ou nulles — escalader (ne pas deviner)",
            "marge": round(marge, 4),
            "preuves": preuves,
        }

    if t is None:
        serie = mem._hist.get((entite, attribut), [])
        t = (
            max((tt for tt, _ in serie), default=0.0)
        ) + 1.0  # ré-assertion plus récente
    mem.asserter(entite, attribut, meilleur_val, t=t)
    apres = mem.courant(entite, attribut)
    return {
        "statut": "resolu",
        "valeur": meilleur_val,
        "marge": round(marge, 4),
        "preuve": preuves[0],
        "preuves": preuves,
        "confiance_apres": apres["confiance"] if apres else None,
        "conteste_apres": apres["conteste"] if apres else None,
    }
