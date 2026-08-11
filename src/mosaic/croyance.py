"""Mémoire de croyance VSA — faits assertés par les agents, requêtes vérité-courante / historique
/ confiance. Déterministe, souverain, sans LLM.

Un fait = (entité, attribut, valeur, t). Les agents assertent ; la mémoire répond :
  - `courant(entité, attribut)` : la valeur ACTUELLE (le fait le plus récent domine), + une
    CONFIANCE tirée de la géométrie VSA (marge de cleanup) et un drapeau CONTESTÉ ;
  - `historique(entité, attribut)` : toutes les valeurs passées, dans l'ordre.

Substrat : MAP-VSA bipolaire (±1), vecteurs SEEDÉS par SHA-256 → reproductibles au bit près sur
TOUTE machine (entier, aucun flottant BLAS). Partitionné par emplacement (entité, attribut) :
chaque emplacement porte la superposition pondérée par récence de son historique de valeurs ;
la vérité-courante émerge du cleanup, la confiance de la marge. Prouvé au banc (docs/research).

Stockage : les FAITS eux-mêmes (JSONL, source de vérité, historique exact) ; les faisceaux VSA
sont reconstruits déterministiquement au chargement. Compact, inspectable, souverain.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

D_DEFAULT = 1024
GAMMA_DEFAULT = (
    0.6  # pondération de récence : le plus récent a le poids 1, le précédent γ, γ²…
)
SEUIL_CONTESTE = 0.05  # marge de cleanup en-deçà de laquelle on déclare « contesté »


def _vec(symbole: str, dim: int) -> np.ndarray:
    """Hypervecteur bipolaire ±1 DÉTERMINISTE d'un symbole (graine SHA-256). Même symbole →
    même vecteur, sur toute machine (entier pur, pas de flottant)."""
    graine = int.from_bytes(hashlib.sha256(symbole.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(graine)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=dim)


class MemoireCroyance:
    """Mémoire de croyance en mémoire vive. `asserter` pour écrire, `courant`/`historique` pour
    lire, `sauver`/`charger` pour persister. Déterministe : recharger reproduit l'état exact."""

    def __init__(
        self,
        dim: int = D_DEFAULT,
        gamma: float = GAMMA_DEFAULT,
        seuil_conteste: float = SEUIL_CONTESTE,
    ) -> None:
        self.dim = dim
        self.gamma = gamma
        # Seuil de marge en-deçà duquel une croyance est « contestée ». Défaut historique
        # 0.05 (heuristique) ; `calibrer_conteste()` le remplace par un seuil CALIBRÉ
        # (prédiction conforme sur les slots du store — recherche conformal_croyance).
        self.seuil_conteste = seuil_conteste
        self._acc: dict[
            tuple[str, str], np.ndarray
        ] = {}  # (ent,attr) -> faisceau récence
        self._hist: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
        self._vocab: dict[str, set[str]] = defaultdict(set)  # attribut -> valeurs vues

    def asserter(
        self, entite: str, attribut: str, valeur: str, t: float | None = None
    ) -> None:
        """Enregistre un fait. `t` = ordre temporel (défaut : à la suite de l'historique).

        Le faisceau de récence se met à jour EN O(1) dans le cas courant (t monotone croissant,
        dont le défaut t=None) : un nouvel instant plus récent vieillit tout l'historique d'un
        rang (× γ) puis ajoute le nouveau vecteur au poids 1. Repli sur `_recompute` intégral
        seulement pour une insertion hors-ordre (t < instant courant) — exact mais rare. C'est
        ce qui évite le O(n²) de la reconstruction systématique (cf. bench_croyance_echelle)."""
        valeur = str(valeur)
        slot = (entite, attribut)
        serie = self._hist[slot]
        if t is None:
            t = float(len(serie))
        t = float(t)
        t_max_avant = max((tt for tt, _ in serie), default=None)
        serie.append((t, valeur))
        self._vocab[attribut].add(valeur)
        if t_max_avant is not None and t < t_max_avant:
            self._recompute(slot)  # insertion hors-ordre : reconstruction exacte
            return
        vec = _vec(f"V:{attribut}:{valeur}", self.dim).astype(np.float64)
        if t_max_avant is None:
            self._acc[slot] = vec  # premier fait de l'emplacement
        elif t > t_max_avant:
            self._acc[slot] = (
                self.gamma * self._acc[slot] + vec
            )  # nouveau rang le + récent
        else:  # t == t_max_avant : même instant que le sommet -> même poids (1)
            self._acc[slot] = self._acc[slot] + vec

    def _recompute(self, slot: tuple[str, str]) -> None:
        """(Re)construit le faisceau de récence d'un emplacement. Le poids dépend du RANG du t
        (distinct) : le t le plus récent a le poids 1, le précédent γ, γ²… Deux valeurs au MÊME
        t reçoivent donc le même poids — elles se disputent la vérité-courante (→ « contesté »)."""
        serie = self._hist[slot]
        temps = sorted({t for t, _ in serie})
        rang = {
            t: i for i, t in enumerate(temps)
        }  # plus ancien = 0 … plus récent = max
        prof = len(temps)
        acc = np.zeros(self.dim, dtype=np.float64)
        _entite, attribut = slot
        for t, valeur in serie:
            poids = self.gamma ** (prof - 1 - rang[t])
            acc += poids * _vec(f"V:{attribut}:{valeur}", self.dim)
        self._acc[slot] = acc

    def courant(self, entite: str, attribut: str) -> dict | None:
        """Vérité-courante HYBRIDE. None si l'emplacement est inconnu. Sinon :
          - `valeur` : la valeur EXACTE (fait le plus récent) — toujours juste ; None si deux
            valeurs se disputent le même instant le plus récent ;
          - `confiance` : marge de cleanup VSA (nuance graduée) ;
          - `conteste` : vrai si égalité exacte au dernier instant OU marge VSA faible ;
          - `a_preciser` : vrai quand le résultat est incertain (contesté ou les deux canaux
            divergent) — invite à demander du contexte pour trancher ;
          - `candidats` / `message` : les valeurs concurrentes et une suggestion, si incertain.

        Deux canaux qui se recoupent : l'EXACT (structuré) donne la réponse, le VSA la confiance ;
        s'ils divergent, c'est un signal — on ne renvoie pas un pari silencieux, on demande."""
        slot = (entite, attribut)
        serie = self._hist.get(slot)
        if not serie:
            return None

        # Canal EXACT : valeur(s) au t le plus récent.
        t_max = max(t for t, _ in serie)
        distinctes = sorted({v for t, v in serie if t == t_max})
        conteste_exact = len(distinctes) > 1
        valeur_exacte = distinctes[0] if len(distinctes) == 1 else None

        # Canal VSA : valeur par cleanup (recoupement) + confiance graduée.
        candidats = sorted(self._vocab[attribut])
        mat = np.stack(
            [_vec(f"V:{attribut}:{v}", self.dim).astype(np.float64) for v in candidats]
        )
        sims = mat @ self._acc[slot]
        ordre = np.argsort(-sims)
        valeur_vsa = candidats[ordre[0]]
        marge = (
            float(sims[ordre[0]] - (sims[ordre[1]] if len(sims) > 1 else 0.0))
            / self.dim
        )
        conteste_vsa = marge < self.seuil_conteste
        divergence = valeur_exacte is not None and valeur_vsa != valeur_exacte

        res: dict = {
            "valeur": valeur_exacte,
            "confiance": round(marge, 4),
            "conteste": conteste_exact or conteste_vsa,
            "a_preciser": conteste_exact or conteste_vsa or divergence,
        }
        if conteste_exact:
            res["candidats"] = distinctes
            res["message"] = (
                f"Deux valeurs concurrentes au dernier instant : {', '.join(distinctes)}. "
                "Préciser laquelle est à jour."
            )
        elif conteste_vsa:
            res["candidats"] = [candidats[i] for i in ordre[:2]]
            res["message"] = (
                "Croyance peu nette (superposition ambiguë) — donner plus de contexte "
                "pour trancher."
            )
        elif divergence:
            res["message"] = (
                f"Canaux en désaccord : exact={valeur_exacte}, VSA={valeur_vsa}. À vérifier."
            )
        return res

    def calibrer_conteste(self, alpha: float = 0.05, min_slots: int = 30) -> dict:
        """CALIBRATION CONFORME du seuil « contesté » — sur le store lui-même, sans donnée
        externe (recherche conformal_croyance : le seuil fixe 0.05 ne garantit rien ; un
        seuil calibré garantit un taux d'erreur cible).

        La vérité-terrain du canal VSA est DÉJÀ dans le store : le canal EXACT (le fait le
        plus récent) dit la bonne valeur de chaque emplacement. On mesure donc, slot par
        slot, (marge VSA, top-1 VSA == valeur exacte), puis on choisit le plus petit seuil τ
        tel que l'erreur des slots « confiants » (marge ≥ τ) soit ≤ alpha. `self.seuil_conteste`
        est mis à jour ; `courant()` l'utilise dès lors. Déterministe et reproductible (mêmes
        faits → même seuil). Moins de `min_slots` emplacements exploitables -> ValueError
        (une calibration sur trois points serait un mensonge de précision)."""
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha doit être dans ]0, 1[ : {alpha!r}")
        points: list[tuple[float, bool]] = []
        for (entite, attribut), serie in self._hist.items():
            t_max = max(t for t, _ in serie)
            distinctes = sorted({v for t, v in serie if t == t_max})
            if len(distinctes) != 1:
                continue  # égalité exacte : pas de vérité unique, slot inexploitable
            verite = distinctes[0]
            candidats = sorted(self._vocab[attribut])
            if len(candidats) < 2:
                continue  # une seule valeur connue : la marge n'a pas de sens
            mat = np.stack(
                [
                    _vec(f"V:{attribut}:{v}", self.dim).astype(np.float64)
                    for v in candidats
                ]
            )
            sims = mat @ self._acc[(entite, attribut)]
            ordre = np.argsort(-sims)
            marge = float(sims[ordre[0]] - sims[ordre[1]]) / self.dim
            points.append((marge, candidats[int(ordre[0])] == verite))
        if len(points) < min_slots:
            raise ValueError(
                f"calibration non fiable : {len(points)} emplacements exploitables "
                f"(minimum {min_slots}) — accumuler plus de faits d'abord"
            )
        seuils = sorted({m for m, _ in points})
        tau = None
        for candidat in seuils:
            retenus = [ok for m, ok in points if m >= candidat]
            if retenus and (1 - sum(retenus) / len(retenus)) <= alpha:
                tau = candidat
                break
        if tau is None:
            tau = (
                seuils[-1] + 1e-9
            )  # aucun niveau ne garantit alpha : tout est incertain
        avant = self.seuil_conteste
        tau = round(tau, 9)  # même valeur stockée et rapportée (cohérence stricte)
        self.seuil_conteste = tau
        retenus = [ok for m, ok in points if m >= tau]
        return {
            "alpha": alpha,
            "seuil_avant": round(avant, 9),
            "seuil_calibre": tau,
            "slots_exploites": len(points),
            "taux_reponse": round(len(retenus) / len(points), 4),
            "erreur_sur_retenus": round(
                (1 - sum(retenus) / len(retenus)) if retenus else 0.0, 4
            ),
        }

    def historique(self, entite: str, attribut: str) -> list[dict]:
        """Toutes les valeurs passées de l'emplacement, ordre chronologique (t croissant)."""
        serie = sorted(self._hist.get((entite, attribut), []), key=lambda x: x[0])
        return [{"t": t, "valeur": v} for t, v in serie]

    def sauver(self, chemin: Path) -> None:
        """Persiste les FAITS (JSONL) — source de vérité ; les faisceaux se reconstruisent."""
        chemin = Path(chemin)
        lignes = []
        for (entite, attribut), serie in self._hist.items():
            for t, valeur in serie:
                lignes.append(
                    json.dumps(
                        {
                            "entite": entite,
                            "attribut": attribut,
                            "valeur": valeur,
                            "t": t,
                        },
                        ensure_ascii=False,
                    )
                )
        tmp = chemin.with_suffix(chemin.suffix + ".tmp")
        tmp.write_text("\n".join(lignes) + ("\n" if lignes else ""), encoding="utf-8")
        tmp.replace(chemin)  # écriture atomique

    @classmethod
    def charger(
        cls, chemin: Path, dim: int = D_DEFAULT, gamma: float = GAMMA_DEFAULT
    ) -> "MemoireCroyance":
        m = cls(dim=dim, gamma=gamma)
        for ligne in Path(chemin).read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                f = json.loads(ligne)
                m.asserter(f["entite"], f["attribut"], f["valeur"], t=f.get("t"))
        return m
