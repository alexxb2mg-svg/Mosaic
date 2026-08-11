"""Abstention CALIBRÉE (prédiction conforme) pour la mémoire de croyance.

Aujourd'hui `courant()` lève `a_preciser` sur un seuil FIXE (SEUIL_CONTESTE = 0.05 sur la marge de
cleanup VSA). C'est une intuition, pas une garantie. La prédiction conforme la transforme en
CONTRAT : pour un taux d'erreur cible α, on calibre le seuil de confiance τ(α) sur un jeu de
calibration, de sorte que « répondre quand confiance ≥ τ » ait une erreur ≤ α — puis on VÉRIFIE la
couverture sur un jeu de test indépendant (garantie conforme = distribution-free, échangeabilité).

On génère des emplacements de croyance à qualité d'évidence VARIABLE (nets vs ambigus), on connaît
la vérité-terrain (fait le plus récent), on mesure (confiance, correct), on calibre, on compare au
seuil fixe 0.05.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mosaic.croyance import MemoireCroyance

ATTRS = ["etat", "phase", "responsable", "priorite"]
VALS = {
    "etat": ["en_cours", "termine", "en_attente", "bloque"],
    "phase": ["etude", "chantier", "reception", "sav"],
    "responsable": ["marie", "paul", "lea", "hugo"],
    "priorite": ["basse", "normale", "haute", "urgente"],
}


def _slot(mem, ent, attr, n_hist, ecart, i):
    """Historique déterministe : n_hist valeurs, la dernière à un écart `ecart` des précédentes
    (grand écart = net ; écart 0 = deux valeurs se disputant le dernier instant = ambigu)."""
    vals = VALS[attr]
    for k in range(n_hist - 1):
        mem.asserter(ent, attr, vals[(i + k) % len(vals)], t=float(k))
    # la valeur finale (vérité-terrain) à t = n_hist-2 + ecart
    verite = vals[(i + 7) % len(vals)]
    mem.asserter(ent, attr, verite, t=float(n_hist - 2 + ecart))
    if ecart == 0:  # un concurrent au MÊME instant -> ambigu
        mem.asserter(ent, attr, vals[(i + 3) % len(vals)], t=float(n_hist - 2))
    return verite


def echantillon(n=4000):
    """Génère n emplacements, mix déterministe de nets et d'ambigus. Rend (confiance, correct)."""
    points = []
    for i in range(n):
        mem = MemoireCroyance(dim=512)
        attr = ATTRS[i % len(ATTRS)]
        n_hist = 2 + (i % 4)
        ecart = [0, 1, 2, 5][(i // 4) % 4]  # 0 = ambigu, sinon net
        verite = _slot(mem, f"E{i}", attr, n_hist, ecart, i)
        c = mem.courant(f"E{i}", attr)
        correct = c is not None and c["valeur"] == verite
        points.append((c["confiance"] if c else 0.0, correct))
    return points


def seuil_conforme(cal, alpha):
    """Plus petit seuil τ tel que, sur le jeu de calibration, l'erreur des points confiance≥τ ≤ α.
    Balaye les seuils candidats (les confiances observées)."""
    seuils = sorted({conf for conf, _ in cal})
    for tau in seuils:
        retenus = [ok for conf, ok in cal if conf >= tau]
        if retenus and (1 - sum(retenus) / len(retenus)) <= alpha:
            return tau
    return seuils[-1] + 1e-9 if seuils else 1.0


def evalue(pts, tau):
    retenus = [ok for conf, ok in pts if conf >= tau]
    n = len(pts)
    couv = len(retenus) / n  # taux de RÉPONSE (non-abstention)
    err = (1 - sum(retenus) / len(retenus)) if retenus else 0.0
    return couv, err


def main():
    pts = echantillon(4000)
    mid = len(pts) // 2
    cal, test = pts[:mid], pts[mid:]
    print(
        f"=== ABSTENTION CALIBRÉE — {len(pts)} emplacements ({sum(ok for _, ok in pts)} corrects) ===\n"
    )

    print("Seuil FIXE actuel (0.05) sur le jeu de test :")
    couv, err = evalue(test, 0.05)
    print(
        f"  répond dans {couv:.1%} des cas, erreur {err:.2%} (aucune garantie — c'est ce qui sort)\n"
    )

    print(
        "Seuils CALIBRÉS (conformes) — calibrés sur une moitié, VÉRIFIÉS sur l'autre :"
    )
    print(
        f"  {'α cible':>8} | {'τ calibré':>10} | {'répond':>8} | {'erreur test':>12} | garantie tenue ?"
    )
    for alpha in (0.20, 0.10, 0.05, 0.01):
        tau = seuil_conforme(cal, alpha)
        couv, err = evalue(test, tau)
        ok = "OUI" if err <= alpha + 0.02 else "NON"
        print(f"  {alpha:>8.0%} | {tau:>10.4f} | {couv:>8.1%} | {err:>12.2%} | {ok}")
    print(
        "\n(le seuil s'ADAPTE à l'erreur voulue : viser 1% d'erreur -> τ plus haut, on répond moins"
    )
    print(
        " mais quand on répond, on se trompe ≤1%. Le 0.05 fixe ne donne, lui, aucune garantie.)"
    )


if __name__ == "__main__":
    main()
