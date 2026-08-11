"""Banc d'ÉCHELLE de la mémoire de croyance — répond à la seule question qui compte pour la
production : est-ce que ça tient à 1 k / 10 k / 50 k faits, en latence ET en mémoire, ou est-ce
qu'une structure O(n²) cachée fait exploser le coût ?

On mesure trois choses, sans complaisance :
  1. temps de CONSTRUCTION (asserter N faits) — révèle le coût amorti d'un assert ;
  2. MÉMOIRE — objets Python (tracemalloc, portable) + taille réelle des faisceaux VSA ;
  3. latence de LECTURE `courant()` — médiane et p95 sur un échantillon d'emplacements.

Répartition RÉALISTE : beaucoup d'entités (chantiers/clients), peu d'attributs par entité,
quelques mises à jour par emplacement — plus un emplacement à HISTORIQUE LONG pour débusquer le
coût quadratique du `_recompute` intégral à chaque assert.
"""

import statistics
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mosaic.croyance import MemoireCroyance

ATTRIBUTS = ["etat", "responsable", "phase", "materiel", "priorite"]
VALEURS = {
    "etat": ["en_cours", "termine", "en_attente", "bloque", "recupere"],
    "responsable": ["marie", "paul", "lea", "hugo", "sous_traitant"],
    "phase": ["etude", "chantier", "reception", "sav", "cloture"],
    "materiel": ["fabricanta", "schneider", "legrand", "abb", "mixte"],
    "priorite": ["basse", "normale", "haute", "urgente"],
}


def _construire(n_faits: int) -> tuple[MemoireCroyance, float]:
    """Assère n_faits répartis sur ~n_faits/5 entités × 5 attributs (≈1 màj/emplacement).
    Déterministe (aucun aléa) : le fait i choisit sa valeur par un simple modulo."""
    m = MemoireCroyance()
    n_entites = max(1, n_faits // 5)
    t0 = time.perf_counter()
    for i in range(n_faits):
        ent = f"CHT{i % n_entites:05d}"
        attr = ATTRIBUTS[i % len(ATTRIBUTS)]
        vals = VALEURS[attr]
        val = vals[(i // n_entites) % len(vals)]
        m.asserter(ent, attr, val)
    return m, time.perf_counter() - t0


def _latence_lecture(
    m: MemoireCroyance, n_entites: int, echantillon: int
) -> list[float]:
    """Latence de courant() sur un échantillon déterministe d'emplacements existants (ms)."""
    lats = []
    pas = max(1, n_entites // echantillon)
    for k in range(0, n_entites, pas):
        ent = f"CHT{k:05d}"
        attr = ATTRIBUTS[k % len(ATTRIBUTS)]
        t0 = time.perf_counter()
        m.courant(ent, attr)
        lats.append((time.perf_counter() - t0) * 1000.0)
    return lats


def _octets_faisceaux(m: MemoireCroyance) -> int:
    """Taille réelle des faisceaux VSA en RAM (float64 × dim × nb d'emplacements)."""
    return sum(a.nbytes for a in m._acc.values())


def _historique_long(profondeur: int) -> float:
    """Coût de construction d'un SEUL emplacement à historique très long — sonde directement
    le quadratique du _recompute intégral. Retourne le temps total (s)."""
    m = MemoireCroyance()
    t0 = time.perf_counter()
    for i in range(profondeur):
        m.asserter("CHT_CHAUD", "etat", ["en_cours", "en_attente"][i % 2], t=float(i))
    return time.perf_counter() - t0


def main() -> None:
    print("=== BANC D'ÉCHELLE — mémoire de croyance ===\n")
    for n in (1_000, 10_000, 50_000):
        tracemalloc.start()
        m, t_build = _construire(n)
        _, pic = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        n_slots = len(m._acc)
        lats = _latence_lecture(m, n // 5, echantillon=200)
        print(f"--- {n:,} faits ({n_slots:,} emplacements) ---".replace(",", " "))
        print(
            f"  construction : {t_build:.2f} s  ({n / t_build:,.0f} assert/s)".replace(
                ",", " "
            )
        )
        print(f"  assert amorti : {t_build / n * 1000:.3f} ms/fait")
        print(
            f"  lecture courant() : médiane {statistics.median(lats):.3f} ms, "
            f"p95 {sorted(lats)[int(len(lats) * 0.95)]:.3f} ms"
        )
        print(f"  RAM objets Python (pic) : {pic / 1e6:.1f} Mo")
        print(
            f"  faisceaux VSA : {_octets_faisceaux(m) / 1e6:.1f} Mo "
            f"({n_slots} × {m.dim} × 8 o)\n"
        )

    print("--- SONDE QUADRATIQUE : un emplacement à historique long ---")
    for prof in (100, 500, 2_000):
        t = _historique_long(prof)
        print(
            f"  {prof:>5} asserts sur 1 slot : {t:.3f} s  ({t / prof * 1000:.3f} ms/assert)"
        )
    print(
        "\n(si le ms/assert croît ~linéairement avec la profondeur → _recompute est O(n) par "
        "assert, donc O(n²) pour bâtir l'historique : cible d'optimisation si un slot dépasse "
        "quelques milliers de faits.)"
    )


if __name__ == "__main__":
    main()
