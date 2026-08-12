"""Décomposition du temps de build — Python pur vs BLAS, mesurée pas estimée.

QUESTION (exercice de pensée 12/08 : « et si c'était écrit dans un autre
langage ? ») : quelle part du build est du Python interprété (boucles learn/
finalize, tokenisation, canonicalisation — ce qu'un langage compilé accélère)
et quelle part est déjà du BLAS/C (SVD, encodages — insensible au langage) ?
C'est LE chiffre qui arbitre « descendre un noyau en natif » vs « rien à
gagner » : la loi d'Amdahl fait le reste.

MÉTHODE : chronométrage direct des composants par enveloppement (perf_counter
autour des vraies fonctions, sur un vrai build) — pas d'échantillonnage, pas
de dépendance, pas de distorsion d'instrumentation mesurable (l'enveloppe ne
s'exécute qu'une fois par appel de haut niveau ; tokenize/canonicalize sont
appelés une fois par document, learn une fois par document, smooth une fois).

Composants PYTHON PUR (un langage compilé les accélérerait 10-50×) :
  lecture+tokenisation, canonicalisation+collocations, Profiles.learn
  (fenêtrage des cooccurrences), Profiles.finalize (itération des paires).
Composants BLAS/NUMPY (déjà en C/Fortran, langage indifférent) :
  smooth (SVD randomisée), encode (superpositions + quantification).

Usage : python research/profil_build.py <corpus> <index_out>

VERDICT MESURÉ (Alloprof 2 556 docs / 50k vocab, 119,2 s total, 12/08) :
  finalize (itération des paires)   49,1 s   41 %   <- LE monstre, Python pur
  encodage (numpy)                  40,3 s   34 %   <- petites ops numpy : borné
                                                       par le surcoût d'appel,
                                                       pas par le calcul
  lissage SVD (BLAS)                15,6 s   13 %   <- le seul vrai plancher
  learn (fenêtrage)                  3,5 s    3 %   <- hypothèse « learn domine »
                                                       FALSIFIÉE (les tableaux
                                                       l'ont rendu bon marché)
  tokenisation+canon+colloc+lecture  3,8 s    3 %
  écritures                          3,0 s    3 %

Lecture Amdahl : un noyau natif (ou une restructuration) de finalize + encode
ramènerait le build à ~25-35 s (×3,5-4,5) — le plafond utile d'une réécriture
totale. Piste algorithmique notée SANS la coder : les signatures n'ont que 40
composantes non nulles sur 12 288 — finalize fait aujourd'hui deux additions
DENSES par paire ; un scatter-add épars diviserait les opérations par ~300,
même en Python… mais les poids PPMI sont des logs (non entiers) : changer
l'ordre des sommes change les bits -> c'est un CHANGEMENT DE VERSION avec
rebuild général, jamais une optimisation silencieuse. Décision à prendre le
jour où le temps de build comptera vraiment.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosaic import index as index_module
from mosaic.index import Index
from mosaic.profiles import Profiles

TEMPS: dict[str, float] = {}


def _chrono_attr(porteur, nom: str, cle: str) -> None:
    orig = getattr(porteur, nom)

    def enveloppe(*args, **kwargs):
        t0 = time.perf_counter()
        r = orig(*args, **kwargs)
        TEMPS[cle] = TEMPS.get(cle, 0.0) + (time.perf_counter() - t0)
        return r

    setattr(porteur, nom, enveloppe)


def main() -> int:
    corpus, sortie = Path(sys.argv[1]), Path(sys.argv[2])

    # Python pur — les cibles telles qu'importées dans le namespace de index.py
    _chrono_attr(index_module, "_read_text", "lecture")
    _chrono_attr(index_module, "tokenize", "tokenisation")
    _chrono_attr(index_module, "canonicalize", "canonicalisation")
    _chrono_attr(index_module, "detect", "collocations")
    _chrono_attr(Profiles, "learn", "learn (fenetrage cooccurrences)")
    _chrono_attr(Profiles, "finalize", "finalize (iteration des paires)")
    # BLAS/numpy
    _chrono_attr(index_module, "smooth", "lissage SVD (BLAS)")
    _chrono_attr(index_module, "encode", "encodage (numpy)")
    # persistance
    _chrono_attr(index_module, "save_vocab", "ecriture vocab")
    _chrono_attr(index_module, "save_docs", "ecriture docs")

    t0 = time.perf_counter()
    idx = Index.build(corpus, sortie, index_paths=False)
    total = time.perf_counter() - t0

    mesure = sum(TEMPS.values())
    python_pur = sum(
        TEMPS.get(k, 0.0)
        for k in (
            "lecture",
            "tokenisation",
            "canonicalisation",
            "collocations",
            "learn (fenetrage cooccurrences)",
            "finalize (iteration des paires)",
        )
    )
    blas = TEMPS.get("lissage SVD (BLAS)", 0.0) + TEMPS.get("encodage (numpy)", 0.0)
    rapport = {
        "docs": len(idx.ids),
        "vocab": len(idx.profiles.rows),
        "total_s": round(total, 1),
        "composants_s": {k: round(v, 1) for k, v in sorted(TEMPS.items())},
        "python_pur_s": round(python_pur, 1),
        "python_pur_pct": round(100 * python_pur / total, 1),
        "blas_numpy_s": round(blas, 1),
        "blas_numpy_pct": round(100 * blas / total, 1),
        "reste_s": round(total - mesure, 1),
    }
    print(json.dumps(rapport, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
