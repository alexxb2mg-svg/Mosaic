"""Étape 3 du chantier atlas (briefing #367) — la PYRAMIDE comme préfiltre sous-linéaire.

Sur un atlas ORGANISÉ (tokens voisins par le sens → cellules voisines), réduire la
carte (somme par blocs 64→32→16) conserve le signal sémantique localisé — c'est
l'opération que la permutation (#366) rend absurde sur la grille SHA : sommer des
cellules arbitraires n'agrège que du bruit. Si la carte grossière suffit à présélectionner
les candidats, le score plein-format ne se paye que sur eux : préfiltre sous-linéaire.

CRITÈRE DU BRIEFING (déclaré avant mesure) : >=5× de calcul gagné pour <1 pt de
recall@10 perdu (vs le scan plein format du canal atlas seul).

PROTOCOLE : échantillon Alloprof 500 docs / 300 requêtes (conventions des étapes 1-2) ;
cartes de chaleur tf×idf sur SOM 64×64 (σ=0). Baseline = cosinus plein corpus à 4 096 d.
Pyramide = classement grossier à 16×16 (256 d, somme par blocs 4×4) → top-C candidats →
re-score plein format des seuls candidats. Balayage C ∈ {25, 50, 100, 200}. Coût compté
en PRODUITS SCALAIRES pondérés par la dimension (N·256 + C·4 096 vs N·4 096) + chrono
mural indicatif. Contrôle : la même pyramide sur un atlas PERMUTÉ (cellules mélangées,
graine fixe) doit s'effondrer — c'est la preuve que le gain vient de l'ORGANISATION.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os

os.environ.setdefault("MOSAIC_ATLAS", "64")
_ARGS = sys.argv[1:3]
sys.argv = sys.argv[:1]  # atlas_som lit argv à l'import

import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

from atlas_som import (  # noqa: E402
    ATLAS,
    DIM_SOM,
    ECHANTILLON_DOCS,
    GRID_BUILD,
    MAX_REQUETES,
    _carte,
    _projeter,
    _som,
)
from mosaic.calibration import _preparer  # noqa: E402
from mosaic.collocations import merge  # noqa: E402
from mosaic.lexicon import canonicalize, compile_lexicon  # noqa: E402
from mosaic.tokenize import tokenize  # noqa: E402

COTE_GROSSIER = 16  # carte du préfiltre (256 dims)
CANDIDATS = [25, 50, 100, 200]
GRAINE_PERMUTATION = 0x20260812


def _pool(cartes: np.ndarray, cote_cible: int) -> np.ndarray:
    """Somme par blocs (N, H, W) -> (N, cote_cible, cote_cible)."""
    n, h, _w = cartes.shape
    f = h // cote_cible
    return cartes.reshape(n, cote_cible, f, cote_cible, f).sum(axis=(2, 4))


def _metriques(classements: list[list[str]], verites: list[list[str]]) -> float:
    recalls = []
    for cls, rel in zip(classements, verites):
        rel_set = set(rel)
        recalls.append(len(rel_set & set(cls[:10])) / max(1, len(rel_set)))
    return round(sum(recalls) / max(1, len(classements)), 4)


def _cos_lignes(mat: np.ndarray, q: np.ndarray) -> np.ndarray:
    normes = np.linalg.norm(mat, axis=1)
    normes[normes == 0] = 1.0
    nq = np.linalg.norm(q)
    return (mat @ q) / (normes * (nq if nq else 1.0))


def main() -> int:
    if len(_ARGS) < 2:
        raise SystemExit(
            "usage : python research/atlas_pyramide.py <corpus> <verite.jsonl>"
        )
    corpus_src, verite_path = Path(_ARGS[0]), Path(_ARGS[1])
    requetes = [
        (str(o["query"]), [str(x) for x in o["relevant"]])
        for o in (
            json.loads(line)
            for line in verite_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    fichiers = sorted(p for p in corpus_src.iterdir() if p.is_file())
    dossier = tempfile.TemporaryDirectory()
    corpus = corpus_src
    if len(fichiers) > ECHANTILLON_DOCS:
        corpus = Path(dossier.name) / "corpus"
        corpus.mkdir()
        garde = fichiers[:ECHANTILLON_DOCS]
        for p in garde:
            (corpus / p.name).write_bytes(p.read_bytes())
        noms = {p.name for p in garde}
        requetes = [(q, rel) for q, rel in requetes if all(r in noms for r in rel)][
            :MAX_REQUETES
        ]
    verites = [rel for _q, rel in requetes]

    docs, profiles, colloc, lexicon = _preparer(corpus, None, 300, GRID_BUILD)
    ids = [d for d, _ in docs]
    n = len(ids)
    compiled = compile_lexicon(lexicon)

    def tokens_requete(q: str) -> list[str]:
        return merge(merge(canonicalize(tokenize(q), compiled), colloc), colloc)

    v = len(profiles.rows)
    features = _projeter(profiles.acc[:v], DIM_SOM)
    bmu = _som(features)
    h, w = ATLAS
    print(
        f"corpus {corpus_src.name} : {n} docs, {len(requetes)} requêtes, "
        f"atlas {h}x{w} -> préfiltre {COTE_GROSSIER}x{COTE_GROSSIER}"
    )

    # deux assignations : organisée (SOM) et PERMUTÉE (contrôle — mêmes cellules,
    # positions mélangées : détruit l'organisation, conserve les collisions)
    rng = np.random.default_rng(GRAINE_PERMUTATION)
    melange = rng.permutation(h * w)
    variantes = {
        "organisé (SOM)": {t: int(bmu[i]) for t, i in profiles.rows.items()},
        "permuté (contrôle)": {
            t: int(melange[int(bmu[i])]) for t, i in profiles.rows.items()
        },
    }

    dim_plein = h * w
    dim_gros = COTE_GROSSIER * COTE_GROSSIER
    for nom_var, position in variantes.items():
        cartes = np.stack(
            [_carte(toks, position, profiles.idf, h, w) for _d, toks in docs]
        )
        plein = cartes.reshape(n, -1)
        gros = _pool(cartes, COTE_GROSSIER).reshape(n, -1)

        cartes_q, gros_q = [], []
        for q, _rel in requetes:
            cq = _carte(tokens_requete(q), position, profiles.idf, h, w)
            cartes_q.append(cq.ravel())
            gros_q.append(_pool(cq[None, :, :], COTE_GROSSIER).ravel())

        # baseline : scan plein format
        t0 = time.perf_counter()
        cls_plein = []
        for cq in cartes_q:
            scores = _cos_lignes(plein, cq)
            cls_plein.append([ids[i] for i in np.argsort(-scores, kind="stable")[:10]])
        t_plein = time.perf_counter() - t0
        base = _metriques(cls_plein, verites)
        print(
            f"\n--- {nom_var} — baseline plein format : R@10 {base} "
            f"({t_plein * 1000 / len(requetes):.2f} ms/req)"
        )

        for c in CANDIDATS:
            t0 = time.perf_counter()
            cls_pyr = []
            for cq, gq in zip(cartes_q, gros_q):
                pre = np.argpartition(-_cos_lignes(gros, gq), min(c, n) - 1)[:c]
                scores = _cos_lignes(plein[pre], cq)
                ordre = pre[np.argsort(-scores, kind="stable")][:10]
                cls_pyr.append([ids[i] for i in ordre])
            t_pyr = time.perf_counter() - t0
            r = _metriques(cls_pyr, verites)
            flops = (n * dim_gros + c * dim_plein) / (n * dim_plein)
            print(
                f"  pyramide C={c:<4} R@10 {r:<7} (Δ {round((r - base) * 100, 2):>6} pt) "
                f"coût {flops:.3f}x du scan ({1 / flops:.1f}x gagné) "
                f"[{t_pyr * 1000 / len(requetes):.2f} ms/req]"
            )
    print(
        "\nCritère (en-tête) : >=5x gagné à <1 pt de R@10 perdu — sur l'ORGANISÉ "
        "seulement ; le permuté doit s'effondrer (preuve que le gain vient de la SOM)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
