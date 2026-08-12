"""Étape 2 du chantier atlas sémantique (briefing #367) — le COÛT en capacité de superposition.

LA QUESTION CENTRALE DU BRIEFING : l'assignation apprise (tokens voisins par le sens →
cellules voisines) corrèle les supports des signatures — or la capacité de superposition
(50+ tokens tenus sans bouillie) repose précisément sur leur indépendance (Kanerva a
choisi l'aléatoire POUR ça). Combien la localité coûte-t-elle, exactement ?

MÉCANISME TESTÉ : le canal signatures seul, isolé (pas de cooc, pas d'embeddings, pas de
tf-idf) — superposer K signatures ternaires et retrouver les K membres par cosinus contre
tout le vocabulaire (précision@K, même esprit que bench/horizon/capacite_deliaison.py).

ASSIGNATIONS COMPARÉES (même densité 20/+1 20/−1, même graine SHA par token — seule la
LOCALITÉ du support change) :
- sha (l'état actuel) : support uniforme sur les 12 288 dims — la vraie signature() ;
- atlas W : support tiré dans la fenêtre W×W de cellules autour de la cellule SOM du
  token (3 dims/cellule, grille 64×64×3, SOM 64×64 apprise sur les profils réels du
  corpus recettes — la même machinerie que l'étape 1, MOSAIC_ATLAS=64).

JEUX ADVERSES ET CONTRÔLE :
- cohérents : les listes de tokens RÉELLES des documents (les tokens d'un même doc
  cooccurrent, donc la SOM les rapproche — c'est exactement le cas où les supports se
  chevauchent : le pire cas structurel, et le cas d'usage réel) ;
- aléatoires : K tokens uniformes du même vocabulaire (contrôle : sans corrélation
  sémantique, l'atlas ne devrait presque rien coûter).

PRÉDICTIONS DÉCLARÉES AVANT MESURE (falsifiables) :
- P1 : sha ≈ 1.0 jusqu'à K=60 sur les deux jeux (l'indépendance paie) ;
- P2 : atlas perd d'abord sur les jeux COHÉRENTS, et d'autant plus que W est petit ;
- P3 (seuils de verdict, posés avec le prior « faible à moyen » du briefing) :
  la greffe dans l'encodeur reste VIVANTE si precision@40 cohérent >= 0.95 pour un
  W <= 9 (localité utile au flou sans perte de capacité) ; elle est MORTE pour le
  canal signatures si precision@20 cohérent <= 0.80 pour tout W teste — l'atlas
  resterait alors un canal séparé (cartes de chaleur, profil rappel de l'étape 1).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os

os.environ.setdefault(
    "MOSAIC_ATLAS", "64"
)  # atlas = grille du build (cellule = 3 dims)
# corpus : 1er argument (capturé AVANT la troncature) — défaut bench/corpus (recettes)
_CORPUS_ARG = Path(sys.argv[1]) if len(sys.argv) > 1 else None
sys.argv = sys.argv[:1]  # atlas_som lit argv à l'import

import hashlib  # noqa: E402
import tempfile  # noqa: E402

# atlas_som lit MOSAIC_ATLAS et argv à l'import : le réglage ci-dessus doit précéder
from atlas_som import (  # noqa: E402
    ATLAS,
    DIM_SOM,
    ECHANTILLON_DOCS,
    GRID_BUILD,
    RACINE,
    _projeter,
    _som,
)
from mosaic.calibration import _preparer  # noqa: E402

K_ACTIVE = 20  # même densité que mosaic.signatures (20 +1, 20 −1)
KS = [5, 10, 20, 40, 60, 100, 150]
FENETRES = [5, 9, 15]  # côté de la fenêtre de voisinage (cellules)
N_JEUX_ALEATOIRES = 40
GRAINE_JEUX = 0x20260812


def _graine_token(token: str) -> int:
    """La même graine SHA que mosaic.signatures.signature — seule la localité varie."""
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")


def _sig_depuis_candidats(token: str, candidats: np.ndarray) -> np.ndarray:
    """Positions de la signature en CREUX : 40 indices (20 premiers = +1, 20 derniers
    = −1). Toutes les signatures ont exactement 40 non-zéros → même norme, le cosinus
    se réduit au produit scalaire (la normalisation est une constante partagée)."""
    rng = np.random.default_rng(_graine_token(token))
    return rng.choice(candidats, size=2 * K_ACTIVE, replace=False)


def _candidats_fenetre(cellule: int, cote: int, w: int) -> np.ndarray:
    """Dims des cellules de la fenêtre w×w autour de `cellule`, en TORE : toujours w²
    cellules (une fenêtre tronquée au bord descendrait sous les 2·K_ACTIVE positions
    requises — w=5 donne 75 dims, il en faut 40)."""
    cy, cx = divmod(cellule, cote)
    demi = w // 2
    ys = [(cy - demi + o) % cote for o in range(w)]
    xs = [(cx - demi + o) % cote for o in range(w)]
    cellules = np.array([y * cote + x for y in ys for x in xs], dtype=np.int64)
    return (cellules[:, None] * 3 + np.arange(3)[None, :]).ravel()


def _precision(
    positions: np.ndarray, dim: int, membres_par_jeu: list[np.ndarray], k: int
) -> float:
    """Superpose les k premiers membres de chaque jeu, retrouve les membres par produit
    scalaire contre TOUT le vocabulaire : précision@k moyenne (1.0 = parfait).
    `positions` : (v, 40) indices — les 20 premiers valent +1, les 20 derniers −1."""
    signes = np.concatenate(
        [np.ones(K_ACTIVE, dtype=np.float32), -np.ones(K_ACTIVE, dtype=np.float32)]
    )
    total, n = 0.0, 0
    for membres in membres_par_jeu:
        if len(membres) < k:
            continue
        pris = membres[:k]
        s = np.zeros(dim, dtype=np.float32)
        for i in pris:
            np.add.at(s, positions[i], signes)
        scores = (s[positions] * signes[None, :]).sum(axis=1)
        top = np.argpartition(-scores, k - 1)[:k]
        total += len(set(top.tolist()) & set(pris.tolist())) / k
        n += 1
    return total / n if n else float("nan")


def main() -> int:
    corpus = _CORPUS_ARG or RACINE / "bench" / "corpus"
    if not corpus.is_dir():
        raise SystemExit(
            f"corpus introuvable : {corpus} (passer le chemin en argument)"
        )
    # gros corpus (Alloprof) : échantillon des ECHANTILLON_DOCS premiers fichiers,
    # même convention déterministe que l'étape 1 (atlas_som)
    fichiers = sorted(p for p in corpus.iterdir() if p.is_file())
    dossier_echantillon = tempfile.TemporaryDirectory()
    if len(fichiers) > ECHANTILLON_DOCS:
        echantillon = Path(dossier_echantillon.name) / "corpus"
        echantillon.mkdir()
        for p in fichiers[:ECHANTILLON_DOCS]:
            (echantillon / p.name).write_bytes(p.read_bytes())
        corpus = echantillon
    docs, profiles, _colloc, _lexicon = _preparer(corpus, None, 300, GRID_BUILD)
    vocab = list(profiles.rows)
    index_de = {t: i for i, t in enumerate(vocab)}
    v = len(vocab)
    dim = GRID_BUILD[0] * GRID_BUILD[1] * GRID_BUILD[2]
    cote = ATLAS[0]
    print(
        f"corpus {corpus.name} : {len(docs)} docs, vocab {v}, "
        f"grille {GRID_BUILD}, atlas {ATLAS}"
    )

    # jeux cohérents = tokens réels par document (distincts, ordre du texte)
    coherents = []
    for _doc_id, tokens in docs:
        vus: list[int] = []
        deja = set()
        for t in tokens:
            i = index_de.get(t)
            if i is not None and i not in deja:
                deja.add(i)
                vus.append(i)
        coherents.append(np.array(vus, dtype=np.int64))
    # jeux aléatoires = même taille max, tokens uniformes (contrôle)
    rng = np.random.default_rng(GRAINE_JEUX)
    aleatoires = [
        rng.choice(v, size=min(max(KS), v), replace=False)
        for _ in range(N_JEUX_ALEATOIRES)
    ]

    # SOM réelle (même machinerie que l'étape 1) → cellule par token
    features = _projeter(profiles.acc[:v], DIM_SOM)
    bmu = _som(features)
    occupation = len(set(int(b) for b in bmu))
    print(
        f"SOM : {occupation} cellules occupées / {cote * cote} (collisions moyennes "
        f"{v / max(1, occupation):.2f} token/cellule)\n"
    )

    tout = np.arange(dim, dtype=np.int64)
    configs: list[tuple[str, int | None]] = [("sha (aléatoire, réel)", None)]
    configs += [(f"atlas W={w}", w) for w in FENETRES]

    entete = "assignation \\ K       |" + "".join(f"{k:>8}" for k in KS)
    resultats: dict[tuple[str, str, int], float] = {}
    for nom_jeu, jeux in (
        ("COHÉRENTS (docs réels)", coherents),
        ("ALÉATOIRES (contrôle)", aleatoires),
    ):
        print(f"--- jeux {nom_jeu} — précision@K ---")
        print(entete)
        for nom, w in configs:
            positions = np.zeros((v, 2 * K_ACTIVE), dtype=np.int64)
            for t, i in index_de.items():
                cand = tout if w is None else _candidats_fenetre(int(bmu[i]), cote, w)
                positions[i] = _sig_depuis_candidats(t, cand)
            ligne = f"{nom:<22}|"
            for k in KS:
                p = _precision(positions, dim, jeux, k)
                resultats[(nom_jeu, nom, k)] = p
                ligne += f"{p:>8.4f}"
            print(ligne)
        print()
    print(
        "Verdict à lire contre P3 (en-tête) : vivant si atlas W<=9 fait >=0.95 à K=40 "
        "cohérent ; mort pour le canal signatures si <=0.80 à K=20 partout."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
