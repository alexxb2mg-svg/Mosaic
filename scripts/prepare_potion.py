"""Prépare la table d'embeddings recommandée : mots français encodés par model2vec potion.

Profil recommandé Mosaic (mesuré au banc du 09/08/2026, config record 0.750) :
    mosaic build <corpus> -o <index> --embeddings potion_fr.msee --abtt 2 --weights 0.25,0.15,0.60

Usage (table de mots) :
    python scripts/prepare_potion.py --words-from <table.msee|liste.txt> -o data_externes/potion_fr.msee

--words-from accepte :
  - une table .msee existante (ex. issue de fastText) → réutilise sa liste de mots ;
  - un fichier texte, un mot par ligne.

Usage (v1.5, copie locale du modèle rerank — coupe les vérifications hub à chaque
`search --rerank`, cf. mosaic.rerank._model_source) :
    python scripts/prepare_potion.py --save-model data_externes/potion_model/

Les deux options sont indépendantes et cumulables dans le même appel. Au moins l'une des
deux doit être fournie (--words-from + --output ensemble, ou --save-model seul, ou les deux).

Dépendance : model2vec==0.8.2 (pip install "model2vec==0.8.2") — modèle MIT MinishLab,
téléchargé au premier run depuis HuggingFace (minishlab/potion-multilingual-128M).
"""

import argparse
import gzip
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosaic.embeddings import Embeddings, prepare

MODELE = "minishlab/potion-multilingual-128M"


def _charger_mots(source: Path) -> list[str]:
    if source.suffix == ".msee":
        return list(Embeddings.load(source).rows)
    return [
        ligne.strip()
        for ligne in source.read_text(encoding="utf-8").splitlines()
        if ligne.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(prog="prepare_potion")
    parser.add_argument(
        "--words-from",
        default=None,
        help="table .msee existante ou liste de mots (un par ligne)",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="chemin de la table .msee à produire"
    )
    parser.add_argument(
        "--abtt",
        type=int,
        default=0,
        help="applique all-but-the-top À LA PRÉPARATION (v1.6 §C) — table déjà nettoyée à "
        "l'écriture, plus de coût PCA au premier Embeddings.load(abtt=N) si N correspond ; "
        "entier dans [0, 255], défaut 0",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help="sauvegarde le modèle potion localement dans ce répertoire (v1.5) — "
        "`mosaic search --rerank` le charge ensuite SANS vérification hub, "
        "cf. mosaic.rerank._model_source",
    )
    args = parser.parse_args()

    if bool(args.words_from) != bool(args.output):
        parser.error("--words-from et --output vont toujours ensemble")
    if not args.words_from and not args.save_model:
        parser.error("rien à faire : fournir --words-from/--output et/ou --save-model")

    try:
        from model2vec import StaticModel
    except ImportError:
        print(
            'model2vec absent — installer : pip install "model2vec==0.8.2"',
            file=sys.stderr,
        )
        raise SystemExit(1)

    model = None
    if args.words_from:
        mots = _charger_mots(Path(args.words_from))
        print(f"{len(mots)} mots à encoder via {MODELE}")
        model = StaticModel.from_pretrained(MODELE)
        t0 = time.perf_counter()
        vecs = model.encode(mots)
        print(f"encodés en {time.perf_counter() - t0:.0f}s (dim {vecs.shape[1]})")

        with tempfile.TemporaryDirectory() as tmp:
            vec_gz = Path(tmp) / "potion.vec.gz"
            with gzip.open(vec_gz, "wt", encoding="utf-8") as f:
                f.write(f"{len(mots)} {vecs.shape[1]}\n")
                for w, v in zip(mots, vecs, strict=True):
                    f.write(w + " " + " ".join(f"{x:.4f}" for x in v) + "\n")
            stats = prepare(
                vec_gz, Path(args.output), keep=len(mots) + 1, abtt=args.abtt
            )
        print(
            f"table écrite : {args.output} ({stats['kept']} mots, dim {stats['dim']})"
        )

    if args.save_model:
        if model is None:
            model = StaticModel.from_pretrained(MODELE)
        out_dir = Path(args.save_model)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        print(
            f"modèle sauvegardé localement : {out_dir} (chargement sans vérification hub)"
        )


if __name__ == "__main__":
    main()
