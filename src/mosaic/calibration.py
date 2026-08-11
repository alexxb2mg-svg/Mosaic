"""Calibration niveau 3 — choisir les poids d'encodage par la MESURE, jamais au curseur.

La revue de paramétrage (11/08) a posé la règle : la géométrie de l'espace (le mélange
signature/cooccurrence/embedding α,β,γ) n'est pas intuitive — la régler à la main est une
machine à régression invisible. La seule interface honnête est la calibration : l'utilisateur
fournit des REQUÊTES-VÉRITÉ (« cette question doit trouver ces documents »), le banc balaye
une grille de poids, mesure (MRR + Recall@k), et ne recommande un changement QUE si le gain
sur le défaut est net. Sinon : « gardez le défaut » — c'est un résultat, pas un échec.

Efficacité : les profils de cooccurrence (+ SVD) ne dépendent PAS des poids — construits UNE
fois, chaque configuration ne coûte que le ré-encodage des documents (index recomposé en
mémoire, jamais persisté). Le corpus est préparé comme au build (canonicalize + collocations
+ profils + lissage — garder cette séquence EN PHASE avec Index.build ; hors périmètre de
calibration : OCR, type_doc, relations — sans effet sur le classement sémantique mesuré).

Garde-fous : moins de 10 requêtes-vérité -> avertissement de fiabilité ; le défaut est
toujours dans la grille (jamais de recommandation sans comparaison) ; le gain minimal pour
recommander est explicite dans le rapport.
"""

from pathlib import Path

from mosaic import GRID_DEFAULT
from mosaic.collocations import detect, merge
from mosaic.docio import _EXTS, _path_tokens, _read_text, _read_text_convertible
from mosaic.embeddings import Embeddings
from mosaic.encoder import WEIGHTS_DEFAULT
from mosaic.index import (
    EXCLUDED_DIRS,
    PROFILE_WEIGHTING_DEFAULT,
    SMOOTHING_RANK_DEFAULT,
    Index,
    _apply_smoothing,
)
from mosaic.lexicon import canonicalize, compile_lexicon, load_lexicon
from mosaic.profiles import Profiles
from mosaic.tokenize import tokenize

import numpy as np

from mosaic.encoder import encode

# Grille standard : le DÉFAUT d'abord (la référence), puis des déplacements raisonnés sur
# chaque axe (plus de signature / de cooccurrence / d'embedding, et sans embedding du tout).
# Les poids sont relatifs (word_vector normalise) : seuls les ratios comptent.
GRILLE_STANDARD: list[tuple[float, float, float]] = [
    WEIGHTS_DEFAULT,  # (0.25, 0.15, 0.60) — le défaut mesuré au banc v1.4
    (0.40, 0.15, 0.45),
    (0.15, 0.30, 0.55),
    (0.20, 0.20, 0.60),
    (0.15, 0.10, 0.75),
    (0.33, 0.33, 0.34),
    (0.50, 0.30, 0.20),
    (0.45, 0.45, 0.10),
]
SEUIL_REQUETES_FIABLES = 10
GAIN_MINIMAL_MRR = 0.02  # en-deçà, le rapport dit « gardez le défaut »


def _preparer(
    corpus_dir: Path,
    lexicon: dict | None,
    smoothing_rank: int,
    grid,
    index_paths: bool = True,
):
    """Lecture + tokenisation + collocations + profils, UNE fois (indépendant des poids).
    Séquence identique à Index.build (à garder en phase) — y compris `index_paths` :
    calibrer avec les tokens de chemin un corpus qui sera construit sans (noms de
    fichiers opaques, ex. UUID) choisirait les poids sur un espace qui n'existera pas."""
    dim = grid[0] * grid[1] * grid[2]
    files = sorted(
        p
        for p in corpus_dir.rglob("*")
        if p.suffix.lower() in _EXTS
        or p.suffix.lower() in {".pdf", ".docx", ".xlsx", ".html", ".pptx"}
        if not (EXCLUDED_DIRS & set(p.relative_to(corpus_dir).parts))
    )
    raw: list[tuple[str, list[str]]] = []
    for p in files:
        doc_id = p.relative_to(corpus_dir).as_posix()
        if p.suffix.lower() in _EXTS:
            text = _read_text(p)
        else:
            text = _read_text_convertible(p, None, ocr=False)
            if text is None:
                continue
        content_tokens = tokenize(text)
        raw.append(
            (
                doc_id,
                _path_tokens(doc_id) + content_tokens
                if index_paths
                else content_tokens,
            )
        )
    if lexicon is None:
        lexicon = load_lexicon()
    compiled = compile_lexicon(lexicon)
    canon = [(doc_id, canonicalize(tokens, compiled)) for doc_id, tokens in raw]
    colloc = detect([t for _, t in canon])
    docs = [(doc_id, merge(merge(tokens, colloc), colloc)) for doc_id, tokens in canon]
    profiles = Profiles(dim)
    for _, tokens in docs:
        profiles.learn(tokens)
    profiles.finalize(PROFILE_WEIGHTING_DEFAULT)
    _apply_smoothing(profiles, smoothing_rank)
    return docs, profiles, colloc, lexicon


MIN_TOKENS_HELD_OUT = 40  # un doc plus court ne donne pas deux moitiés exploitables
TERMES_PAR_REQUETE = 6


def generer_verite_held_out(
    docs: list[tuple[str, list[str]]],
    profiles: Profiles,
    max_requetes: int = 40,
) -> tuple[list[tuple[str, list[str]]], list[dict]]:
    """VÉRITÉ DÉTERMINISTE, sans LLM ni humain (held-out) : chaque document assez long est
    COUPÉ en deux moitiés — la moitié A remplace le document dans l'index de calibration, la
    requête = les termes les plus distinctifs (tf×idf) de la moitié B, vérité = le document.
    Les moitiés partagent le SUJET mais pas les phrases : pseudo-paraphrase naturelle.

    Limite assumée (dans le rapport) : mesure la robustesse au décalage de vocabulaire
    INTRA-corpus, pas l'intention humaine — un plancher déterministe, les requêtes rédigées
    restent l'étalon-or. Rend (docs_indexables_moitié_A, requêtes)."""
    docs_a: list[tuple[str, list[str]]] = []
    requetes: list[dict] = []
    for doc_id, tokens in docs:
        if len(tokens) < MIN_TOKENS_HELD_OUT or len(requetes) >= max_requetes:
            docs_a.append(
                (doc_id, tokens)
            )  # trop court (ou quota atteint) : doc entier
            continue
        milieu = len(tokens) // 2
        moitie_a, moitie_b = tokens[:milieu], tokens[milieu:]
        docs_a.append((doc_id, moitie_a))
        # termes distinctifs de B : tf(B) × idf(corpus), déterministe (tri par score puis mot)
        tf: dict[str, int] = {}
        for t in moitie_b:
            tf[t] = tf.get(t, 0) + 1
        scores = sorted(
            ((cnt * profiles.idf(t), t) for t, cnt in tf.items()),
            key=lambda x: (-x[0], x[1]),
        )
        termes = [t for _s, t in scores[:TERMES_PAR_REQUETE]]
        if termes:
            requetes.append({"query": " ".join(termes), "relevant": [doc_id]})
    return docs_a, requetes


def _evaluer(idx: Index, requetes: list[dict], k: int = 10) -> dict:
    """MRR + Recall@k sur les requêtes-vérité (mêmes métriques que le banc historique)."""
    rr, hits = [], 0
    for q in requetes:
        res = idx.search(q["query"], k=k)
        ids = [r["id"] for r in res]
        rangs = [ids.index(d) + 1 for d in q["relevant"] if d in ids]
        rr.append(1.0 / min(rangs) if rangs else 0.0)
        hits += 1 if rangs else 0
    n = max(1, len(requetes))
    return {"mrr": round(sum(rr) / n, 4), f"recall@{k}": round(hits / n, 4)}


def calibrer(
    corpus_dir: Path,
    requetes: list[dict] | None = None,
    grille: list[tuple[float, float, float]] | None = None,
    embeddings_path: Path | None = None,
    abtt: int = 0,
    smoothing_rank: int = SMOOTHING_RANK_DEFAULT,
    grid=GRID_DEFAULT,
    k: int = 10,
    verite_auto: bool = False,
    index_paths: bool = True,
) -> dict:
    """Balaye la grille de poids sur le corpus, évalue chaque configuration contre les
    requêtes-vérité, et rend un rapport : classement, gagnante, gain vs défaut, et la
    recommandation honnête (changer seulement si le gain est net).

    `verite_auto=True` : AUCUNE requête à fournir — la vérité est GÉNÉRÉE déterministiquement
    (held-out : moitié A indexée, requête = termes distinctifs de la moitié B). Sans LLM ni
    humain. Limite portée dans le rapport : mesure la robustesse au vocabulaire décalé, pas
    l'intention humaine — plancher déterministe, les requêtes rédigées restent l'étalon-or."""
    if verite_auto and requetes:
        raise ValueError("verite_auto=True : ne pas fournir de requêtes en même temps")
    if not verite_auto:
        if not requetes:
            raise ValueError("fournir des requêtes-vérité, ou passer verite_auto=True")
        for i, q in enumerate(requetes):
            if "query" not in q or "relevant" not in q or not q["relevant"]:
                raise ValueError(
                    f"requêtes-vérité : l'entrée {i} doit avoir 'query' et 'relevant' (non vide)"
                )
    grille = list(grille) if grille else list(GRILLE_STANDARD)
    if WEIGHTS_DEFAULT not in grille:
        grille.insert(0, WEIGHTS_DEFAULT)  # le défaut est TOUJOURS la référence

    embeddings = (
        Embeddings.load(Path(embeddings_path), abtt=abtt)
        if embeddings_path is not None
        else None
    )
    dim_grid = grid[0] * grid[1] * grid[2]
    docs, profiles, colloc, lexicon = _preparer(
        Path(corpus_dir), None, smoothing_rank, grid, index_paths=index_paths
    )
    if not docs:
        raise ValueError(f"corpus vide (aucun document lisible) : {corpus_dir}")
    if verite_auto:
        docs, requetes = generer_verite_held_out(docs, profiles)
        if not requetes:
            raise ValueError(
                "verite_auto : aucun document assez long pour le held-out "
                f"(minimum {MIN_TOKENS_HELD_OUT} tokens)"
            )
        # Held-out PROPRE : les profils de l'index de calibration sont ré-appris sur les
        # moitiés A seulement — la moitié B (source des requêtes) ne fuit pas dans l'index.
        # (Les collocations du corpus complet sont conservées : signal de paires fréquentes,
        # non doc-spécifique — fuite négligeable, documentée.)
        profiles = Profiles(dim_grid)
        for _, tokens in docs:
            profiles.learn(tokens)
        profiles.finalize(PROFILE_WEIGHTING_DEFAULT)
        _apply_smoothing(profiles, smoothing_rank)
    assert requetes is not None

    dim = grid[0] * grid[1] * grid[2]
    mesures: list[tuple[tuple[float, float, float], float, float]] = []
    for weights in grille:
        mat = np.zeros((len(docs), dim), dtype=np.int8)
        norms = np.zeros(len(docs), dtype=np.float32)
        ids: list[str] = []
        for row, (doc_id, tokens) in enumerate(docs):
            q, n = encode(tokens, profiles, embeddings=embeddings, weights=weights)
            mat[row], norms[row] = q, n
            ids.append(doc_id)
        idx = Index(  # en MÉMOIRE seulement — jamais _save()
            Path("."),
            profiles,
            colloc,
            mat,
            norms,
            ids,
            grid,
            lexicon,
            embeddings=embeddings,
            embed_path=Path(embeddings_path) if embeddings_path else None,
            weights=weights,
            smoothing_rank=smoothing_rank,
        )
        scores = _evaluer(idx, requetes, k=k)
        mesures.append((weights, float(scores["mrr"]), float(scores[f"recall@{k}"])))

    mesures.sort(key=lambda m: (-m[1], -m[2]))
    mrr_defaut = next(m[1] for m in mesures if m[0] == WEIGHTS_DEFAULT)
    w_gagnante, mrr_gagnante, _recall_gagnante = mesures[0]
    gain = round(mrr_gagnante - mrr_defaut, 4)
    recommandation = (
        "changer"
        if gain >= GAIN_MINIMAL_MRR and w_gagnante != WEIGHTS_DEFAULT
        else "garder_defaut"
    )

    def _ligne(m):
        return {"weights": list(m[0]), "mrr": m[1], f"recall@{k}": m[2]}

    return {
        "n_documents": len(docs),
        "n_requetes": len(requetes),
        "fiable": len(requetes) >= SEUIL_REQUETES_FIABLES,
        "verite": "auto (held-out déterministe)" if verite_auto else "fournie",
        "defaut": {"weights": list(WEIGHTS_DEFAULT), "mrr": mrr_defaut},
        "gagnante": _ligne(mesures[0]),
        "gain_mrr_vs_defaut": gain,
        "recommandation": recommandation,
        "classement": [_ligne(m) for m in mesures],
    }


def expliquer_calibration(rapport: dict, langue: str = "fr") -> str:
    """MODE HUMAIN : le verdict de la calibration en clair (fr/en) — jamais un tableau brut."""
    if langue not in ("fr", "en"):
        raise ValueError(f"langue : 'fr' ou 'en' attendu, reçu {langue!r}")
    en = langue == "en"
    g = rapport["gagnante"]
    d = rapport["defaut"]
    w = ",".join(str(x) for x in g["weights"])
    lignes = []
    auto = "auto" in str(rapport.get("verite", ""))
    if en:
        lignes.append(
            f"Calibration on {rapport['n_documents']} documents, "
            f"{rapport['n_requetes']} ground-truth queries."
        )
        if auto:
            lignes.append(
                "Ground truth was GENERATED deterministically (held-out halves, no LLM). "
                "It measures robustness to vocabulary shift, not human intent — "
                "hand-written queries remain the gold standard."
            )
        if not rapport["fiable"]:
            lignes.append(
                f"WARNING: fewer than {SEUIL_REQUETES_FIABLES} queries — the verdict is "
                "not statistically reliable. Add more before trusting it."
            )
        lignes.append(f"Default weights: MRR {d['mrr']} — best found: MRR {g['mrr']}.")
        if rapport["recommandation"] == "changer":
            lignes.append(
                f"Recommendation: REBUILD with --weights {w} "
                f"(clear gain: +{rapport['gain_mrr_vs_defaut']} MRR)."
            )
        else:
            lignes.append(
                "Recommendation: KEEP THE DEFAULT — no configuration beats it clearly. "
                "That is a result, not a failure."
            )
    else:
        lignes.append(
            f"Calibration sur {rapport['n_documents']} documents, "
            f"{rapport['n_requetes']} requêtes-vérité."
        )
        if auto:
            lignes.append(
                "La vérité a été GÉNÉRÉE déterministiquement (moitiés held-out, sans LLM). "
                "Elle mesure la robustesse au vocabulaire décalé, pas l'intention humaine — "
                "les requêtes rédigées restent l'étalon-or."
            )
        if not rapport["fiable"]:
            lignes.append(
                f"ATTENTION : moins de {SEUIL_REQUETES_FIABLES} requêtes — le verdict n'est "
                "pas statistiquement fiable. En ajouter avant de s'y fier."
            )
        lignes.append(
            f"Poids par défaut : MRR {d['mrr']} — meilleure trouvée : MRR {g['mrr']}."
        )
        if rapport["recommandation"] == "changer":
            lignes.append(
                f"Recommandation : RECONSTRUIRE avec --weights {w} "
                f"(gain net : +{rapport['gain_mrr_vs_defaut']} MRR)."
            )
        else:
            lignes.append(
                "Recommandation : GARDER LE DÉFAUT — aucune configuration ne le bat "
                "nettement. C'est un résultat, pas un échec."
            )
    return "\n".join(lignes)
