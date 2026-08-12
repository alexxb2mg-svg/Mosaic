"""Grilles typées — idée du 12/08 : router chaque TYPE de donnée vers SA grille.

L'IDÉE : au lieu d'une grande grille unique où tout se superpose (et où une référence
produit se fait noyer par 50 mots courants — mesuré en prod, d'où le boost réf en
facette), l'encodeur trie à l'écriture : les mots de sens dans une grille, les
identifiants (réfs, codes) dans une autre, les tokens de chemin dans une troisième —
autant de grilles que les types le nécessitent. À la lecture, une synthèse recombine
les lectures par grille. Corollaire : la séparation permet peut-être de
RÉTRÉCIR chaque grille (3×32×32 < 64×64 unique) — à évaluer.

QUESTION FALSIFIABLE : à budget de dimensions ÉGAL (et inférieur), les grilles typées
battent-elles la grille unique mélangée ? Expérience CONTRÔLÉE : même machinerie
partout (mêmes profils PPMI+lissage, même encodeur, mêmes requêtes), le témoin
« mélangé » est construit par CE script avec le routage désactivé — jamais deux
codes différents comparés.

TROIS TERRAINS :
1. recettes (12 pièges de paraphrase + contrôle lexical) ;
2. NOYADE DE RÉFÉRENCES (synthétique, déterministe) : une réf unique injectée dans
   chaque document ; requête = la réf seule, puis la réf + 3 mots courants d'un AUTRE
   document — le scénario exact où la grille unique souffre en prod ;
3. Alloprof (échantillon, requêtes réelles), tokens de chemin UUID INCLUS : la grille
   mélangée subit le bruit hexa (mesuré −1,5 pt au banc public) ; prédiction à tester —
   le tri par type met ce bruit en quarantaine automatiquement (la requête n'a jamais
   de token de type « chemin » → grille silencieuse).

SYNTHÈSES COMPARÉES : pondérée par la masse idf de la requête par type (un token rare
pèse ce qu'il informe), et RRF K=60 (même constante que la fusion livrée). Une grille
sans signal sur la requête est écartée (même principe que search --fusion).

Statut : recherche (hypothèse ouverte) — résultats posés pour discussion, jamais un
verdict unilatéral.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mosaic.collocations import detect, merge
from mosaic.docio import _path_tokens
from mosaic.encoder import encode
from mosaic.facettes import _est_ref
from mosaic.index import _apply_smoothing
from mosaic.lexicon import canonicalize, compile_lexicon, load_lexicon
from mosaic.meta import K_RRF_DEFAULT
from mosaic.profiles import Profiles
from mosaic.tokenize import tokenize

RACINE = Path(__file__).resolve().parent.parent
CORPUS = Path(sys.argv[1]) if len(sys.argv) > 1 else RACINE / "bench" / "corpus"
VERITE = Path(sys.argv[2]) if len(sys.argv) > 2 else RACINE / "bench" / "verite.jsonl"

K_EVAL = 10
SMOOTHING = 300
SEUIL_VOCAB_LISSAGE = 350  # en-deçà, rang 300 n'a pas de sens (vocab minuscule)
DF_MAX_IDENTIFIANT = 3  # un token « ref » présent dans <= 3 docs est un identifiant
TERMES_LEXICAUX = 4
# configurations : nom -> dims par grille (le témoin « melange » = une seule grille)
CONFIGS: dict[str, dict[str, int]] = {
    "melange 12288 (temoin)": {"tout": 12288},
    "melange 9216 (temoin reduit)": {"tout": 9216},
    "3 typees x 4096 (=12288)": {"sens": 4096, "ref": 4096, "chemin": 4096},
    "3 typees x 3072 (=9216)": {"sens": 3072, "ref": 3072, "chemin": 3072},
    "3 typees x 1024 (=3072)": {"sens": 1024, "ref": 1024, "chemin": 1024},
}
# gros corpus : échantillon déterministe (mêmes règles que research/atlas_som.py)
SEUIL_ECHANTILLON = 600
ECHANTILLON_DOCS = 500
MAX_REQUETES = 300


def _router(tok: str) -> str:
    return "ref" if _est_ref(tok) else "sens"


def _metriques(classements: list[list[str]], verites: list[list[str]]) -> dict:
    rr, hits, hits1 = [], 0, 0
    for ids, rel in zip(classements, verites):
        rangs = [ids.index(d) + 1 for d in rel if d in ids]
        rr.append(1.0 / min(rangs) if rangs else 0.0)
        hits += 1 if rangs else 0
        hits1 += 1 if rangs and min(rangs) == 1 else 0
    n = max(1, len(classements))
    return {
        "r1": round(hits1 / n, 4),
        "mrr": round(sum(rr) / n, 4),
        "r10": round(hits / n, 4),
    }


def _preparer_types(
    corpus: Path,
) -> tuple[list[str], list[dict[str, list[str]]], set, dict]:
    """Tokens par TYPE et par document. Routage par provenance (chemin) puis par règle
    (réf/sens) sur le flux canonique — mêmes canonicalisation/collocations que le build."""
    lexicon = load_lexicon()
    compiled = compile_lexicon(lexicon)
    fichiers = sorted(p for p in corpus.iterdir() if p.is_file())
    ids: list[str] = []
    bruts: list[
        tuple[list[str], list[str]]
    ] = []  # (contenu canonique, chemin canonique)
    for p in fichiers:
        texte = p.read_text(encoding="utf-8", errors="replace")
        ids.append(p.name)
        bruts.append(
            (
                canonicalize(tokenize(texte), compiled),
                canonicalize(_path_tokens(p.name), compiled),
            )
        )
    colloc = detect([c for c, _ in bruts])
    par_type: list[dict[str, list[str]]] = []
    for contenu, chemin in bruts:
        fusionne = merge(merge(contenu, colloc), colloc)
        d: dict[str, list[str]] = {"sens": [], "ref": [], "chemin": list(chemin)}
        for t in fusionne:
            d[_router(t)].append(t)
        par_type.append(d)
    return ids, par_type, colloc, lexicon


class MoteurTypes:
    """N grilles (une par type), même encodeur partout. Le témoin mélangé est le cas
    particulier dims={'tout': D} : tous les flux concaténés dans une seule grille."""

    def __init__(self, dims: dict[str, int], docs_types: list[dict[str, list[str]]]):
        self.dims = dims
        self.profils: dict[str, Profiles] = {}
        self.mats: dict[str, np.ndarray] = {}
        self.norms: dict[str, np.ndarray] = {}
        flux = (
            {"tout": [sum(d.values(), []) for d in docs_types]}
            if list(dims) == ["tout"]
            else {t: [d[t] for d in docs_types] for t in dims}
        )
        self.flux = flux
        for t, dim in dims.items():
            prof = Profiles(dim)
            for tokens in flux[t]:
                prof.learn(tokens)
            if (
                prof.rows
            ):  # une grille peut être VIDE (aucune réf dans le corpus) : zéros
                prof.finalize("ppmi")
                if len(prof.rows) > SEUIL_VOCAB_LISSAGE:
                    _apply_smoothing(prof, SMOOTHING)
            mat = np.zeros((len(docs_types), dim), dtype=np.int8)
            norms = np.zeros(len(docs_types), dtype=np.float32)
            for i, tokens in enumerate(flux[t]):
                q, n = encode(tokens, prof)
                mat[i], norms[i] = q, n
            self.profils[t] = prof
            self.mats[t] = mat.astype(np.float32)  # petit corpus : chauffé d'office
            self.norms[t] = norms

    def _flux_requete(self, tokens: list[str]) -> dict[str, list[str]]:
        if list(self.dims) == ["tout"]:
            return {"tout": tokens}
        d: dict[str, list[str]] = {t: [] for t in self.dims}
        for t in tokens:
            typ = _router(t)
            if typ in d:
                d[typ].append(t)
        return d

    def classer(self, tokens: list[str], mode: str) -> list[int]:
        """Classement des documents (indices) — `mode` : 'ponderee', 'rrf' ou 'priorite'.

        `priorite` (leçon du banc produits) : quand la requête contient des tokens de
        type « ref », la lecture de la grille ref a PRÉSÉANCE lexicographique — le
        porteur de l'identifiant passe devant, la pondération ne départage que le
        reste. C'est la sémantique du boost réf des facettes, mais réalisée DANS la
        représentation : possible uniquement parce que la lecture ref est isolée —
        la grille mélangée ne peut pas le faire."""
        n_docs = next(iter(self.mats.values())).shape[0]
        canaux: list[tuple[float, np.ndarray]] = []
        cos_ref: np.ndarray | None = None
        ref_identifiant = False
        for t, qtoks in self._flux_requete(tokens).items():
            if not qtoks:
                continue
            q, qn = encode(qtoks, self.profils[t])
            if qn == 0.0:
                continue
            denom = self.norms[t] * np.float32(qn)
            denom[denom == 0] = 1.0
            cos = (self.mats[t] @ q.astype(np.float32)) / denom
            if not np.any(cos):
                continue
            masse = sum(self.profils[t].idf(x) for x in qtoks)
            canaux.append((masse, cos))
            if t == "ref":
                cos_ref = cos
                # la préséance ne vaut que pour un IDENTIFIANT (rare par définition,
                # df<=3) — un descripteur technique partagé (u1000r2v, présent dans des
                # dizaines de désignations) ressemble à une réf mais n'identifie rien :
                # lui donner la préséance vole le rang aux bons documents (mesuré).
                ref_identifiant = all(
                    self.profils[t].df.get(x, 0) <= DF_MAX_IDENTIFIANT for x in qtoks
                )
        if not canaux:
            return list(range(min(K_EVAL, n_docs)))
        if mode == "priorite" and cos_ref is not None and ref_identifiant:
            total = sum(m for m, _ in canaux)
            reste = np.zeros(n_docs, dtype=np.float64)
            for masse, cos in canaux:
                reste += (masse / total) * cos.astype(np.float64)
            # lexicographique VRAI (np.lexsort, dernière clé = primaire) : rang par
            # cos_ref d'abord (quantifié pour absorber le bruit numérique), le score
            # pondéré ne départage qu'à cos_ref égal
            ordre = np.lexsort((-reste, -np.round(cos_ref.astype(np.float64), 4)))
            return list(ordre[:K_EVAL])
        if mode in ("ponderee", "priorite"):  # priorite sans réf en requête -> ponderee
            total = sum(m for m, _ in canaux)
            score = np.zeros(n_docs, dtype=np.float64)
            for masse, cos in canaux:
                score += (masse / total) * cos.astype(np.float64)
        else:  # rrf
            score = np.zeros(n_docs, dtype=np.float64)
            contrib = 1.0 / (K_RRF_DEFAULT + np.arange(1, n_docs + 1, dtype=np.float64))
            for _masse, cos in canaux:
                ordre = np.argsort(-cos, kind="stable")
                score[ordre] += contrib
        return list(np.argsort(-score, kind="stable")[:K_EVAL])


def _evaluer(
    moteur: MoteurTypes,
    ids: list[str],
    colloc: set,
    lexicon: dict,
    jeu: list[tuple[str, list[str]]],
    mode: str,
) -> dict:
    compiled = compile_lexicon(lexicon)
    cls = []
    for q, _rel in jeu:
        toks = merge(merge(canonicalize(tokenize(q), compiled), colloc), colloc)
        cls.append([ids[i] for i in moteur.classer(toks, mode)])
    return _metriques(cls, [rel for _q, rel in jeu])


def _jeu_lexical(
    par_type, ids, profils_ref: MoteurTypes
) -> list[tuple[str, list[str]]]:
    """Contrôle lexical déterministe : top tf×idf du flux de SENS de chaque document.
    L'idf vient du témoin mélangé (même référence pour toutes les configurations)."""
    prof = next(iter(profils_ref.profils.values()))
    jeu = []
    for doc_id, d in zip(ids, par_type):
        tf: dict[str, int] = {}
        for t in d["sens"]:
            tf[t] = tf.get(t, 0) + 1
        tops = sorted(
            ((c * prof.idf(t), t) for t, c in tf.items()), key=lambda x: (-x[0], x[1])
        )
        jeu.append((" ".join(t for _s, t in tops[:TERMES_LEXICAUX]), [doc_id]))
    return jeu


def main() -> int:
    # -- corpus (échantillonné si gros), + variante réfs injectées sur les recettes -----
    dossier_tmp = tempfile.TemporaryDirectory()
    corpus = CORPUS
    fichiers = sorted(p for p in CORPUS.iterdir() if p.is_file())
    echantillonne = len(fichiers) > SEUIL_ECHANTILLON
    pieges: list[tuple[str, list[str]]] = [
        (str(o["query"]), [str(x) for x in o["relevant"]])
        for o in (
            json.loads(li)
            for li in VERITE.read_text(encoding="utf-8").splitlines()
            if li.strip()
        )
    ]
    if echantillonne:
        corpus = Path(dossier_tmp.name) / "corpus"
        corpus.mkdir()
        garde = fichiers[:ECHANTILLON_DOCS]
        for p in garde:
            (corpus / p.name).write_bytes(p.read_bytes())
        noms = {p.name for p in garde}
        pieges = [(q, rel) for q, rel in pieges if all(r in noms for r in rel)][
            :MAX_REQUETES
        ]

    # réfs synthétiques : petit corpus seulement (le scénario noyade est le but du jeu)
    jeu_ref_seule: list[tuple[str, list[str]]] = []
    jeu_ref_noyee: list[tuple[str, list[str]]] = []
    if not echantillonne:
        corpus_refs = Path(dossier_tmp.name) / "corpus_refs"
        corpus_refs.mkdir()
        srcs = sorted(p for p in CORPUS.iterdir() if p.is_file())
        for i, p in enumerate(srcs):
            ref = f"zq{i:03d}7k"  # mixte alphanum >= 5 -> type « ref » par la règle du moteur
            (corpus_refs / p.name).write_text(
                p.read_text(encoding="utf-8", errors="replace")
                + f"\n\nreference {ref}\n",
                encoding="utf-8",
            )
            jeu_ref_seule.append((ref, [p.name]))
            autre = srcs[(i + len(srcs) // 2) % len(srcs)]
            bruit = " ".join(
                tokenize(autre.read_text(encoding="utf-8", errors="replace"))[:3]
            )
            jeu_ref_noyee.append((f"{ref} {bruit}", [p.name]))
    else:
        corpus_refs = None

    print("=== Grilles typées (idée du 12/08) — expérience contrôlée ===")
    print(
        f"corpus : {corpus}",
        f"(échantillon {ECHANTILLON_DOCS})" if echantillonne else "",
    )
    print()

    for nom_corpus, corp, jeux_specifiques in (
        ("principal", corpus, None),
        *((("refs injectées", corpus_refs, True),) if corpus_refs else ()),
    ):
        ids, par_type, colloc, lexicon = _preparer_types(corp)
        stats_flux = {
            t: sum(len(d[t]) for d in par_type) for t in ("sens", "ref", "chemin")
        }
        print(
            f"--- corpus {nom_corpus} : {len(ids)} docs, tokens par type {stats_flux} ---"
        )
        temoin: MoteurTypes | None = None
        for nom_cfg, dims in CONFIGS.items():
            moteur = MoteurTypes(dims, par_type)
            if temoin is None:
                temoin = moteur
            if jeux_specifiques:
                jeux = (("ref seule", jeu_ref_seule), ("ref noyée", jeu_ref_noyee))
            else:
                jeux = (
                    ("pièges", pieges),
                    ("lexical", _jeu_lexical(par_type, ids, temoin)),
                )
            for mode in ("ponderee", "rrf"):
                ligne = f"{nom_cfg:<30} [{mode:<8}]"
                for nom_jeu, jeu in jeux:
                    m = _evaluer(moteur, ids, colloc, lexicon, jeu, mode)
                    ligne += f"  {nom_jeu} R@1 {m['r1']:<6} MRR {m['mrr']:<6}"
                print(ligne)
        print()
    print("Résultats posés pour discussion — pas de verdict unilatéral.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
