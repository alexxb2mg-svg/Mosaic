# The measures behind Mosaic

Every design decision in this engine was settled by measurement — and the defeats are
kept on the record next to the wins, because a benchmark you only quote when it
flatters you is advertising, not evidence. This document is the campaign log; the
README keeps only the headline numbers. Every figure here is replayable from a script
in `bench/` or `research/`.

## The bundled bench (40 documents, 12 paraphrase traps)

`bench/corpus` + `bench/verite.jsonl` — French cooking articles with ground-truth
queries written to share **no keywords** with their target documents. Small by design:
it fits in a test run and exposes paraphrase behavior.

- Default engine: **11/12 top-1, 12/12 top-3**.
- With `--grilles-typees`: **12/12 top-1** — the last trap falls once path noise is
  quarantined in its own grid.
- Replay: `python bench/run_bench.py bench/corpus bench/verite.jsonl [--grilles-typees]`

## The Alloprof campaign (2,556 docs, 2,316 real queries)

The first large public bench ([Alloprof](https://huggingface.co/datasets/lyon-nlp/alloprof),
the corpus MTEB uses for French retrieval). What it taught us, kept in full because it
is the honest story:

- **This is BM25's home turf.** Student questions reuse the documents' exact
  vocabulary, and the queries are long and noisy. Mosaic's stock defaults — calibrated
  on a different corpus — scored 0.307 Recall@10.
- **The shipped levers close most of the gap.** `--no-path-tokens` (the corpus files
  are named by UUID; indexing path tokens injects hex noise into every grid) is worth
  +1.5 pts. Running `mosaic calibrer` on 386 sample queries picked markedly more
  lexical weights (0.50/0.30/0.20) for +6.3 more pts. Calibrated, Mosaic (0.385) edges
  out model2vec (0.379) — an engine with no pretraining, tuned by measurement, passing
  a trained model on terrain that favors neither.
- **The three-channel fusion wins outright on this terrain.** RRF over
  (Mosaic + BM25 + model2vec) reached 0.517 vs 0.498 for the standard BM25+model2vec
  hybrid: Mosaic's errors are decorrelated from both. The same fusion with
  *uncalibrated* Mosaic scored 0.475 — below BM25 alone (0.482). A fusion is only as
  good as its worst-configured channel.
- **Determinism holds at scale.** Two independent builds returned identical metrics to
  the fourth decimal (0.3848 / 0.2593).
- Replay: `python bench/alloprof.py` then the commands in `bench/README.md`.

## The fusion lesson — the same architecture, two verdicts

On a paraphrase-heavy private corpus (366 real business documents, 40 ground-truth
queries), the exact fusion that wins Alloprof **loses**: +2.5 pts of recall, −4 pts of
MRR, and the semantic-trap MRR divided by three (0.153 solo vs 0.051 fused). A rank
fusion averages its channels' opinions; on lexical terrain that averages toward the
truth, on semantic terrain it dilutes exactly the wins the grid exists for. This pair
of verdicts is why the README routes by terrain instead of recommending one setup —
and why `--fusion` will never be a silent default.

## The typed-grids campaign (v4)

Idea: route each kind of data (meaning words / identifiers / path tokens) to its own
grid and synthesize at read time. Measured on 500 real product records against the
standard engine *with* its facet-based reference boost:

- a reference drowned in noise words: **0.90 vs 0.825** (the production failure mode,
  cured by routing + idf-weighted synthesis — a rank-based synthesis fails here, 0.2:
  informativeness weighting is required);
- bare reference 0.9833 vs 0.975; cross-vendor join 1.0 both sides;
- plain designations 0.9333 vs **0.9667** — the typed engine pays a small toll on
  prose, which is why it is opt-in per corpus, never a default;
- identifier-reading precedence must be **gated on rarity** (df ≤ 2, threshold
  calibrated): a shared technical descriptor looks like a reference but identifies
  nothing — ungated precedence costs designations 0.75 → 0.9333 gated;
- reranking composes: bare ref 0.9917 with `--rerank`, precedence kept as the primary
  sort key (an embedding cosine cannot dethrone the exact holder of an identifier);
- grid dimensions are sized to each grid's vocabulary — a pure-signature grid has
  per-document capacity, not per-vocabulary (measured: ref grid 12,288 → 768 dims,
  identical scores, 16× less RAM). Growth is automatic when vocabulary demands it, so
  the memory saving is corpus-dependent.
- On pure prose with nothing to sort (Alloprof) it is neutral-to-slightly-negative:
  −2.7 R@1. Opt-in per corpus.

## The atlas saga (from metaphor to mechanism)

The grid renders as an image, but a permutation test proved the layout was inert:
shuffle every cell and the scores do not move a bit (`research/permutation_grille.py`).
The atlas track asked whether a **learned** assignment (SOM over co-occurrence
profiles: neighboring cells hold related tokens) could turn the picture into a
mechanism. Five measured steps:

1. **Prototype** (`research/atlas_som.py`): heatmaps on a learned 64×64 atlas. On
   recipes: at the bench ceiling. On Alloprof 500: better *recall* than the flat grid
   (R@10 0.6233 vs 0.5567) but behind on precision — a recall channel profile, not a
   replacement. Gaussian blur **never** helps on hostile terrain (σ=0 wins).
2. **Superposition capacity** (`research/atlas_capacite.py`): the feared cost of
   locality (Kanerva chose random supports for a reason) does **not** materialize at
   this engine's scales — membership precision ≥ 0.9916 at K=150 with a 9×9 window,
   losses only on semantically coherent sets, no cliff.
3. **Fusion channel** (`research/atlas_fusion.py`): added as a 4th RRF channel on
   Alloprof — sample +2.2 pts R@10, **full corpus +2.84 pts / +3.65 MRR** over the
   trio. The map's errors decorrelate from the flat grid's.
4. **Pyramid prefilter: dead.** (`research/atlas_pyramide.py`) The pre-declared
   criterion (≥5× compute saved at <1 pt recall lost) failed — 6.2× savings cost 9.16
   pts. The control is the finding: a *permuted* atlas tracks the organized curve
   almost exactly, so what little the coarse map does comes from pooling-as-projection,
   not from semantic locality. Sub-linear search by map coarsening is not the atlas's
   argument.
5. **Engine acceptance** (`research/valider_atlas_moteur.py`): the graft (`--atlas`)
   rebuilt full Alloprof through the engine: quartet **0.5461 R@10 / 0.3572 MRR at
   29 ms/query** — +4.26 pts over the engine trio, and *above* the research prototype
   (0.5319) because the engine builds every channel on the same clean token stream
   (the prototype's SOM had ingested UUID path noise; the gap is explained, not
   mysterious). Nothing is lost to int8 quantization of the maps.

## The semantic diff (planted bench)

Two deterministic builds of the same corpus are bit-comparable, so their difference is
a semantic object. Validated on a planted bench (`research/diff_semantique.py`,
predictions declared before measurement):

- **P1, specificity** — identical corpora yield a *strictly* empty diff (an
  exact-equality fast path carries the contract: without it, a float cosine turns the
  guaranteed zero into 1e-13).
- **P2, sensitivity** — substituting an ingredient in 6 documents: the new word
  surfaces in the appeared vocabulary, the replaced word tops the usage declines at
  −75%, and 70% of the top-10 context drift is the substitution's co-occurrence
  neighborhood. Lesson learned on the first run: profile drift detects *context*
  change, not frequency change — the df-based usage reading is a separate signal.
- **P3, locality** — untouched documents far from the change drift 4.5× less than its
  neighbors, and the context-drift ranking is semantically coherent (caramel, ganache,
  mayonnaise: the neighbors of fat). The engine does not cry change everywhere.

Shipped as `mosaic diff <t1> <t2>` (two corpus folders — ephemeral builds — or two
index folders; mixed natures and mismatched encoding spaces are refused loudly).

## Buried tracks (ratified)

- **Pyramid prefilter** (atlas step 3): failed its pre-declared criterion, and the
  permuted control revealed the mechanism — pooling-as-projection, not semantic
  locality. `research/atlas_pyramide.py`.
- **Deterministic postings expansion** (a "SPLADE without the network",
  `research/expansion_postings.py`): real but marginal gains (+0.83 pt R@10 at best on
  Alloprof, threshold declared at +2). The lesson outvalues the feature: the token
  stream's canonicalization + collocations already close most of the vocabulary gap
  that learned sparse expansion closes elsewhere. Also caught a bench trap: with path
  tokens on, BM25 scored a perfect MRR on the paraphrase bench because file names name
  the dishes — invalid control, fixed with `--no-path-tokens`.

## Reproduce

```bash
python bench/run_bench.py bench/corpus bench/verite.jsonl        # bundled bench
python bench/alloprof.py                                          # fetch Alloprof
python bench/run_bench.py bench/alloprof/corpus bench/alloprof/verite.jsonl \
    --config alloprof --no-path-tokens --weights 0.5,0.3,0.2
python bench/fusion_bench.py bench/alloprof/corpus bench/alloprof/verite.jsonl
python research/valider_atlas_moteur.py bench/alloprof/corpus \
    bench/alloprof/verite.jsonl <table.msee>                      # engine quartet
```

Private-corpus figures (the product bench, the fusion-dilution bench) are labeled as
such wherever they appear: the corpora cannot ship, the scripts and the method do.
