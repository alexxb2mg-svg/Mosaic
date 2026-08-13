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

## The small-bench trap — two graft decisions settled by the full-scale judge

Two candidates looked excellent on the 12-query bundled bench and went to the
2,316-query Alloprof judge before any graft. Both verdicts reversed or confirmed
what the small bench could not see:

- **Binary rerank quantization** (`research/rerank_binaire.py`): zero loss and 16×
  compression on 12 queries → at scale, **−3.13 pts R@10 / −2.53 MRR**, with 100% of
  top-10 lists modified. The compression is real (16.0× exactly, proven
  Hamming ≡ ±1 dot product), the quality is not. No graft — at best a
  memory-constrained opt-in someday. The 12-query bench was simply too small to see
  the degradation.
- **potion-retrieval-32M (English) vs potion-multilingual-128M**: the English model
  "won" the French recipe bench on a single query flip → at scale it **loses by
  2.93 pts R@10** on 2,316 French queries (BM25 witness identical in both runs; model
  identity proven in index metadata). The multilingual 128M stays the default.

The meta-lesson is the doctrine itself: never change a default on a 12-query bench —
the judge runs before the graft, always. (Free datum from the same runs: the γ
embeddings table is worth ~+4.7 pts R@10 in this configuration.)

## The build RAM ceiling (measured, then lowered — bit-for-bit)

Building full Alloprof (2,556 docs, 50k vocabulary, grid 64) peaked at **13.3 GB**
of working set (`research/ram_build.py`). The obvious suspect — the co-occurrence
pair dict, ~230 bytes of Python overhead per pair — turned out to be the *wrong*
one: converting it to sorted int64/float64 arrays (16 bytes/pair) saved almost
nothing at this scale. The measurement falsified the hypothesis; the real whale
was the **SVD smoothing**, which materialized three full vocab×dims copies at
once (the float64 input copy, the float64 reconstruction, and the float32 cast),
plus a signature cache holding ternary {−1, 0, 1} values in int32.

Three fixes, all provably bit-identical (same operations in the same order — only
buffer lifetimes and storage widths changed; verified by sha256 of the complete
index artifacts before/after, three times):

1. pair counts as sorted arrays with a bounded consolidation buffer (the dict is
   gone — and integer-valued float weights make every accumulation order exact);
2. smoothing frees each n×d intermediate the moment it is dead, and writes its
   result through an `out=` parameter instead of materializing a second cast copy;
3. the signature cache stores int8 (same float32 values after the consumers' cast).

Result: **8.6 GB peak (−36 %) and a 14 % faster build** (139 s → 120 s — fewer
giant allocations). The remaining floor is arithmetic, not waste: the float64
smoothing copy plus the float32 profile matrix ≈ 12 bytes × vocab × dims. Going
below that (float32 SVD, chunked BLAS) would change the artifacts bit-for-bit and
therefore invalidate every existing index — an explicit decision, not a cleanup.

- Replay: `python research/ram_build.py <corpus> <index_out>` (Windows peak
  working set; the same script runs on both code generations for an honest A/B).

Follow-up, same corpus (`research/profil_build.py` — wall time per component,
answering "what would a compiled language buy?"): true BLAS is only **13 %** of
the build (the SVD, 15.6 s of 119 s). The single dominant component is
`finalize`'s pure-Python pair iteration (**49.1 s, 41 %**), and the "numpy"
encode phase (40.3 s, 34 %) is bound by many-small-ops dispatch overhead, not
by arithmetic. Amdahl's ceiling for a native kernel (or a sparse restructure —
signatures carry 40 non-zeros out of 12,288, but PPMI weights are logs, so any
reordering changes bits = a declared version change): build ≈ 25–35 s, ×3.5–4.5.
The hypothesis "the co-occurrence window loop dominates" was falsified: after
the array refactor, `learn` costs 3.5 s.

## The sharding bench (can capped indexes beat one big index?)

The scale question, put to the judge (`research/shards_fusion.py`): split
Alloprof into 4 indexes (balanced ~639 docs each, and deliberately unbalanced
6 % / 19 % / 25 % / 50 %), search each shard per query, merge the local top-10s
— predictions declared before measuring, and mostly falsified:

- **Raw cosine concatenation loses ~5 pts of R@10 even with equal-size shards.**
  Each shard learns its own profiles and IDF, so score scales drift apart:
  cross-index scores are not directly comparable. (P1 falsified.)
- **Rank fusion (RRF) over balanced shards beats the single index on recall**
  (0.3809 vs 0.3216 R@10) but pays in MRR — a recall-channel profile, the same
  shape the atlas showed. And the size bias is real and huge: in the unbalanced
  split, the 6 % shard grabs **4.0×** its fair share of the fused top-10 (rank k
  of a 164-doc index is worth much less than rank k of a 1,316-doc index; RRF
  can't see that). (P2 half-falsified — the bias exists, but recall wins anyway.)
- **The winner: per-shard z-normalization of scores.** Balanced shards + z-norm
  reaches **0.3832 R@10 / 0.2229 MRR — better than the single index on both
  metrics**. Reading: an ensemble effect — each shard's learned profiles make
  its errors decorrelate from the others', and z-norm makes the scales
  comparable without sacrificing top-rank precision the way RRF does. Honest
  caveat: μ/σ estimated on only k=10 local scores; and on the unbalanced split
  the small-shard bias persists (3.81×), though it costs less (0.3488/0.2056,
  still the best fusion there).
- **The naive bias fix is a disaster.** Weighting RRF contributions by corpus
  share crushes small shards entirely (zero docs in any top-10, R@10 0.2051 —
  worse than everything). A useful correction should act on the *depth*
  requested from each shard, not on the weight of its ranks. (P5 falsified,
  kept on the record.)

Design rule this establishes: **shard at equal sizes** (cap each index's volume,
open the next when full) **and fuse by per-shard z-normalized scores**. Sharded
search costs +13 % sequentially and parallelizes per shard; the build peak is
bounded by the largest shard. Replay:
`python research/shards_fusion.py bench/alloprof/corpus bench/alloprof/verite.jsonl <workdir>`.

## The English bench, and what it corrected (SciFact / BEIR)

Every bench above is French. That made every number here unverifiable by an
English reader — the exact reproach this repository makes to others. SciFact
(BEIR: 5,183 abstracts, 300 claims, public ground truth) fixes that.
Replay: `python bench/scifact.py`, then `bench/run_bench.py`.

| On SciFact | R@10 | MRR |
|---|---:|---:|
| Mosaic, factory defaults (0.25/0.15/0.60) | 0.6574 | 0.5167 |
| Mosaic, calibrated (0.60/0.30/0.10) | 0.70 | 0.5326 |
| BM25 | **0.7645** | **0.6061** |

**Our BM25 lands where the BEIR literature puts it** — which validates the
measuring tools, not just the engine. Numbers here are comparable to anyone's.

Three findings, two of which corrected a belief we held:

1. **It was never the language — it is query NOISE.** Relative to BM25, Mosaic
   reaches **67 %** on Alloprof (students' messy questions) and **86 %** on
   SciFact (clean scientific claims). The prediction was the opposite. What
   penalises this engine is the *tidiness of the query*, invisible with a single
   bench, and it makes deterministic query pre-processing the highest-marge
   lead we have (19 points between the two regimes).
2. **The factory defaults are overfitted to a 40-document bench.** SciFact's
   three best weightings are all heavily lexical (0.60/0.30/0.10 first), and
   Alloprof already said the same (0.50/0.30/0.20). Two corpora, two languages,
   2,616 queries, one direction. The default (0.25/0.15/0.60) was chosen on the
   bundled cooking bench — forty documents. It is not changed here: the SciFact
   gain (+1.59 MRR) sits under the tool's own +2 threshold, and changing a
   default on thin evidence is the very mistake being diagnosed. It is a terrain
   line: defaults suit paraphrase, general corpora want calibration.
3. **A second bench exposes broken instruments.** `mosaic calibrer` without an
   embeddings table was calibrating *nothing* — the weights only apply when the
   gamma channel exists; without it the encoder forces 0.5/0.5/0.0 and ignores
   them. Eleven configurations, one identical score, fifteen minutes of compute,
   and a correct conclusion for an unstated reason. Four tests had been
   validating that emptiness for months. Both fixed. Lesson kept: *a perfectly
   flat result is not a verdict, it is a hint that the knob may not be connected.*

## The challenger — a dense encoder against the engine, on the engine's bench

The obvious question this repository never asked itself: how does Mosaic fare
against what someone would build today with a modern embedder and nothing else?
Same corpus (full Alloprof), same ground truth, no favours — dense only, no
reranking, no fusion.

| Alloprof, 2,316 queries | R@10 | MRR | indexing |
|---|---:|---:|---:|
| Mosaic, defaults | 0.3216 | 0.2164 | 47 ms/doc |
| Mosaic, best configuration (four-channel fusion) | 0.5461 | 0.3572 | 47 ms/doc |
| **LFM2.5-Embedding-350M, dense only** | **0.7082** | **0.5242** | **1,140 ms/doc** |

**The challenger wins by 16.2 points of recall and 16.7 of MRR** over Mosaic's
*best* configuration — and it runs locally too (219 MB, CPU, no GPU, no network),
so "but that one needs the cloud" is not an argument here.

What survives the defeat, measured rather than invoked: indexing is **24× faster**
(2 minutes against 49 on this corpus), determinism is bit-for-bit, and this bench
measures **prose and paraphrase** — not the terrains this engine was built for
(references drowned in noise, identifiers, negation scope), which remain
unmeasured against a dense baseline. That does not rescue the result; it bounds
what it says.

What it decides: the gamma channel runs on potion-128M, a **static** table (one
vector per word, no context). Replacing it with a contextual encoder is now the
engine's highest-leverage lead — measured, not assumed.

Methodological note kept on the record: a first run on 300 documents returned
0.9256 and was nearly reported as a win. Searching 300 documents is mechanically
easier than searching 2,556. Same small-bench trap as the binary-rerank campaign,
retensioned every single time.

## Deterministic query pre-processing — the best cost/benefit ratio in the project

The English bench had shown that what penalises this engine is **query noise**, not
language: relative to BM25, Mosaic reaches 67 % on Alloprof (students' messy
questions) and 86 % on SciFact (clean claims). `mosaic.requete` attacks exactly
that — it strips conversational noise before the search, using **closed classes**
of French (greetings, requests for help, admissions of confusion, meta-discourse),
by list, never by judgement. No model, no dependency, ~100 lines.

Criterion declared before measuring: **+3 points of recall on Alloprof, no
degradation on SciFact**. One index built, two passes over the queries — the only
difference is the question asked.

| | recall@10 | MRR |
|---|---:|---:|
| Alloprof, raw queries | 0.3216 | 0.2164 |
| **Alloprof, cleaned queries** | **0.3866** | **0.2646** |
| SciFact, raw | 0.6574 | 0.5135 |
| SciFact, cleaned | 0.6574 | 0.5135 |

**+6.50 points of recall and +4.82 of MRR where there is noise, and rigorously
nothing where there is none** — identical to the fourth decimal on SciFact. That
is the ideal behaviour of a pre-processor.

Put next to everything else measured in this repository:

| Lever | Gain | Cost |
|---|---:|---|
| Atlas channel (SOM) | +4.26 pts | minutes of build, GBs of RAM |
| Weight calibration | +6.3 pts | full index rebuild |
| **Query pre-processing** | **+6.50 pts** | **no rebuild at all** |

It is the only one that applies to **existing indexes without touching them** —
it transforms the question, not the index. Opt-in (`search --nettoyer-requete`)
despite the gain: a result measured on one corpus does not become a universal
default without evidence elsewhere, which is what the SciFact control provides.

Two content-destroying bugs were caught by a **diagnostic run before the
measurement**, not by the measurement itself: `hi` stripped from "Ly6C hi
monocytes" (a domain term read as a greeting), and `(AZT).` truncated to `(AZT`.
The second fix removed code rather than adding any — the engine's tokeniser
already ignores punctuation, so cleaning it changed no result and only added
risk. Keep only what moves a measurement.

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
