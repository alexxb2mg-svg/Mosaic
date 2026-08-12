# Architecture — the mental map

This is the document that lets a newcomer enter the codebase without the author in
the room. It answers three questions: how a document becomes searchable, why each
retrieval channel exists, and which rules are never negotiated. Design *decisions*
and their evidence live in [MEASURES.md](MEASURES.md); this file is the map, that
one is the log.

## The pipeline, end to end

```
folder of documents
  │  ingest.py        conversion is a PRISM: .pdf/.docx/.xlsx/... are converted
  │                   in memory (markitdown), OCR is an opt-in hook (rapidocr);
  │                   nothing transcribed is ever persisted — a document that
  │                   cannot be read is skipped AND counted, never silently lost
  ▼
tokens
  │  tokenize.py      lowercase word stream, French-aware, stopwords kept out of
  │                   co-occurrence learning
  │  lexicon.py       canonicalization EN→FR through a pivot lexicon (multi-word
  │                   keys resolved before collocations)
  │  collocations.py  frequent pairs fused into single tokens ("pompe_chaleur")
  ▼
per-token vectors                      src/mosaic/signatures.py + profiles.py
  │  signature(token) : SHA-seeded ternary vector (20 ones, 20 minus-ones) —
  │                     deterministic, corpus-independent, the token's identity
  │  profile(token)   : co-occurrence accumulation (window 5, sorted pair arrays)
  │                     → PPMI weighting → randomized-SVD smoothing (rank 300)
  │                     — the token's learned MEANING in this corpus
  │  embeddings.py    : optional third component (γ) from a static table (.msee,
  │                     model2vec/fastText format) — the only pretrained part
  │  encoder.py       : word_vector = α·signature + β·profile + γ·embedding,
  │                     normalized; weights are calibrated per corpus
  ▼
per-document vector                    encoder.py + index.py
  │  TF-IDF-weighted superposition of its tokens' vectors, quantized to int8
  │  (the "grid": dims = w×h×3, default 64×64×3 = 12,288 — renderable as a
  │  PNG but the LAYOUT is inert; proven by permutation, see MEASURES)
  ▼
index on disk                          store.py (.msei/.msev/.msee/.msbm/.msat)
  │  atomic writes (os.replace), magic + version headers, loud refusal on
  │  truncation; search opens lazily (memmap) — a query touches only the rows
  │  it needs
  ▼
search                                 queries.py (+ index.py Index.search)
     cosine over int8 matrices, then the opt-in channels below
```

`Index.build` orchestrates the whole left column; `Index.search` routes among the
channels. `cli.py` is a thin dispatch table over all of it (one `_cmd_*` handler
per subcommand); `infra_mcp/mosaic_mcp.py` exposes the same engine as an MCP
server with indexes cached warm.

## The channels, and why each exists

Every channel answers one measured failure of the bare grid. Each is **opt-in and
separate**: off means bit-for-bit absent. The dilution lesson (MEASURES: typed
grids, fusion-loses-on-paraphrase) forbids silently mixing a new signal into the
main vector.

| Channel | Module | The failure it answers |
|---|---|---|
| BM25 postings (`--hybride`) | `bm25.py` | lexical terrain: queries that reuse the documents' exact words beat any semantic blur |
| Rank fusion (`--fusion`, RRF K=60) | `queries.py` | no single ranker wins everywhere; fusion refuses to run degraded (all channels or none) |
| Semantic atlas (`--atlas`) | `atlas.py` | a recall channel: SOM over token profiles, heat-maps per document, errors decorrelated from the grid's |
| Typed grids (`--grilles-typees`) | `typage.py` | identifiers drown in prose; routing meaning/refs/paths to separate grids cures reference lookup |
| Relations (`--relations`) | `relations.py` | free structure in the corpus tree (folder/year/month), bound to signatures by circular shift (`np.roll`) — multi-hop traversal (`chemin`) |
| Grammatical roles (`--grammatical`) | `grammaire.py` | bag-of-words cannot tell "A upstream of B" from "B upstream of A"; closed-class rule analyzer, **abstention over guessing** |
| Rerank (`--rerank`) | `rerank.py` | semantic traps sit in the top-50 but not top-10; model2vec re-scores a candidate window (λ=0.70, depth 50) |
| Connectors (`--connecteurs`) | `connecteurs.py` | query algebra: "A sans B" must push B *down*, not ignore it |
| Facets | `facettes.py` | document-type and recency as filters/boosts, not tokens |
| Temporal truth | `temporel.py` | newest version canonical, older flagged stale (`actuel`) |
| Belief memory | `croyance.py` | agents assert facts; conformal-calibrated confidence, full history kept |
| Semantic diff | `diff.py` | two deterministic builds are bit-comparable; their difference is a semantic object (vocabulary drift / usage drift / changed docs) |
| Meta-search | `meta.py` | several indexes queried at once, rank-fused with provenance |

Supporting cast: `smoothing.py` (randomized SVD, buffer-disciplined),
`calibration.py` (`mosaic calibrer` — picks weights on *your* ground truth),
`profil.py` (declarative per-corpus profile: roles, types, refs criteria,
grammatical verb lists), `resolution.py`/`docio.py`/`render.py`/`carte.py`
(id resolution, document IO, PNG rendering, folder identity cards).

## Invariants — never negotiated

1. **Per-machine bit determinism.** Same corpus + same config + same machine →
   byte-identical index. int8 quantization absorbs BLAS variance; float artifacts
   may differ *across* machines, rankings do not. Any change to tokenization or
   encoding semantically invalidates existing indexes — rebuild, and say so.
2. **Abstention over guessing.** A channel that is unsure emits *nothing* (null
   vector, skipped doc — always counted, never silent). A wrong signal lies to
   the agent downstream; silence is cheaper than noise. (Measured: the grammatical
   analyzer's 2.1 % error rate rests on closed classes + abstention.)
3. **Separate channels, explicit fusion.** New signals never leak into the main
   vector. Fusion is an opt-in read-time act, and it refuses degraded subsets.
4. **Loud refusal over silent degradation.** Empty corpus → build refuses and
   removes the artifact. Missing channel → search refuses the flag. Truncated
   file → ValueError, never a partial result.
5. **Every published number is replayable.** Each figure in the README or
   MEASURES traces to a committed script (`bench/`, `research/`) in the same
   repo. Defeats are published with autopsies, next to the wins.
6. **Declared predictions.** Research scripts state their expected outcomes in
   the docstring *before* the run; the verdicts are appended after, including
   the falsified ones.

## Buried decisions (do not resurrect without new evidence)

Full autopsies in [MEASURES.md](MEASURES.md): pyramid prefilter (pooling, not
locality), deterministic postings expansion (+0.83 < declared +2 threshold),
binary rerank quantization (−3.13 pts at scale), English 32M model (−2.93 pts on
French at scale), raw-score and size-weighted inter-shard fusions (both lose;
per-shard z-norm wins).

## Reading order for a newcomer

1. `signatures.py` (30 lines — the identity trick), then `profiles.py`
   (learning), then `encoder.py` (the blend). That is the heart.
2. `index.py::Index.build` with this map next to it, phase by phase.
3. `queries.py` for read-time; one channel module of your choice with its tests.
4. `tests/` — the properties tested (determinism, abstention, add-equals-rebuild)
   are the real spec.

## Verifying a change

```bash
pytest -q                    # ~590 tests; RuntimeWarnings are errors
ruff format --check . && ruff check .
```

For anything touching build/encoding: build a corpus twice (before/after your
change) and compare `sha256sum` of `vocab.msev`/`docs.msei`. Bit-identity is the
contract; a changed byte is either a bug or a declared version change requiring
rebuild. `research/ram_build.py` doubles as the A/B harness.
