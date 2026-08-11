# Mosaic

**A sovereign semantic engine: search and remember by *meaning* — on a plain CPU, deterministically, with no LLM in the loop.**

No cloud, no GPU, no per-query bill, no data ever leaving your machine. The same corpus
always produces the same index, bit for bit. Every document becomes a small **64×64 color
grid** — a mosaic — and searching means comparing grids: two texts about the same thing get
two grids that are geometrically close.

## The idea

The engine does everything **mechanical**: index, retrieve, remember, and *measure its own
confidence* — deterministic, free, reproducible. An LLM (if you use one at all) only enters
at the **point of judgment**: never in the storage or search loop, only when something must
be *decided* — settling an ambiguity the engine itself has flagged. The deterministic part
carries the weight; intelligence only pays for the irreducible.

A practical consequence for agent builders: your agent sees **five lines of JSON instead of
thousands of tokens of raw files**. The engine is a token-saving machine by construction.

## What it does

- **Semantic search** (`mosaic search`) — paraphrase-friendly retrieval over any folder of
  documents (`.md`, `.txt`, `.pdf`, `.docx`, `.xlsx`, `.html`, `.pptx`, images via optional
  OCR). Query algebra built in: *"A sans B"* pushes concept B **down** the ranking.
- **Multi-channel identity card** — beyond meaning, each document carries exact **facets**:
  its *type* (spreadsheet, scanned pdf, photo…), its *date*, its *reference codes*. A code
  in your query triggers an **automatic exact-match boost**; `--recence` blends semantic
  rank with freshness; `--type` filters exactly.
- **Temporal truth** (`mosaic actuel`) — on an evolving folder, versions of the same aspect
  are grouped; the newest is **canonical**, older ones are flagged **stale**. An agent can
  no longer mistake outdated data for truth.
- **Cross-index meta-search** (`mosaic meta`) — query several indexes at once, fused by
  rank (RRF), each result keeping its provenance.
- **Graph traversal without a graph database** (`mosaic chemin`) — documents are linked to
  entities extracted from the folder tree (configurable roles); two vector-space hops
  answer *"the other documents of the same project / the same year"*.
- **Belief memory** (`mosaic croyance`) — agents assert facts (`entity, attribute, value`);
  the most recent fact wins, history is kept, and a VSA hypervector layer provides a
  **confidence margin**. When two values compete, the memory *says so* instead of guessing —
  and the threshold for "contested" can be **conformally calibrated on your own store**
  (`croyance calibrer --alpha 0.05`: after calibration, the error rate of confident answers
  is guaranteed ≤ α).
- **Declarative per-index profile** (`mosaic profil`) — adapt the engine to *your* world in
  ten lines of JSON: folder-tree roles (client/case, project/version…), custom document
  types (`.dwg` → plan), your trade's reference-code shape. `--suggere` scans a corpus and
  proposes a profile; `--explique` explains any profile in plain language (`--langue en`).
- **Measured calibration** (`mosaic calibrer`) — encoding weights are never hand-tuned:
  provide ground-truth queries (or generate them deterministically with `--verite-auto`,
  no LLM needed) and the benchmark picks the weights, recommending a change **only if the
  gain is clear**.
- **MCP server** (`infra_mcp/`) — all of the above exposed to agent frameworks as 10 MCP
  tools, with usage guidance embedded in the tool descriptions and dynamic domain
  discovery. Zero dependencies beyond the engine itself.

## Quickstart

```bash
pip install -e ".[dev,ingest]"        # core + document conversion
# optional extras: .[ocr] (scanned documents), .[rerank] (model2vec reranker)

mosaic build ./my_documents -o ./index_docs
mosaic search "how do I wire the differential breaker" ./index_docs --top 5
mosaic explain <doc_id> ./index_docs --query "..."   # why did it match?
```

Everything is JSON on stdout — built for agents first, humans second.

### Try the bundled benchmark

A small self-contained corpus (40 French cooking articles + 12 paraphrased ground-truth
queries) lives in `bench/`:

```bash
mosaic build bench/corpus -o ./index_bench
mosaic calibrer bench/corpus --requetes bench/verite.jsonl --explique
```

On this corpus the engine reaches **11/12 top-1, 12/12 top-3** with the recommended
profile — and the calibration demo shows the optimal weights *differ* from the defaults,
which is the point: every corpus has its own optimum, and measurement finds it.

### Embeddings (optional but recommended)

The third encoding channel uses a static [model2vec](https://github.com/MinishLab/model2vec)
table distilled from fastText. Build it once, locally:

```bash
python scripts/import_wikdict.py      # optional EN->FR lexicon bridge
python scripts/prepare_potion.py      # builds the local embedding table
mosaic build ./docs -o ./index --embeddings <table.msee> --abtt 2 --rerank-vectors
```

## How it works

Each document is encoded into a **12,288-dimension grid** (64×64×3) that superposes three
channels: a deterministic SHA-seeded **signature** per token, a **co-occurrence profile**
learned from *your* corpus (PPMI + truncated SVD), and optionally a static **embedding**.
Search is a cosine against int8-quantized vectors. Separate channels carry **relations**
(hyperdimensional binding by circular permutation) and the **belief memory** (bipolar MAP
vectors). Everything is seeded, integer-quantized, and reproducible — the int8 quantization
provably absorbs cross-machine BLAS variance (measured: 0 changed values out of 24.5M).

The research behind the design decisions ships with the code (`research/`): superposition
capacity limits, multi-hop traversal viability (pure-value binding, cleanup per hop),
conformal abstention, held-out deterministic ground truth — every mechanism was measured
before it was built.

## Design principles

1. **Deterministic or explicit.** Same input, same output, any machine. What cannot be
   guaranteed is stated, never silently degraded.
2. **The engine recalls, the caller judges.** Scores, margins, provenance and explanations
   are always exposed; ambiguity raises a flag instead of a silent guess.
3. **Parameters describe *your world*, never the geometry.** Profiles configure roles,
   types and codes; encoding weights are calibrated by measurement; the core invariants
   (hashing, quantization, dimensions) are not knobs.
4. **Agent-first interfaces.** Compact JSON everywhere, usage guidance embedded in the MCP
   tool descriptions, errors that tell the caller how to fix the call.

## Development

```bash
pip install -e ".[dev]"
pytest -q            # ~500 tests, two CI regimes (with and without optional extras)
ruff check && ruff format
```

Architecture is enforced by import-linter contracts (core bricks never depend on the
orchestrator). CI runs both a lean profile (no extras) and a full one.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

The bundled French↔English lexicon derived from [WikDict](https://www.wikdict.com/) is
redistributed under its own terms (CC BY-SA); see `src/mosaic/data/LICENSE_wikdict.txt`.
