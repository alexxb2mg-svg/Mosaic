# `mosaic` — MCP server

A **stdio** MCP server, hand-written, with **zero dependency** on the official `mcp` SDK
(JSON-RPC 2.0, newline-delimited JSON — one JSON object per line, both ways, no
Content-Length framing). It exposes 11 tools over your Mosaic indexes and keeps every
opened index **cached in memory** — that is the point of the server over the CLI: a CLI
`mosaic search` reloads the index on every call (~1–2 s), the server opens it once and
answers in ~50 ms. A rebuilt index is picked up automatically (mtime check).

Single file: `infra_mcp/mosaic_mcp.py`. Pure, IO-free dispatch, fully testable:
`handle_request(request, state) -> response | None` (see `tests/test_mcp.py`).

## Tools

| Tool | What it does |
|---|---|
| `mosaic_search` | semantic search (paraphrase-friendly) + facets: `type`, `recence`, automatic exact-reference boost; opt-in `fusion` (RRF channels) and `grammatical` (role-aware scoring) |
| `mosaic_explain` | why did this document match? (token contributions) |
| `mosaic_like` | use a whole document as the query |
| `mosaic_meta` | query several domains at once, rank-fused (RRF) with provenance |
| `mosaic_actuel` | temporal truth: newest version canonical, older ones flagged stale |
| `mosaic_chemin` | multi-hop traversal: doc → its entities → sibling documents |
| `mosaic_stats` | domain discovery: doc count, active channels, index profile |
| `mosaic_diff` | semantic diff between two domains: vocabulary drift, usage drift, changed documents |
| `mosaic_croyance_assert` / `courant` / `historique` | belief memory: assert facts, read current truth (with calibrated confidence), full history |

## Setup

Domains are **discovered dynamically**: every `index_<name>` directory under the data root
is a queryable domain — drop a new index there and it is available without touching the
server. Configure through environment variables:

- `MOSAIC_MCP_DATA_DIR` — root directory containing your `index_*` folders
  (default: `~/.mosaic`)
- `MOSAIC_POTION_MODEL_DIR` — optional, path to the local model2vec model used by
  `--rerank` (only needed if your indexes were built with `--rerank-vectors`)

Register the server with your MCP client using
`claude_desktop_config_snippet.json` as a template (adjust the absolute paths).
