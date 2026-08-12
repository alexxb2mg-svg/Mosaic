# Contributing

Mosaic is a didactic experiment that runs in production — contributions feed an
**educational resource** more than a product roadmap. That framing is not a
disclaimer, it is the contribution guide in one sentence: what makes this repository
useful is that every claim in it is measured, and every measure is replayable.

## The one rule: measure

- **Beat a number on its own bench and the result belongs here, whichever way it
  points.** The benches live in `bench/` and `research/`; the campaign log is
  [docs/MEASURES.md](docs/MEASURES.md).
- A feature proposal without a bench is a hypothesis, and welcome as an issue. A
  feature PR without a bench will be asked for one.
- Defeats are first-class contributions. If you can prove a design choice wrong —
  a stronger baseline, a terrain where a documented setup loses — that proof is
  exactly the kind of material this repository exists to collect.

## Ground rules of the engine

- **Determinism is a contract**: same corpus, same index, same ranking, on any
  machine. Seeded randomness only; no network access at runtime; anything that
  cannot be guaranteed must be stated, never silently degraded.
- **No LLM in the loop** — the engine stays mechanical; judgment belongs to the
  caller.
- **Loud refusals** over silent degradation: an inoperative flag is an error, not
  a no-op.

## Practicalities

```bash
pip install -e ".[dev]"
pytest -q          # the suite runs with RuntimeWarning promoted to errors
ruff check && ruff format
```

- Architecture boundaries are enforced by import-linter; CI runs a lean profile
  (no extras) and a full one.
- Code comments and docstrings are largely in French — the engine is French-first
  and so is its inner voice. English contributions are entirely welcome; do not
  translate existing docstrings as a side effect of a change.
- Keep PRs single-subject, with the measure (or the test) that motivated them.
