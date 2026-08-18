# Agent usage — the retrieval loop, not just the calls

This file documents a contract Mosaic's design *assumes* but never states out loud:
the engine returns ranked, explainable results — it does not decide whether a result
is good enough, nor what to try next when it isn't. That judgment belongs to the
caller. If the caller is an LLM agent, "call `search` once and read the top hit" is
not the contract this engine was built for — the tools below exist precisely because
a single query is a starting point, not an answer.

This is not architecture (see [ARCHITECTURE.md](ARCHITECTURE.md)) and not a benchmark
(see [MEASURES.md](MEASURES.md)). It is the missing piece between them: how an agent
should drive the tool, in what order, and when to stop.

## Gate 0 — enter calibrated, not blind

Before the first query of a session against an unfamiliar index, call `stats` on it —
document counts, active channels, profile. Cost is a metadata read, not a search; it
turns a guess into a scoped first query instead of a query shaped by assumption.

```
mosaic stats <index>
```

On a corpus never profiled for its domain, `profil --suggere` goes one step further:
it scans the environment and proposes a calibration (roles observed in the folder
structure, extensions to map) before the first build. Skipping this gate does not
break anything — it just means the first query is a guess that the retry loop below
will have to correct.

## Reading the engine's own signals before judging

Two fields exist specifically to shortcut the loop:

- **`conseil`** — a count/order/join question sent to `search` comes back naming the
  tool that actually answers it (`compter`, `recents`, `refs`). Results are still
  joined, but switch tools; re-querying `search` with different words won't fix a
  wrong tool choice.
- **`filtre_ecarte_tout`** — a type/date filter that excludes every result returns
  what search would have found *without* the filter, plus the types genuinely
  present in the domain. A qualified query never fails silently.

Zero results: skip straight to the retry moves below, don't insist on the same
phrasing.

## Judge against intent, not score

A result is relevant if it answers what was actually asked, not because it shares
words with the query or ranks highest. Scores across different indexes are not
comparable — never arbitrate on a raw number alone. The moment one result clearly
satisfies the intent, stop; the loop below is for when none does.

## The retry loop — fixed order, stop at the first move that works

1. **Reformulate with the words of the document you expect to find, not the words
   of your need.** A scanned invoice does not spell out its own vendor name (a logo
   isn't OCR'd) or its own document type ("BL" is never expanded to "delivery
   note"). Observed repeatedly on production corpora: a query built from the words
   literally printed on the page finds the target near the top; the same request
   dressed in tidy business vocabulary — vendor name, document type spelled out,
   product category — buries it, outranked by every other document that happens to
   share that decorative vocabulary. Short and discriminating first; decorate only
   if this fails.
2. **A known reference or code** — paste it verbatim; exact-reference matches are
   boosted automatically.
3. **A question that could live in more than one domain** — `meta`, not a manual
   loop of `search` over each index one at a time.
4. **A result present but surprising** — `explain` on it before reformulating blind.
   Understanding *why* it ranked there informs the next query; guessing again
   without that doesn't.
5. **More than one version of the same thing might exist** — `actuel` to collapse
   versions and surface the current one.
6. **A neighboring document was already found, but not the right one** — `chemin`
   from it (multi-hop: other documents sharing its folder, its time period, its
   thread).
7. **The question is about a fact, not a document** — check a belief store (if the
   corpus maintains one) before searching text for it.
8. **Last resort only** — broaden with synonyms, dates, spelling variants. Widening
   too early dilutes a query and buries the right document under noise (same
   phenomenon as move 1, in reverse).

## Stopping — honest, never infinite

After three retries from the loop above without a satisfying result, stop. Report
what was tried and that it did not resolve — not a best-effort low-confidence match
presented as the answer, and not another five reformulations on the hope the sixth
works. A retrieval engine that cannot find something is telling you something too;
looping past that signal wastes calls without raising the odds much further.

## Why this isn't encoded in the engine itself

The engine could refuse to answer below a score threshold, or auto-retry with
synonyms, or auto-widen on zero results — and in fact it already does some of this
via `conseil` and `filtre_ecarte_tout`. What it cannot do deterministically is decide
*whether a plausible-looking result actually answers what was meant* — that call
requires understanding intent, which is exactly the boundary between what the engine
computes (ranked, explainable candidates, no model in the loop at query time — see
ARCHITECTURE.md) and what the calling agent must still bring.
