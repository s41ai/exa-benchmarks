# Super Carl on the simple-company-benchmark

Status: stood up 2026-06-11. Adapter + runner features landed; first 50-query
prod batch scored. Full 605 retrieval run NOT yet executed (separate decision).

## What's wired

- `shared/shared/searchers/supercarl_company.py` — `SuperCarlCompanySearcher`
  (registered as `supercarl` in this runner). POSTs
  `{SUPERCARL_BASE_URL}/api/v1/companies/search/preview` with
  `{query, preview_limit, result_mode: "detailed", include_evidence_text: true}`.
  `result_mode: "detailed"` is required — the default preview projection omits
  `website_url`/`linkedin_company_url`/`evidence_text`, and URL is what the 120
  URL-graded queries (named_lookup + disambiguation) grade on.
  Result mapping: `url` = `website_url` → `linkedin_company_url` →
  `{base}/api/v1/companies/{id}` fallback (`metadata.url_source` records which);
  `title` = company name; `text` = inline attributes (industries, HQ, headcount,
  founded, stage/status, funding round, technologies, description) +
  `evidence_text`.
- `src/benchmark.py` — ported from the people runner: empty-result grading fix
  (zero results → rank-1 `is_match: 0`, so empties count as failures),
  checkpoint/resume (`--checkpoint`, `--resume`, default
  `<output-stem>.checkpoint.jsonl`), `--sample`/`--seed`, `--query-id`,
  `--searcher-concurrency`/`--grading-concurrency`
  (env: `CBENCH_SEARCHER_CONCURRENCY`/`CBENCH_GRADING_CONCURRENCY`, falls back
  to `PBENCH_*`). Retrieval grade records now carry `url`/`title`/`url_source`
  for post-hoc analysis. `--searchers` accepts comma- or space-separated names.

## Env

- `../.env` → local dev social-connector (`http://localhost:5051`, local key).
- `../.env.prod.local` → prod (`https://api.supercarl.ai`, prod key). Both also
  carry `OPENAI_API_KEY` (graders, `gpt-5.4`) and `EXA_API_KEY` (baseline).

## Commands

```bash
cd simple-company-benchmark && uv sync

# Exa harness sanity (10 queries)
uv run --env-file ../.env cbench --searchers exa --track retrieval --limit 10 \
  --output runs/exa-retrieval-smoke.json

# Super Carl smoke (10 queries; ../.env = local, ../.env.prod.local = prod)
uv run --env-file ../.env cbench --searchers supercarl --track retrieval --limit 10 \
  --output runs/supercarl-retrieval-smoke.json

# 50-query prod batch (deterministic sample, checkpointed)
uv run --env-file ../.env.prod.local cbench --searchers supercarl,exa \
  --track retrieval --sample 50 --seed 20260611 \
  --output runs/batch50-prod-retrieval.json

# Resume an interrupted run (point at the same checkpoint)
... --checkpoint runs/batch50-prod-retrieval.checkpoint.jsonl --resume
```

## Current scores

50-query retrieval sample (seed 20260611, prod, 2026-06-11;
`runs/batch50-prod-retrieval.json`):

| Searcher  | R@1 | R@10 | Precision |
|-----------|-----|------|-----------|
| supercarl | 32.0% | 34.0% | 29.2% |
| exa       | 82.0% | 88.0% | 48.6% |

By query class (R@1 / R@10):

| Class | n | supercarl | exa |
|-------|---|-----------|-----|
| URL-graded (named_lookup + disambiguation) | 10 | 20% / 20% | 100% / 100% |
| LLM-graded (attribute buckets) | 40 | 35% / 38% | 78% / 85% |
| — industry_geo | 12 | 33% / 33% | 83% / 83% |
| — semantic | 7 | 57% / 57% | 86% / 100% |
| — founded_year | 6 | 50% / 50% | 100% / 100% |
| — employee_count | 5 | 40% / 60% | 80% / 100% |
| — composite / funding_stage / funding_amount / status_type | 10 | 10% / 10% | 50% / 60% |

Local smoke (first 10 queries, localhost:5051): supercarl 30% / 40% / 17%.

## Known blockers (measured on the prod batch)

1. **Zero-result hard filtering — 17/50 queries (34%) returned 0 results**
   (7 of 12 industry_geo, plus composite/funding/status singles). Filter
   translation ANDs `country` + generated `categories` lists; estimate comes
   back 0 (verified via direct probes: "HR tech company based in Sweden",
   "supply chain software company in Belgium" → `result_count_estimate: 0`).
   Sparse category/industry coverage (~35% locally) makes the AND empty instead
   of degrading to semantic ranking. Biggest single lever.
2. **Exact-name recall gap — named lookups miss companies that ARE indexed.**
   Sanctuary AI, Aerospacelab, SigTuple, Greenlyte, Aibidia all resolve
   correctly (with websites) via `resolve_only: true` exact resolution, but the
   ranked preview search returns unrelated companies or nothing for the same
   names. The fix is product-side (merge exact-resolution candidates into
   ranked results), not benchmark-side.
3. **Duplicate rows — 41% of returned results (120/291) are duplicates** of an
   earlier result in the same response (e.g. the same company at ranks 1-5).
   Tanks precision and crowds the top-10.
4. **Right-company-no-domain rate: 0% in this batch.** No query failed because
   the right company was returned without a `website_url`; 98% (285/291) of
   returned rows had one. The domain half of the Coresignal backfill is NOT the
   observed bottleneck on prod; the category/industry/founded half would move
   blocker #1.

## Full-run estimate (605 retrieval queries, both searchers)

- Wall clock: ~13 min supercarl + ~8 min exa at searcher-concurrency 5
  (extrapolated from 65s/40s per 50). Budget ~30 min with retries.
- Grader: 485 LLM-graded queries x ≤10 results x 2 searchers ≈ 9,000–9,700
  gpt-5.4 calls (~12M input / ~1.5M output tokens) ≈ $30–45. URL-graded
  queries are free.
- Exa: 605 searches ≈ $3. Super Carl: 605 detailed preview calls (~0.8/s) —
  modest, but each does company-row + evidence-text DB reads; run off-peak.
- RAG track (234 queries) intentionally NOT run.
