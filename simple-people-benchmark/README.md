# Simple People Search Benchmark

Open benchmark for evaluating people search.

## Usage

```bash
uv sync
export EXA_API_KEY=...
export OPENAI_API_KEY=...
uv run pbench --limit 10
```

## Super Carl

Use the public API by default:

```bash
uv sync
uv run --env-file ../.env pbench --searchers supercarl --query-id people_role_0001
```

Point the benchmark at a local API server:

```bash
SUPERCARL_BASE_URL=http://localhost:5050 \
uv run --env-file ../.env pbench --searchers supercarl --query-id people_role_0001
```

Optional richer grading context. This first uses inline `evidence_text` from
`/api/v1/search/people`, then falls back to `/api/v1/profiles/:id/text` only if needed:

```bash
SUPERCARL_INCLUDE_PROFILE_TEXT=true \
uv run --env-file ../.env pbench --searchers supercarl --query-id people_role_0001
```

For full production runs, keep secrets in `../.env` and only pass non-secret knobs inline.
`../.env` should provide `OPENAI_API_KEY`, `SUPERCARL_API_KEY`, and any needed
`SUPERCARL_DELEGATE_USER_ID`.

```bash
PBENCH_SEARCHER_CONCURRENCY=8 \
PBENCH_GRADING_CONCURRENCY=50 \
SUPERCARL_NETWORK_FILTER_MODE=ignore \
SUPERCARL_INCLUDE_PROFILE_TEXT=true \
uv run --env-file ../.env pbench --searchers supercarl --output runs/supercarl-prod-full.json
```

`PBENCH_SEARCHER_CONCURRENCY` limits concurrent SuperCarl search requests.
`PBENCH_GRADING_CONCURRENCY` or `--grading-concurrency` limits concurrent OpenAI grading requests.
When `--output` is set, the benchmark also writes a per-query checkpoint at
`<output-stem>.checkpoint.jsonl`. If a run is interrupted, rerun the same command with
`--resume` to skip completed queries and write the final output JSON when the remaining
queries finish.

For deterministic subset debugging, pin both `--sample` and `--seed`:

```bash
PBENCH_SEARCHER_CONCURRENCY=1 \
PBENCH_GRADING_CONCURRENCY=5 \
SUPERCARL_NETWORK_FILTER_MODE=ignore \
SUPERCARL_INCLUDE_PROFILE_TEXT=true \
uv run --env-file ../.env pbench --searchers supercarl \
  --sample 50 \
  --seed 20260527 \
  --output runs/supercarl-prod-sample-50-seed-20260527.json
```
