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

The default Super Carl searcher uses the same public natural-language endpoint as the product:
`POST /api/v2/search/people/query`. It sends `query` rather than the legacy `description` field;
optional paging, network, delegate, and profile-text controls remain available.

Select the legacy description endpoint only for a historical comparison or rollback:

```bash
SUPERCARL_SEARCH_MODE=legacy_description \
uv run --env-file ../.env pbench --searchers supercarl --query-id people_role_0001
```

`legacy_description` continues to use `POST /api/v1/search/people`. The V2 endpoint can be
overridden with `SUPERCARL_NATURAL_LANGUAGE_ENDPOINT` for a canary deployment.

Optional richer grading context requests inline `evidence_text`, then falls back to
`/api/v1/profiles/:id/text` only if needed:

```bash
SUPERCARL_INCLUDE_PROFILE_TEXT=true \
uv run --env-file ../.env pbench --searchers supercarl --query-id people_role_0001
```

For full production runs, keep secrets in `../.env` and only pass non-secret knobs inline.
`../.env` should provide `OPENAI_API_KEY`, `SUPERCARL_API_KEY`, and any needed
`SUPERCARL_DELEGATE_USER_ID`.

```bash
PBENCH_SEARCHER_CONCURRENCY=10 \
PBENCH_GRADING_CONCURRENCY=50 \
SUPERCARL_BASE_URL=https://api.supercarl.ai \
SUPERCARL_SEARCH_MODE=natural_language_v2 \
SUPERCARL_INCLUDE_PROFILE_TEXT=true \
uv run --env-file ../.env pbench --searchers supercarl --output runs/supercarl-prod-full.json
```

`PBENCH_SEARCHER_CONCURRENCY` limits concurrent SuperCarl search requests.
`PBENCH_GRADING_CONCURRENCY` or `--grading-concurrency` limits concurrent OpenAI grading requests.
When `--output` is set, the benchmark also writes a per-query checkpoint at
`<output-stem>.checkpoint.jsonl`. If a run is interrupted, rerun the same command with
`--resume` to skip completed queries and write the final output JSON when the remaining
queries finish.

Full production run command:

```bash
PBENCH_SEARCHER_CONCURRENCY=10 \
PBENCH_GRADING_CONCURRENCY=50 \
SUPERCARL_BASE_URL=https://api.supercarl.ai \
SUPERCARL_SEARCH_MODE=natural_language_v2 \
SUPERCARL_INCLUDE_PROFILE_TEXT=true \
uv run --env-file ../.env pbench --searchers supercarl \
  --output runs/supercarl-prod-full.json \
  --resume
```

For deterministic subset debugging, pin both `--sample` and `--seed`:

```bash
PBENCH_SEARCHER_CONCURRENCY=1 \
PBENCH_GRADING_CONCURRENCY=5 \
SUPERCARL_SEARCH_MODE=natural_language_v2 \
SUPERCARL_INCLUDE_PROFILE_TEXT=true \
uv run --env-file ../.env pbench --searchers supercarl \
  --sample 50 \
  --seed 202608 \
  --output runs/supercarl-prod-sample-50-seed-202608.json
```

The checked-in
`data/people/manifests/simple_people_search.sample-50.seed-202608.json` freezes the query IDs
selected by `--sample 50 --seed 202608`, together with the source dataset hash. Omitting
`SUPERCARL_NETWORK_FILTER_MODE` preserves the public V2 endpoint's product-default network boost.
