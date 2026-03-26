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
