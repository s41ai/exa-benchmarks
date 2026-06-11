import argparse
import asyncio
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table
from shared.graders import RAGGrader, RetrievalGrader
from shared.searchers import Searcher, SearchResult

from .metrics import compute_rag_metrics, compute_retrieval_metrics

console = Console()
logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"
RUNS_DIR = Path(__file__).parent.parent / "runs"


@dataclass
class Query:
    query_id: str
    text: str
    track: str = ""
    bucket: str = ""
    split: str = ""
    metadata: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    gold_company_homepage: str | None = None
    constraints: dict | None = None
    expected_answer: str | None = None
    homepage: str | None = None


@dataclass
class BenchmarkConfig:
    limit: int | None = None
    query_id: str | None = None
    num_results: int = 10
    output_file: str | None = None
    checkpoint_file: str | None = None
    resume: bool = False
    enrich_exa_contents: bool = False
    track: str | None = None
    split: str | None = None
    sample: int | None = None
    seed: int | None = None
    searcher_concurrency: int = 5
    grading_concurrency: int = 50


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _env_positive_int(names: list[str], default: int) -> int:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        try:
            return _positive_int(value)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(f"{name} must be >= 1") from exc
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    return default


def load_queries(
    track: str | None = None,
    split: str | None = None,
    limit: int | None = None,
    query_id: str | None = None,
    sample: int | None = None,
    seed: int | None = None,
) -> list[Query]:
    filepath = DATA_DIR / "company" / "simple_company_search.jsonl"
    if not filepath.exists():
        return []

    queries = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            query_track = data.get("track", "")
            query_split = data.get("split", "")

            if track and query_track != track:
                continue
            if split and query_split != split:
                continue

            queries.append(
                Query(
                    query_id=data.get("query_id", ""),
                    text=data.get("text", ""),
                    track=query_track,
                    bucket=data.get("bucket", ""),
                    split=query_split,
                    metadata=data.get("metadata", {}),
                    tags=data.get("tags", []),
                    gold_company_homepage=data.get("gold_company_homepage"),
                    constraints=data.get("constraints"),
                    expected_answer=data.get("expected_answer"),
                    homepage=data.get("homepage"),
                )
            )

    if query_id:
        queries = [query for query in queries if query.query_id == query_id]

    if sample and sample < len(queries):
        import random

        rng = random.Random(seed)
        queries = sorted(rng.sample(queries, sample), key=lambda q: q.query_id)

    return queries[:limit] if limit else queries


async def fetch_exa_contents(urls: list[str], api_key: str | None = None) -> dict[str, str]:
    """Fetch page contents via Exa API."""
    api_key = api_key or os.getenv("EXA_API_KEY")
    if not api_key or not urls:
        return {}

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.exa.ai/contents",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={"urls": urls, "text": True, "livecrawl": "fallback"},
        )
        resp.raise_for_status()
        return {
            r["url"]: r["text"]
            for r in resp.json().get("results", [])
            if r.get("url") and r.get("text")
        }


async def enrich_results(results: list[SearchResult]) -> list[SearchResult]:
    """Enrich search results with full page contents."""
    try:
        contents = await fetch_exa_contents([r.url for r in results if r.url])
    except Exception as e:
        logger.warning(f"Content fetch failed: {e}")
        return results
    return [SearchResult(r.url, r.title, contents.get(r.url, r.text), r.metadata) for r in results]


def _default_checkpoint_file(output_file: str | None) -> str | None:
    if not output_file:
        return None
    path = Path(output_file)
    return str(path.with_name(f"{path.stem}.checkpoint.jsonl"))


def _load_checkpoint(
    checkpoint_file: str | None,
    searcher_name: str,
    track: str,
    query_ids: set[str],
) -> dict[str, list[dict]]:
    if not checkpoint_file:
        return {}

    path = Path(checkpoint_file)
    if not path.exists():
        return {}

    completed: dict[str, list[dict]] = {}
    with open(path) as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid checkpoint line %s in %s", line_number, path)
                continue

            if record.get("searcher") != searcher_name:
                continue
            if record.get("track", "retrieval") != track:
                continue

            query_id = record.get("query_id")
            grades = record.get("grades")
            if query_id in query_ids and isinstance(grades, list):
                completed[query_id] = grades

    return completed


def _append_checkpoint(
    checkpoint_file: str | None,
    searcher_name: str,
    track: str,
    query: Query,
    grades: list[dict],
):
    if not checkpoint_file:
        return

    path = Path(checkpoint_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "searcher": searcher_name,
        "track": track,
        "query_id": query.query_id,
        "bucket": query.bucket,
        "text": query.text,
        "result_count": len(grades),
        "grades": grades,
        "timestamp": datetime.now().isoformat(),
    }
    with open(path, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")))
        f.write("\n")
        f.flush()


@dataclass
class RunLog:
    """Log of a benchmark run with per-query results."""

    run_id: str
    timestamp: str
    config: dict
    searchers: list[str]
    retrieval_grades: list[dict] = field(default_factory=list)
    rag_grades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def save(self):
        """Save run log to runs/<run_id>.json."""
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = RUNS_DIR / f"{self.run_id}.json"
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return filepath


class Benchmark:
    """Company search benchmark runner."""

    def __init__(self, searchers: list[Searcher], grading_concurrency: int = 50):
        self.searchers = searchers
        self.retrieval_grader = RetrievalGrader()
        self.rag_grader = RAGGrader()
        self._grade_semaphore = asyncio.Semaphore(grading_concurrency)
        self._run_log: RunLog | None = None

    async def _grade_retrieval(self, query: Query, results: list[SearchResult]) -> list[dict]:
        """Grade retrieval results. Zero results count as a rank-1 miss."""
        if not results:
            return [{"query_id": query.query_id, "rank": 1, "is_match": 0, "url": None}]

        async def grade_one(rank: int, r: SearchResult) -> dict:
            async with self._grade_semaphore:
                g = await self.retrieval_grader.grade(
                    query.text,
                    r,
                    gold_homepage=query.gold_company_homepage,
                    constraints=query.constraints,
                )
            return {
                "query_id": query.query_id,
                "rank": rank,
                "is_match": g.scores.get("is_match", 0),
                "url": r.url or None,
                "title": r.title or None,
                **(
                    {"url_source": r.metadata.get("url_source")}
                    if r.metadata.get("url_source")
                    else {}
                ),
            }

        return await asyncio.gather(*[grade_one(i, r) for i, r in enumerate(results, 1)])

    async def _grade_rag(self, query: Query, answer: str) -> dict:
        """Grade RAG answer."""
        async with self._grade_semaphore:
            g = await self.rag_grader.grade(
                query.text,
                query.expected_answer or "",
                answer,
                bucket=query.bucket,
            )
        return {
            "query_id": query.query_id,
            "is_correct": g.scores.get("is_correct", 0),
        }

    async def _run_retrieval(
        self,
        searcher: Searcher,
        queries: list[Query],
        config: BenchmarkConfig,
        progress: Progress,
        task_id: TaskID,
        checkpoint_file: str | None,
    ) -> list[dict]:
        """Run retrieval track evaluation."""
        grades = []
        semaphore = asyncio.Semaphore(config.searcher_concurrency)
        checkpoint_lock = asyncio.Lock()
        completed = (
            _load_checkpoint(
                checkpoint_file,
                searcher.name,
                "retrieval",
                {q.query_id for q in queries},
            )
            if config.resume
            else {}
        )

        for q in queries:
            cached_grades = completed.get(q.query_id)
            if cached_grades is not None:
                grades.extend(cached_grades)
                progress.advance(task_id)

        async def process(q: Query):
            if q.query_id in completed:
                return
            async with semaphore:
                results = await searcher.search(q.text, config.num_results)
                if config.enrich_exa_contents:
                    results = await enrich_results(results)
                query_grades = await self._grade_retrieval(q, results)
                grades.extend(query_grades)
                async with checkpoint_lock:
                    _append_checkpoint(checkpoint_file, searcher.name, "retrieval", q, query_grades)
                progress.advance(task_id)

        await asyncio.gather(*[process(q) for q in queries])
        return grades

    async def _run_rag(
        self,
        searcher: Searcher,
        queries: list[Query],
        config: BenchmarkConfig,
        progress: Progress,
        task_id: TaskID,
        checkpoint_file: str | None,
    ) -> list[dict]:
        """Run RAG track evaluation."""
        grades = []
        semaphore = asyncio.Semaphore(config.searcher_concurrency)
        checkpoint_lock = asyncio.Lock()
        completed = (
            _load_checkpoint(
                checkpoint_file,
                searcher.name,
                "rag",
                {q.query_id for q in queries},
            )
            if config.resume
            else {}
        )

        for q in queries:
            cached_grades = completed.get(q.query_id)
            if cached_grades is not None:
                grades.extend(cached_grades)
                progress.advance(task_id)

        async def process(q: Query):
            if q.query_id in completed:
                return
            async with semaphore:
                results = await searcher.search(q.text, config.num_results)
                if config.enrich_exa_contents:
                    results = await enrich_results(results)

                combined_text = "\n\n".join(f"[{r.title}]\n{r.text}" for r in results if r.text)

                answer = await self._extract_answer(q.text, combined_text)
                grade = await self._grade_rag(q, answer)
                grades.append(grade)
                async with checkpoint_lock:
                    _append_checkpoint(checkpoint_file, searcher.name, "rag", q, [grade])
                progress.advance(task_id)

        await asyncio.gather(*[process(q) for q in queries])
        return grades

    async def _extract_answer(self, query: str, context: str) -> str:
        """Extract answer from search results using LLM."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1",
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the answer to the question from the provided context. "
                        "Give a concise, direct answer. If the answer is not found, say 'unknown'.",
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\nContext:\n{context[:30000]}",
                    },
                ],
            )
            return response.choices[0].message.content or "unknown"
        except Exception as e:
            logger.warning(f"Answer extraction failed: {e}")
            return "unknown"

    async def run(self, config: BenchmarkConfig | None = None) -> dict[str, Any]:
        """Run the benchmark."""
        config = config or BenchmarkConfig()
        queries = load_queries(
            track=config.track,
            split=config.split,
            limit=config.limit,
            query_id=config.query_id,
            sample=config.sample,
            seed=config.seed,
        )

        if not queries:
            console.print("[red]No queries found![/red]")
            return {}

        retrieval_queries = [q for q in queries if q.track == "retrieval"]
        rag_queries = [q for q in queries if q.track == "rag"]

        run_id = str(uuid.uuid4())
        checkpoint_file = config.checkpoint_file or _default_checkpoint_file(config.output_file)
        if checkpoint_file and not config.resume:
            checkpoint_path = Path(checkpoint_file)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("")

        self._run_log = RunLog(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config={
                "limit": config.limit,
                "query_id": config.query_id,
                "num_results": config.num_results,
                "enrich_exa_contents": config.enrich_exa_contents,
                "track": config.track,
                "split": config.split,
                "checkpoint_file": checkpoint_file,
                "resume": config.resume,
                "sample": config.sample,
                "seed": config.seed,
                "searcher_concurrency": config.searcher_concurrency,
                "grading_concurrency": config.grading_concurrency,
            },
            searchers=[s.name for s in self.searchers],
        )

        console.print("\n[bold]Company Search Benchmark[/bold]")
        console.print(f"  Run ID: {run_id}")
        console.print(f"  Searchers: {[s.name for s in self.searchers]}")
        console.print(f"  Retrieval queries: {len(retrieval_queries)}")
        console.print(f"  RAG queries: {len(rag_queries)}")
        console.print(f"  Exa enrichment: {'on' if config.enrich_exa_contents else 'off'}")
        if checkpoint_file:
            console.print(f"  Checkpoint: {checkpoint_file}")
            console.print(f"  Resume: {'on' if config.resume else 'off'}")
        console.print(f"  Searcher concurrency: {config.searcher_concurrency}")
        console.print(f"  Grading concurrency: {config.grading_concurrency}")
        console.print()

        results: dict[str, Any] = {"config": {"limit": config.limit}, "searchers": {}}

        with Progress(
            TextColumn("[cyan]{task.fields[name]:>10}[/cyan]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for searcher in self.searchers:
                searcher_results: dict[str, Any] = {}

                if retrieval_queries:
                    task_id = progress.add_task(
                        "", name=f"{searcher.name}-ret", total=len(retrieval_queries)
                    )
                    grades = await self._run_retrieval(
                        searcher, retrieval_queries, config, progress, task_id, checkpoint_file
                    )
                    self._run_log.retrieval_grades.extend(grades)
                    metrics = compute_retrieval_metrics(grades)
                    searcher_results["retrieval"] = {
                        "metrics": {
                            "match": metrics.match,
                            "recall_at_10": metrics.recall_at_10,
                            "precision": metrics.precision,
                            "num_queries": metrics.num_queries,
                        },
                        "grades": grades,
                    }

                if rag_queries:
                    task_id = progress.add_task(
                        "", name=f"{searcher.name}-rag", total=len(rag_queries)
                    )
                    grades = await self._run_rag(
                        searcher, rag_queries, config, progress, task_id, checkpoint_file
                    )
                    self._run_log.rag_grades.extend(grades)
                    metrics = compute_rag_metrics(grades)
                    searcher_results["rag"] = {
                        "metrics": {
                            "accuracy": metrics.accuracy,
                            "num_queries": metrics.num_queries,
                        },
                        "grades": grades,
                    }

                results["searchers"][searcher.name] = searcher_results

        self._run_log.metrics = {
            searcher: {
                track: data.get("metrics", {})
                for track, data in results["searchers"].get(searcher, {}).items()
            }
            for searcher in results.get("searchers", {})
        }
        run_file = self._run_log.save()
        console.print(f"\n[green]Run log saved to {run_file}[/green]")

        _print_summary(results)

        if config.output_file:
            with open(config.output_file, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]Saved to {config.output_file}[/green]")

        return results


def _print_summary(results: dict[str, Any]):
    """Print benchmark results summary."""
    console.print("\n[bold]Results[/bold]\n")
    searchers = results.get("searchers", {})

    if not searchers:
        return

    has_retrieval = any("retrieval" in s for s in searchers.values())
    has_rag = any("rag" in s for s in searchers.values())

    if has_retrieval:
        t = Table(title="Retrieval Track")
        t.add_column("Searcher", style="cyan")
        for col in ["R@1", "R@10", "Precision", "Queries"]:
            t.add_column(col, justify="right")

        for name, data in searchers.items():
            if "retrieval" in data:
                m = data["retrieval"].get("metrics", {})
                t.add_row(
                    name,
                    f"{m.get('match', 0):.1%}",
                    f"{m.get('recall_at_10', 0):.1%}",
                    f"{m.get('precision', 0):.1%}",
                    str(m.get("num_queries", 0)),
                )

        console.print(t)
        console.print()

    if has_rag:
        t = Table(title="RAG Track")
        t.add_column("Searcher", style="cyan")
        for col in ["Accuracy", "Queries"]:
            t.add_column(col, justify="right")

        for name, data in searchers.items():
            if "rag" in data:
                m = data["rag"].get("metrics", {})
                t.add_row(
                    name,
                    f"{m.get('accuracy', 0):.1%}",
                    str(m.get("num_queries", 0)),
                )

        console.print(t)


def _build_searcher(name: str) -> Searcher | None:
    """Build a searcher by name."""
    try:
        if name == "exa":
            from shared.searchers import ExaSearcher

            return ExaSearcher(category="company")
        if name == "supercarl":
            from shared.searchers import SuperCarlCompanySearcher

            return SuperCarlCompanySearcher()
    except (ValueError, ImportError) as e:
        console.print(f"[yellow]{name}: {e}[/yellow]")
    return None


def main():
    queries_exist = (DATA_DIR / "company" / "simple_company_search.jsonl").exists()

    if not queries_exist:
        console.print("[red]No benchmark data found![/red]")
        console.print("\nMake sure data/company/simple_company_search.jsonl exists.")
        return

    try:
        default_searcher_concurrency = _env_positive_int(
            ["CBENCH_SEARCHER_CONCURRENCY", "PBENCH_SEARCHER_CONCURRENCY"], 5
        )
        default_grading_concurrency = _env_positive_int(
            ["CBENCH_GRADING_CONCURRENCY", "PBENCH_GRADING_CONCURRENCY"], 50
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    parser = argparse.ArgumentParser(description="Company Search Benchmark")
    parser.add_argument("--limit", type=int, help="Limit number of queries")
    parser.add_argument("--query-id", help="Run a single query by query_id")
    parser.add_argument("--num-results", type=int, default=10, help="Results per query")
    parser.add_argument("--output", "-o", help="Output file for results JSON")
    parser.add_argument(
        "--checkpoint",
        help="JSONL file for per-query checkpoints "
        "(default: <output-stem>.checkpoint.jsonl when --output is set)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip queries already present in the checkpoint file",
    )
    parser.add_argument(
        "--enrich-exa-contents", action="store_true", help="Fetch page contents via Exa API"
    )
    parser.add_argument("--track", choices=["retrieval", "rag"], help="Run only specific track")
    parser.add_argument("--split", choices=["static", "dynamic"], help="Run only specific split")
    parser.add_argument(
        "--searchers",
        nargs="+",
        help="Searchers to use, space- or comma-separated (default: exa)",
    )
    parser.add_argument(
        "--sample", type=int, help="Run a deterministic random subset of N queries (use with --seed)"
    )
    parser.add_argument(
        "--seed", type=int, help="Seed for the --sample subset (deterministic when set)"
    )
    parser.add_argument(
        "--searcher-concurrency",
        type=_positive_int,
        default=default_searcher_concurrency,
        help="Concurrent search requests per searcher "
        "(default: CBENCH_SEARCHER_CONCURRENCY or 5)",
    )
    parser.add_argument(
        "--grading-concurrency",
        type=_positive_int,
        default=default_grading_concurrency,
        help="Concurrent grading requests (default: CBENCH_GRADING_CONCURRENCY or 50)",
    )
    args = parser.parse_args()

    searcher_names = [
        name
        for token in (args.searchers or ["exa"])
        for name in token.split(",")
        if name
    ]
    searchers = [s for name in searcher_names if (s := _build_searcher(name))]

    if not searchers:
        console.print("[red]No searchers available![/red]")
        return

    config = BenchmarkConfig(
        limit=args.limit,
        query_id=args.query_id,
        num_results=args.num_results,
        output_file=args.output,
        checkpoint_file=args.checkpoint,
        resume=args.resume,
        enrich_exa_contents=args.enrich_exa_contents,
        track=args.track,
        split=args.split,
        sample=args.sample,
        seed=args.seed,
        searcher_concurrency=args.searcher_concurrency,
        grading_concurrency=args.grading_concurrency,
    )
    asyncio.run(Benchmark(searchers, grading_concurrency=args.grading_concurrency).run(config))


if __name__ == "__main__":
    main()
