import hashlib
import json
import random
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = BENCHMARK_ROOT / "data" / "people" / "simple_people_search.jsonl"
MANIFEST_PATH = (
    BENCHMARK_ROOT
    / "data"
    / "people"
    / "manifests"
    / "simple_people_search.sample-50.seed-202608.json"
)


class FrozenPeopleSampleManifestTests(unittest.TestCase):
    def test_manifest_matches_the_canonical_seeded_sample(self):
        manifest = json.loads(MANIFEST_PATH.read_text())
        source_bytes = SOURCE_PATH.read_bytes()
        rows = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
        selected = sorted(
            random.Random(manifest["selection"]["seed"]).sample(
                rows, manifest["selection"]["sample_size"]
            ),
            key=lambda row: row["query_id"],
        )

        self.assertEqual(manifest["source"], "data/people/simple_people_search.jsonl")
        self.assertEqual(manifest["source_query_count"], len(rows))
        self.assertEqual(manifest["source_sha256"], hashlib.sha256(source_bytes).hexdigest())
        self.assertEqual(
            manifest["query_ids"],
            [row["query_id"] for row in selected],
        )
        self.assertEqual(len(manifest["query_ids"]), 50)
        self.assertEqual(len(set(manifest["query_ids"])), 50)


if __name__ == "__main__":
    unittest.main()
