import json
import os
import unittest
from unittest.mock import patch

from shared.searchers.supercarl import MAX_DIAGNOSTIC_BYTES, SuperCarlSearcher


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _RecordingClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.closed = False

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return _StubResponse(self._responses.pop(0))

    async def aclose(self):
        self.closed = True


class SuperCarlSearcherModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_mode_uses_v2_query_ingress_without_legacy_fields(self):
        client = _RecordingClient([{"users": []}])
        with patch.dict(os.environ, {}, clear=True):
            searcher = SuperCarlSearcher(
                api_key="test-key",
                base_url="https://example.test",
                client=client,
            )
            await searcher.search("policy manager based in sf at gaming", 7)
            await searcher.close()

        self.assertEqual(
            client.calls,
            [
                (
                    "POST",
                    "https://example.test/api/v2/search/people/query",
                    {
                        "headers": {
                            "X-API-Key": "test-key",
                            "Content-Type": "application/json",
                        },
                        "json": {
                            "query": "policy manager based in sf at gaming",
                            "limit": 7,
                            "offset": 0,
                        },
                        "params": None,
                    },
                )
            ],
        )
        self.assertTrue(client.closed)

    def test_legacy_mode_preserves_delegate_evidence_and_debug_options(self):
        client = _RecordingClient([])
        with patch.dict(
            os.environ,
            {
                "SUPERCARL_DISABLE_KNN_UNION": "1",
                "SUPERCARL_FORCE_RRF_FUSION": "1",
                "SUPERCARL_DISABLE_RRF_FUSION": "1",
            },
            clear=True,
        ):
            searcher = SuperCarlSearcher(
                api_key="test-key",
                search_mode="legacy_description",
                delegate_user_id="delegate-1",
                include_profile_text=True,
                profile_text_mode="summary",
                profile_text_posts_limit=3,
                network_filter_mode="boost",
                client=client,
            )
            endpoint, payload = searcher._build_search_request("query", 4)

        self.assertEqual(endpoint, "/api/v1/search/people")
        self.assertEqual(
            payload,
            {
                "description": "query",
                "limit": 4,
                "offset": 0,
                "delegate_user_id": "delegate-1",
                "filters": {"advanced": {"network_filter_mode": "boost"}},
                "include_evidence_text": True,
                "evidence_text_mode": "summary",
                "evidence_posts_limit": 3,
                "debug_disable_knn_union": True,
                "debug_force_rrf_fusion": True,
                "debug_disable_rrf_fusion": True,
            },
        )

    async def test_v2_mode_uses_the_explicit_query_contract_and_flat_controls(self):
        client = _RecordingClient(
            [
                {
                    "users": [
                        {
                            "id": "person-1",
                            "name": "Ada Example",
                            "current_title": "Policy Manager",
                            "evidence_text": "Grounded profile evidence",
                        }
                    ],
                    "request": {
                        "input_mode": "query",
                        "applied_request": {"filters": {"people": {"where": {}}}},
                    },
                    "cohort_refs": [{"key": "cohort_v2_opaque"}],
                }
            ]
        )
        with patch.dict(
            os.environ,
            {
                "SUPERCARL_DISABLE_KNN_UNION": "1",
                "SUPERCARL_FORCE_RRF_FUSION": "1",
                "SUPERCARL_SEARCH_MODE": "natural_language_v2",
                "SUPERCARL_NATURAL_LANGUAGE_ENDPOINT": "/api/v2/search/people/query-canary",
            },
            clear=True,
        ):
            searcher = SuperCarlSearcher(
                api_key="test-key",
                base_url="https://example.test",
                delegate_user_id="delegate-1",
                include_profile_text=True,
                network_filter_mode="ignore",
                client=client,
            )
            results = await searcher.search("policy manager based in sf at gaming", 10)

        self.assertEqual(len(client.calls), 1)
        method, url, kwargs = client.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(url, "https://example.test/api/v2/search/people/query-canary")
        self.assertEqual(
            kwargs["json"],
            {
                "query": "policy manager based in sf at gaming",
                "limit": 10,
                "offset": 0,
                "delegate_user_id": "delegate-1",
                "network_filter_mode": "ignore",
                "include_profile_text": True,
            },
        )
        self.assertNotIn("description", kwargs["json"])
        self.assertNotIn("filters", kwargs["json"])
        self.assertNotIn("debug_disable_knn_union", kwargs["json"])
        self.assertNotIn("debug_force_rrf_fusion", kwargs["json"])
        self.assertEqual(results[0].metadata["request"]["input_mode"], "query")
        self.assertEqual(results[0].metadata["cohort_refs"][0]["key"], "cohort_v2_opaque")
        self.assertIn("Grounded profile evidence", results[0].text)

    async def test_v2_parser_keeps_users_canonical_and_does_not_guess_results_shape(self):
        client = _RecordingClient([{"results": [{"id": "not-a-canonical-user"}]}])
        searcher = SuperCarlSearcher(
            api_key="test-key",
            search_mode="natural_language_v2",
            client=client,
        )

        results = await searcher.search("query")

        self.assertEqual(results, [])
        self.assertEqual(
            client.calls[0][1],
            "https://api.supercarl.ai/api/v2/search/people/query",
        )

    def test_v2_diagnostics_are_bounded(self):
        client = _RecordingClient([])
        searcher = SuperCarlSearcher(
            api_key="test-key",
            search_mode="natural_language_v2",
            client=client,
        )
        diagnostics = searcher._collect_response_diagnostics(
            {
                "request": {"applied_request": {"filters": "x" * (MAX_DIAGNOSTIC_BYTES * 2)}},
                "cohort_refs": [{"small": True}],
            }
        )

        request = diagnostics["request"]
        self.assertTrue(request["truncated"])
        self.assertGreater(request["original_bytes"], MAX_DIAGNOSTIC_BYTES)
        self.assertLessEqual(
            len(json.dumps(request, ensure_ascii=False).encode("utf-8")),
            MAX_DIAGNOSTIC_BYTES,
        )
        self.assertEqual(diagnostics["cohort_refs"], [{"small": True}])

    def test_legacy_sol_composition_alias_still_selects_v2(self):
        searcher = SuperCarlSearcher(
            api_key="test-key",
            search_mode="sol_composition_v2",
            client=_RecordingClient([]),
        )

        self.assertEqual(searcher.search_mode, "natural_language_v2")
        self.assertEqual(
            searcher._build_search_request("query", 10),
            (
                "/api/v2/search/people/query",
                {"query": "query", "limit": 10, "offset": 0},
            ),
        )

    def test_invalid_mode_and_non_path_endpoint_fail_fast(self):
        client = _RecordingClient([])
        with self.assertRaisesRegex(ValueError, "SUPERCARL_SEARCH_MODE"):
            SuperCarlSearcher(api_key="test-key", search_mode="v2-ish", client=client)

        with self.assertRaisesRegex(ValueError, "SUPERCARL_NATURAL_LANGUAGE_ENDPOINT"):
            SuperCarlSearcher(
                api_key="test-key",
                search_mode="natural_language_v2",
                natural_language_endpoint="https://other.example/api/v2/search/people/query",
                client=client,
            )


if __name__ == "__main__":
    unittest.main()
