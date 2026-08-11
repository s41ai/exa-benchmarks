import asyncio
import json
import os
from typing import Any

import httpx

from .base import Searcher, SearchResult

DEFAULT_BASE_URL = "https://api.supercarl.ai"
DEFAULT_PROFILE_TEXT_MODE = "full"
DEFAULT_PROFILE_TEXT_POSTS_LIMIT = 5
DEFAULT_NATURAL_LANGUAGE_ENDPOINT = "/api/v2/search/people/query"
NATURAL_LANGUAGE_V2_MODE = "natural_language_v2"
LEGACY_DESCRIPTION_MODE = "legacy_description"
LEGACY_SOL_COMPOSITION_V2_MODE = "sol_composition_v2"
DEFAULT_SEARCH_MODE = NATURAL_LANGUAGE_V2_MODE
VALID_SEARCH_MODES = {
    NATURAL_LANGUAGE_V2_MODE,
    LEGACY_DESCRIPTION_MODE,
    LEGACY_SOL_COMPOSITION_V2_MODE,
}
VALID_NETWORK_FILTER_MODES = {"boost", "filter", "ignore", "connected_to"}
MAX_DIAGNOSTIC_BYTES = 4096


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _repair_unicode_surrogates(text: str) -> str:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return text


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return _repair_unicode_surrogates(str(value).strip())


def _bounded_diagnostic(value: Any, max_bytes: int = MAX_DIAGNOSTIC_BYTES) -> Any:
    """Keep small JSON values structured and cap oversized response diagnostics."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        encoded = _safe_text(value).encode("utf-8")

    if len(encoded) <= max_bytes:
        return value

    # Reserve enough room that the entire truncation envelope remains below max_bytes.
    preview_budget = max(0, max_bytes - 256)
    preview = encoded[:preview_budget].decode("utf-8", "ignore")
    return {
        "truncated": True,
        "original_bytes": len(encoded),
        "json_prefix": preview,
    }


class SuperCarlSearcher(Searcher):
    name = "supercarl"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        delegate_user_id: str | None = None,
        include_profile_text: bool | None = None,
        profile_text_mode: str | None = None,
        profile_text_posts_limit: int | None = None,
        network_filter_mode: str | None = None,
        search_mode: str | None = None,
        natural_language_endpoint: str | None = None,
        composition_v2_endpoint: str | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.api_key = api_key or os.getenv("SUPERCARL_API_KEY")
        if not self.api_key:
            raise ValueError("SUPERCARL_API_KEY required")

        self.base_url = (base_url or os.getenv("SUPERCARL_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.delegate_user_id = delegate_user_id or os.getenv("SUPERCARL_DELEGATE_USER_ID")
        self.include_profile_text = (
            include_profile_text
            if include_profile_text is not None
            else _env_flag("SUPERCARL_INCLUDE_PROFILE_TEXT", False)
        )
        self.profile_text_mode = (
            profile_text_mode
            or os.getenv("SUPERCARL_PROFILE_TEXT_MODE")
            or DEFAULT_PROFILE_TEXT_MODE
        )
        self.profile_text_posts_limit = profile_text_posts_limit or int(
            os.getenv(
                "SUPERCARL_PROFILE_TEXT_POSTS_LIMIT",
                str(DEFAULT_PROFILE_TEXT_POSTS_LIMIT),
            )
        )
        configured_search_mode = (
            (search_mode or os.getenv("SUPERCARL_SEARCH_MODE") or DEFAULT_SEARCH_MODE)
            .strip()
            .lower()
        )
        if configured_search_mode not in VALID_SEARCH_MODES:
            valid_modes = ", ".join(sorted(VALID_SEARCH_MODES))
            raise ValueError(f"SUPERCARL_SEARCH_MODE must be one of: {valid_modes}")
        self.search_mode = (
            NATURAL_LANGUAGE_V2_MODE
            if configured_search_mode == LEGACY_SOL_COMPOSITION_V2_MODE
            else configured_search_mode
        )

        configured_v2_endpoint = (
            natural_language_endpoint
            or composition_v2_endpoint
            or os.getenv("SUPERCARL_NATURAL_LANGUAGE_ENDPOINT")
            or os.getenv("SUPERCARL_COMPOSITION_V2_ENDPOINT")
            or DEFAULT_NATURAL_LANGUAGE_ENDPOINT
        ).strip()
        invalid_v2_endpoint = not configured_v2_endpoint.startswith(
            "/"
        ) or configured_v2_endpoint.startswith("//")
        if self.search_mode == NATURAL_LANGUAGE_V2_MODE and invalid_v2_endpoint:
            raise ValueError("SUPERCARL_NATURAL_LANGUAGE_ENDPOINT must be an absolute URL path")
        self.natural_language_endpoint = configured_v2_endpoint

        configured_network_filter_mode = network_filter_mode or os.getenv(
            "SUPERCARL_NETWORK_FILTER_MODE"
        )
        if configured_network_filter_mode is None and self.search_mode == LEGACY_DESCRIPTION_MODE:
            configured_network_filter_mode = "ignore"
        configured_network_filter_mode = (configured_network_filter_mode or "").strip().lower()
        self.network_filter_mode = (
            configured_network_filter_mode
            if configured_network_filter_mode in VALID_NETWORK_FILTER_MODES
            else None
        )
        self._client = client or httpx.AsyncClient(timeout=120.0)

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        endpoint, payload = self._build_search_request(query, num_results)

        data = await self._request("POST", endpoint, json=payload)
        users = data.get("users", [])
        diagnostics = self._collect_response_diagnostics(data)
        profile_texts = (
            self._collect_inline_profile_texts(users) if self.include_profile_text else {}
        )
        if self.include_profile_text:
            missing_users = [
                user
                for user in users
                if _safe_text(user.get("id")) and _safe_text(user.get("id")) not in profile_texts
            ]
            if missing_users:
                profile_texts.update(await self._load_profile_texts(missing_users))

        return [
            self._build_result(
                user,
                profile_texts.get(_safe_text(user.get("id"))),
                diagnostics=diagnostics,
            )
            for user in users
        ]

    def _build_search_request(self, query: str, num_results: int) -> tuple[str, dict[str, Any]]:
        if self.search_mode == NATURAL_LANGUAGE_V2_MODE:
            return self._build_natural_language_v2_request(query, num_results)
        return self._build_legacy_request(query, num_results)

    def _build_natural_language_v2_request(
        self, query: str, num_results: int
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "query": query,
            "limit": num_results,
            "offset": 0,
        }
        if self.delegate_user_id:
            payload["delegate_user_id"] = self.delegate_user_id
        if self.network_filter_mode:
            payload["network_filter_mode"] = self.network_filter_mode
        if self.include_profile_text:
            payload["include_profile_text"] = True
        return self.natural_language_endpoint, payload

    def _build_legacy_request(self, query: str, num_results: int) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "description": query,
            "limit": num_results,
            "offset": 0,
        }
        if self.delegate_user_id:
            payload["delegate_user_id"] = self.delegate_user_id
        if self.network_filter_mode:
            payload["filters"] = {"advanced": {"network_filter_mode": self.network_filter_mode}}
        if self.include_profile_text:
            payload["include_evidence_text"] = True
            payload["evidence_text_mode"] = self.profile_text_mode
            payload["evidence_posts_limit"] = self.profile_text_posts_limit

        # A/B knob: set SUPERCARL_DISABLE_KNN_UNION=1 to send the hidden param that
        # skips the kNN UNION arm server-side (the env flag stays on) — for union-off runs.
        if os.environ.get("SUPERCARL_DISABLE_KNN_UNION"):
            payload["debug_disable_knn_union"] = True

        # A/B knob for the RRF union-rerank: SUPERCARL_FORCE_RRF_FUSION=1 turns fusion ON for the
        # request (server env flag can stay OFF) — for fusion-on runs; SUPERCARL_DISABLE_RRF_FUSION=1
        # forces it OFF. Lets us measure fusion-on vs off on the same prod corpus with no global flip.
        if os.environ.get("SUPERCARL_FORCE_RRF_FUSION"):
            payload["debug_force_rrf_fusion"] = True
        if os.environ.get("SUPERCARL_DISABLE_RRF_FUSION"):
            payload["debug_disable_rrf_fusion"] = True
        return "/api/v1/search/people", payload

    def _collect_response_diagnostics(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.search_mode != NATURAL_LANGUAGE_V2_MODE:
            return {}

        diagnostics: dict[str, Any] = {}
        for key in ("request", "cohort_refs", "applied_relations"):
            if key in data and data[key] is not None:
                diagnostics[key] = _bounded_diagnostic(data[key])
        return diagnostics

    def _collect_inline_profile_texts(self, users: list[dict[str, Any]]) -> dict[str, str]:
        profile_texts: dict[str, str] = {}
        for user in users:
            user_id = _safe_text(user.get("id"))
            evidence_text = _safe_text(user.get("evidence_text"))
            if user_id and evidence_text:
                profile_texts[user_id] = evidence_text
        return profile_texts

    async def _load_profile_texts(self, users: list[dict[str, Any]]) -> dict[str, str]:
        user_ids = [_safe_text(user.get("id")) for user in users if _safe_text(user.get("id"))]
        if not user_ids:
            return {}

        results = await asyncio.gather(
            *[self._fetch_profile_text(user_id) for user_id in user_ids],
            return_exceptions=True,
        )

        profile_texts: dict[str, str] = {}
        for user_id, result in zip(user_ids, results, strict=False):
            if isinstance(result, Exception) or not result:
                continue
            profile_texts[user_id] = result
        return profile_texts

    async def _fetch_profile_text(self, user_id: str) -> str | None:
        try:
            data = await self._request(
                "GET",
                f"/api/v1/profiles/{user_id}/text",
                params={
                    "mode": self.profile_text_mode,
                    "posts_limit": self.profile_text_posts_limit,
                },
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code in {401, 403, 404}:
                return None
            raise

        text_payload = data.get("text")
        if isinstance(text_payload, dict):
            return _safe_text(text_payload.get("text")) or None
        return _safe_text(text_payload) or None

    def _build_result(
        self,
        user: dict[str, Any],
        profile_text: str | None,
        diagnostics: dict[str, Any] | None = None,
    ) -> SearchResult:
        user_id = _safe_text(user.get("id"))
        name = _safe_text(user.get("name")) or "Unknown person"
        headline = _safe_text(
            user.get("headline") or user.get("current_title") or user.get("company")
        )
        current_title = _safe_text(user.get("current_title"))
        current_company = _safe_text(user.get("current_company") or user.get("company"))
        location = _safe_text(user.get("location"))
        bio = _safe_text(user.get("bio"))
        linkedin_url = _safe_text(user.get("linkedin_url"))
        supercarl_url = _safe_text(user.get("supercarl_url"))
        profile_url = _safe_text(
            linkedin_url or supercarl_url or f"{self.base_url}/api/v1/profiles/{user_id}"
        )

        title = name
        if headline:
            title = f"{name} - {headline}"
        title = _safe_text(title)

        inline_text = self._build_inline_text(
            name=name,
            headline=headline,
            current_title=current_title,
            current_company=current_company,
            location=location,
            bio=bio,
            match_reasons=user.get("match_reasons"),
        )
        text = inline_text
        if profile_text:
            text = f"{inline_text}\n\nProfile evidence:\n{profile_text}"
        text = _safe_text(text)

        metadata = {
            "user_id": user_id or None,
            "linkedin_url": linkedin_url or None,
            "supercarl_url": supercarl_url or None,
            "social_proximity_score": user.get("social_proximity_score"),
        }
        if diagnostics:
            metadata.update(diagnostics)

        return SearchResult(
            url=profile_url,
            title=title,
            text=text,
            metadata=metadata,
        )

    def _build_inline_text(
        self,
        *,
        name: str,
        headline: str,
        current_title: str,
        current_company: str,
        location: str,
        bio: str,
        match_reasons: Any,
    ) -> str:
        parts = [f"Name: {name}"]

        if headline:
            parts.append(f"Headline: {headline}")
        if current_title or current_company:
            role_text = current_title or "Unknown title"
            if current_company:
                role_text = f"{role_text} at {current_company}"
            parts.append(f"Current role: {role_text}")
        if location:
            parts.append(f"Location: {location}")
        if bio:
            parts.append(f"Summary: {bio}")
        if isinstance(match_reasons, list) and match_reasons:
            reason_text = "; ".join(
                _safe_text(reason) for reason in match_reasons if _safe_text(reason)
            )
            if reason_text:
                parts.append(f"Match reasons: {reason_text}")

        return "\n".join(parts)

    async def _request(
        self,
        method: str,
        endpoint: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        max_retries = 5
        last_exception: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = await self._client.request(
                    method,
                    f"{self.base_url}{endpoint}",
                    headers={
                        "X-API-Key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=json,
                    params=params,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as error:
                last_exception = error
                code = error.response.status_code
                if code == 429 and attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                if code in {502, 503, 504}:
                    # Server-side timeout/unavailable (e.g. the per-request deadline
                    # returning 504): treat as empty results so the benchmark scores
                    # this query 0 and continues, instead of crashing the whole run.
                    return {}
                raise
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as error:
                last_exception = error
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                return {}

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Super Carl request failed")

    async def close(self):
        await self._client.aclose()
