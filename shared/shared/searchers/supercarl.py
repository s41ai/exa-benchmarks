import asyncio
import os
from typing import Any

import httpx

from .base import Searcher, SearchResult

DEFAULT_BASE_URL = "https://api.supercarl.ai"
DEFAULT_PROFILE_TEXT_MODE = "full"
DEFAULT_PROFILE_TEXT_POSTS_LIMIT = 5


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


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
    ):
        self.api_key = api_key or os.getenv("SUPERCARL_API_KEY")
        if not self.api_key:
            raise ValueError("SUPERCARL_API_KEY required")

        self.base_url = (base_url or os.getenv("SUPERCARL_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
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
        self._client = httpx.AsyncClient(timeout=120.0)

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        payload: dict[str, Any] = {
            "description": query,
            "limit": num_results,
            "offset": 0,
        }
        if self.delegate_user_id:
            payload["delegate_user_id"] = self.delegate_user_id
        if self.include_profile_text:
            payload["include_evidence_text"] = True
            payload["evidence_text_mode"] = self.profile_text_mode
            payload["evidence_posts_limit"] = self.profile_text_posts_limit

        data = await self._request("POST", "/api/v1/search/people", json=payload)
        users = data.get("users", [])
        profile_texts = self._collect_inline_profile_texts(users) if self.include_profile_text else {}
        if self.include_profile_text:
            missing_users = [
                user
                for user in users
                if _safe_text(user.get("id")) and _safe_text(user.get("id")) not in profile_texts
            ]
            if missing_users:
                profile_texts.update(await self._load_profile_texts(missing_users))

        return [self._build_result(user, profile_texts.get(_safe_text(user.get("id")))) for user in users]

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

    def _build_result(self, user: dict[str, Any], profile_text: str | None) -> SearchResult:
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
        profile_url = linkedin_url or supercarl_url or f"{self.base_url}/api/v1/profiles/{user.get('id')}"

        title = name
        if headline:
            title = f"{name} - {headline}"

        text = profile_text or self._build_inline_text(
            name=name,
            headline=headline,
            current_title=current_title,
            current_company=current_company,
            location=location,
            bio=bio,
            match_reasons=user.get("match_reasons"),
        )

        return SearchResult(
            url=profile_url,
            title=title,
            text=text,
            metadata={
                "user_id": user.get("id"),
                "linkedin_url": linkedin_url or None,
                "supercarl_url": supercarl_url or None,
                "social_proximity_score": user.get("social_proximity_score"),
            },
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
            reason_text = "; ".join(_safe_text(reason) for reason in match_reasons if _safe_text(reason))
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
                if error.response.status_code == 429 and attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as error:
                last_exception = error
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Super Carl request failed")

    async def close(self):
        await self._client.aclose()
