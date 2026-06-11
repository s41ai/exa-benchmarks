import asyncio
import os
from typing import Any

import httpx

from .base import Searcher, SearchResult
from .supercarl import DEFAULT_BASE_URL, _safe_text

DEFAULT_RESULT_MODE = "detailed"


def _join_list(value: Any, limit: int = 8) -> str:
    if not isinstance(value, list):
        return ""
    items = [_safe_text(item) for item in value if _safe_text(item)]
    return ", ".join(items[:limit])


def _format_funding_amount(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return _safe_text(value)
    if amount <= 0:
        return ""
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:,.1f}M"
    return f"${amount:,.0f}"


class SuperCarlCompanySearcher(Searcher):
    """Company searcher backed by Super Carl's companies preview API.

    POSTs to /api/v1/companies/search/preview with result_mode="detailed" +
    include_evidence_text=true: only the detailed projection returns
    website_url / linkedin_company_url / evidence_text (the default preview
    projection omits the URL fields entirely, and URL is what the
    named_lookup/disambiguation retrieval queries grade on).
    """

    name = "supercarl"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        delegate_user_id: str | None = None,
        result_mode: str | None = None,
        include_evidence_text: bool = True,
    ):
        self.api_key = api_key or os.getenv("SUPERCARL_API_KEY")
        if not self.api_key:
            raise ValueError("SUPERCARL_API_KEY required")

        self.base_url = (base_url or os.getenv("SUPERCARL_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.delegate_user_id = delegate_user_id or os.getenv("SUPERCARL_DELEGATE_USER_ID")
        self.result_mode = (
            result_mode
            or os.getenv("SUPERCARL_COMPANY_RESULT_MODE")
            or DEFAULT_RESULT_MODE
        )
        self.include_evidence_text = include_evidence_text
        self._client = httpx.AsyncClient(timeout=120.0)

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        payload: dict[str, Any] = {
            "query": query,
            "preview_limit": num_results,
            "result_mode": self.result_mode,
            "include_evidence_text": self.include_evidence_text,
        }
        if self.delegate_user_id:
            payload["delegate_user_id"] = self.delegate_user_id

        data = await self._request("POST", "/api/v1/companies/search/preview", json=payload)
        companies = data.get("companies") or []
        return [
            self._build_result(company)
            for company in companies
            if isinstance(company, dict)
        ][:num_results]

    def _build_result(self, company: dict[str, Any]) -> SearchResult:
        company_id = _safe_text(company.get("id"))
        name = _safe_text(company.get("name") or company.get("canonical_name")) or "Unknown company"
        website_url = _safe_text(company.get("website_url"))
        linkedin_url = _safe_text(company.get("linkedin_company_url"))

        if website_url:
            url, url_source = website_url, "website"
        elif linkedin_url:
            url, url_source = linkedin_url, "linkedin"
        else:
            url, url_source = f"{self.base_url}/api/v1/companies/{company_id}", "supercarl_fallback"

        inline_text = self._build_inline_text(company, name)
        evidence_text = _safe_text(company.get("evidence_text"))
        text = inline_text
        if evidence_text:
            text = f"{inline_text}\n\nCompany evidence:\n{evidence_text}"

        return SearchResult(
            url=_safe_text(url),
            title=name,
            text=_safe_text(text),
            metadata={
                "company_id": company_id or None,
                "website_url": website_url or None,
                "linkedin_company_url": linkedin_url or None,
                "url_source": url_source,
            },
        )

    def _build_inline_text(self, company: dict[str, Any], name: str) -> str:
        parts = [f"Company: {name}"]

        industries = _join_list(company.get("industries"))
        if industries:
            parts.append(f"Industries: {industries}")

        categories = _join_list(company.get("categories"), limit=10)
        if categories:
            parts.append(f"Categories: {categories}")

        hq = _safe_text(company.get("location"))
        country = _safe_text(company.get("country"))
        if hq and country and country.lower() not in hq.lower():
            hq = f"{hq}, {country}"
        elif not hq:
            hq = country
        if hq:
            parts.append(f"HQ: {hq}")

        office_locations = _join_list(company.get("office_locations"), limit=6)
        if office_locations:
            parts.append(f"Office locations: {office_locations}")

        employee_count = company.get("employee_count")
        size_range = _safe_text(company.get("size_range"))
        if employee_count is not None:
            headcount = f"Employee count: {employee_count}"
            if size_range:
                headcount = f"{headcount} (LinkedIn size range: {size_range})"
            parts.append(headcount)
        elif size_range:
            parts.append(f"Size range: {size_range}")

        founded_year = company.get("founded_year")
        if founded_year:
            parts.append(f"Founded: {founded_year}")

        company_type = _join_list(company.get("company_type"))
        if company_type:
            parts.append(f"Company type: {company_type}")
        stage_items: list[Any] = []
        for key in ("company_stage", "company_sub_stage"):
            value = company.get(key)
            if isinstance(value, list):
                stage_items.extend(value)
        stage = _join_list(stage_items)
        if stage:
            parts.append(f"Company stage: {stage}")
        status = _join_list(company.get("company_status"))
        if status:
            parts.append(f"Company status: {status}")

        funding_round = _safe_text(
            company.get("last_funding_round_name") or company.get("last_funding_round_type")
        )
        funding_amount = _format_funding_amount(company.get("last_funding_round_amount_raised"))
        funding_date = _safe_text(company.get("last_funding_round_announced_date"))
        if funding_round or funding_amount or funding_date:
            funding_bits = [bit for bit in [funding_round, funding_amount, funding_date] if bit]
            parts.append(f"Last funding round: {' / '.join(funding_bits)}")

        revenue = company.get("revenue_usd")
        if isinstance(revenue, (int, float)) and revenue > 0:
            parts.append(f"Revenue (USD): {revenue:,.0f}")

        is_b2b = company.get("is_b2b")
        if isinstance(is_b2b, bool):
            parts.append(f"B2B: {'yes' if is_b2b else 'no'}")

        technologies = _join_list(company.get("technologies"), limit=12)
        if technologies:
            parts.append(f"Technologies: {technologies}")

        website_url = _safe_text(company.get("website_url"))
        if website_url:
            parts.append(f"Website: {website_url}")
        linkedin_url = _safe_text(company.get("linkedin_company_url"))
        if linkedin_url:
            parts.append(f"LinkedIn: {linkedin_url}")

        description = _safe_text(company.get("description"))
        if description:
            parts.append(f"Description: {description}")

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
        raise RuntimeError("Super Carl company request failed")

    async def close(self):
        await self._client.aclose()
