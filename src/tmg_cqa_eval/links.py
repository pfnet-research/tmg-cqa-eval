"""Check source availability, including error pages returned with HTTP 200."""

import asyncio
import re
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup

ERROR_PAGE = re.compile(
    r"(?:お探し|指定|要求|アクセス)[^。\n]{0,40}(?:ページ|ファイル)[^。\n]{0,40}"
    r"(?:見つかりません|存在しません|表示できません|表示することができません|削除)"
    r"|ページが見つかりません|ページは見つかりません|ページが存在しません"
    r"|page\s+not\s+found|requested\s+page\s+could\s+not\s+be\s+found|404\s*(?:error|not found)",
    re.IGNORECASE,
)


def response_error(response, kind="page"):
    if response.is_error:
        return f"http_{response.status_code}"
    if response.content.startswith(b"%PDF-"):
        return None
    soup = BeautifulSoup(response.content, "html.parser")
    headings = " ".join(element.get_text(" ", strip=True) for element in soup.select("title, h1, h2"))
    main = soup.select_one("main, article, #content, #contents") or soup
    text = main.get_text(" ", strip=True)
    if not text and soup.find("script", src=re.compile("_Incapsula_Resource")) and not soup.find(["img", "iframe"]):
        return "access_challenge"
    if ERROR_PAGE.search(headings) or (len(text) < 2000 and ERROR_PAGE.search(text)):
        return "error_page"
    if not response.content.strip():
        return "empty_response"
    if kind == "pdf":
        return "not_pdf"
    return None


async def probe_url(client, url, kind="page"):
    try:
        for headers in ({"Range": "bytes=0-262143"}, {}):
            async with client.stream("GET", url, headers=headers) as response:
                if response.status_code in {403, 416} and headers:
                    continue
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if content.startswith(b"%PDF-") or len(content) >= 262144:
                        break
                snapshot = httpx.Response(response.status_code, content=bytes(content), request=response.request)
                error = response_error(snapshot, kind)
                if urlsplit(url).path not in {"", "/"} and urlsplit(str(response.url)).path == "/":
                    error = error or "redirected_to_homepage"
                return {"url": url, "final_url": str(response.url), "status_code": response.status_code, "error": error}
    except httpx.HTTPError as exc:
        return {"url": url, "error": f"request_error:{type(exc).__name__}"}


def unavailable_sources(record, checks):
    return [
        {"source_id": source["source_id"], "urls": source["urls"]}
        for source in record.get("sources", [])
        if not source["urls"] or all(checks[url]["error"] for url in source["urls"])
    ]


async def check_sources(records):
    urls = {url: source["kind"] for row in records for source in row.get("sources", []) for url in source["urls"]}
    semaphore = asyncio.Semaphore(16)
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:

        async def one(url, kind):
            async with semaphore:
                return await probe_url(client, url, kind)

        results = await asyncio.gather(*(one(url, kind) for url, kind in urls.items()))
    checks = {result["url"]: result for result in results}
    return checks, [
        {"data_id": row["data_id"], "reason": "source_unavailable", "sources": missing}
        for row in records
        if (missing := unavailable_sources(row, checks)) or row.get("source_unavailable")
    ]


class SourceUnavailable(ValueError):
    """The source cannot currently be retrieved as a document."""
