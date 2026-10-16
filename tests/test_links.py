import asyncio

import httpx
import pytest

from tmg_cqa_eval.links import check_sources, probe_url, response_error


@pytest.mark.parametrize(
    ("status", "body", "kind", "expected"),
    [
        (404, "Missing", "page", "http_404"),
        (200, "<title>お探しのページは見つかりません。 | 東京都</title>", "page", "error_page"),
        (200, "<main>お探しのページを表示することができません。</main>", "page", "error_page"),
        (200, "<h1>Page not found</h1>", "page", "error_page"),
        (200, "<title>資料</title><main>Evidence</main>", "pdf", "not_pdf"),
        (200, "<title>交通局</title><main><img src='page.png'></main>", "page", None),
        (206, "%PDF-document", "pdf", None),
        (200, '<script src="/_Incapsula_Resource?token=value"></script>', "page", "access_challenge"),
        (200, '<main>Evidence</main><script src="/_Incapsula_Resource"></script>', "page", None),
    ],
)
def test_response_error(status, body, kind, expected):
    response = httpx.Response(status, content=body.encode(), request=httpx.Request("GET", "https://example.org"))
    assert response_error(response, kind) == expected


def test_range_rejection_retries_normal_get():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(416) if "Range" in request.headers else httpx.Response(200, content=b"%PDF-document")

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await probe_url(client, "https://example.org/source.pdf", "pdf")

    assert asyncio.run(check())["error"] is None
    assert len(requests) == 2


def test_source_checks_deduplicate_urls_and_skip_whole_question(monkeypatch):
    calls = []

    async def probe(client, url, kind):
        calls.append(url)
        return {"url": url, "error": "error_page" if url.endswith("missing") else None}

    monkeypatch.setattr("tmg_cqa_eval.links.probe_url", probe)
    good = {"source_id": "0", "kind": "page", "urls": ["https://example.org/ok"]}
    missing = {"source_id": "1", "kind": "page", "urls": ["https://example.org/missing"]}
    absent = {"source_id": "2", "kind": "page", "urls": []}
    records = [
        {"data_id": "good", "sources": [good]},
        {"data_id": "broken", "sources": [good, missing]},
        {"data_id": "absent", "sources": [absent]},
    ]
    checks, skipped = asyncio.run(check_sources(records))
    assert len(calls) == len(checks) == 2
    assert [row["data_id"] for row in skipped] == ["broken", "absent"]
    assert skipped[0]["sources"][0]["source_id"] == "1"
