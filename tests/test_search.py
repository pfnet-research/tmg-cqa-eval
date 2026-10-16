from types import SimpleNamespace

import httpx

from tmg_cqa_eval.search import BraveSearch, OpenSearch, execute


def test_brave_search_open_and_cache(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host == "api.search.brave.com":
            assert request.headers["X-Subscription-Token"] == "test-key"
            return httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {"url": "https://example.org/page", "title": "Title", "description": "<b>Snippet</b>"}
                        ]
                    }
                },
            )
        return httpx.Response(200, text="<html><script>Hidden</script><p>Evidence text</p></html>")

    backend = BraveSearch()
    backend.client.close()
    backend.client = httpx.Client(transport=httpx.MockTransport(respond))
    refs = {}
    hit = execute(backend, "search_in_local_search_engine", {"query": "Tokyo"}, refs)["results"][0]
    page_id = hit["page_id"]
    assert hit["ref"] == 1
    opened = execute(backend, "open_local_page_from_page_id", {"page_id": page_id}, refs)
    assert opened["content"] == "Evidence text" and opened["ref"] == 1
    assert (
        execute(backend, "find_keyword_in_local_page_from_page_id", {"page_id": page_id, "keyword": "evidence"}, refs)[
            "match_count"
        ]
        == 1
    )
    assert len(requests) == 2
    assert backend.get_document("unknown") is None
    backend.close()


def test_opensearch_query_and_document(monkeypatch):
    monkeypatch.setenv("OPENSEARCH_URL", "https://example.org")
    monkeypatch.setenv("OPENSEARCH_BEARER_TOKEN", "test-token")
    backend = OpenSearch()
    backend.client.close()
    backend.client = httpx.Client(
        base_url="https://example.org/",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "hits": {
                        "hits": [
                            {
                                "_id": "id",
                                "_source": {"title": "Title", "text": "Evidence"},
                                "highlight": {"text": ["Snippet"]},
                            }
                        ]
                    }
                },
            )
        ),
    )
    refs = {}
    hit = execute(backend, "search_in_local_search_engine", {"query": "東京都 budget", "max_results": 3}, refs)[
        "results"
    ][0]
    assert hit["snippet"] == "Snippet" and hit["ref"] == 1
    opened = execute(backend, "open_local_page_from_page_id", {"page_id": "id"}, refs)
    assert opened["content"] == "Evidence" and opened["ref"] == 1
    backend.close()


def test_page_refs_are_stable_across_tools_and_searches():
    backend = SimpleNamespace(
        search=lambda query, max_results: [{"page_id": page_id} for page_id in query.split()],
        get_document=lambda page_id: None if page_id == "missing" else {"text": "Evidence text"},
    )
    refs = {}
    first = execute(backend, "search_in_local_search_engine", {"query": "a b"}, refs)
    assert [hit["ref"] for hit in first["results"]] == [1, 2]
    opened = execute(backend, "open_local_page_from_page_id", {"page_id": "b"}, refs)
    assert opened["ref"] == 2
    found = execute(backend, "find_keyword_in_local_page_from_page_id", {"page_id": "a", "keyword": "text"}, refs)
    assert found["ref"] == 1 and found["matches"] == ["Evidence text"]
    assert execute(backend, "open_local_page_from_page_id", {"page_id": "missing"}, refs)["content"] is None
    second = execute(backend, "search_in_local_search_engine", {"query": "b c a"}, refs)
    assert [hit["ref"] for hit in second["results"]] == [2, 3, 1]
    fresh = execute(backend, "search_in_local_search_engine", {"query": "c"}, {})
    assert fresh["results"][0]["ref"] == 1
