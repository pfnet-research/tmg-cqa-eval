"""Two small search adapters exposing search() and get_document()."""

import hashlib
import io
import os
import re

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader


class BraveSearch:
    def __init__(self):
        self.client = httpx.Client(timeout=60, follow_redirects=True)
        self.pages = {}

    def search(self, query, max_results):
        response = self.client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": os.environ.get("BRAVE_SEARCH_API_KEY") or os.environ["BRAVE_API_KEY"]},
            params={"q": query, "count": min(max_results, 20), "country": "JP", "search_lang": "ja"},
        )
        response.raise_for_status()
        hits = []
        for item in response.json().get("web", {}).get("results", []):
            page_id = hashlib.sha1(item["url"].encode()).hexdigest()
            self.pages[page_id] = {"url": item["url"], "title": item["title"]}
            hits.append(
                {
                    "page_id": page_id,
                    "title": item["title"],
                    "snippet": BeautifulSoup(item.get("description", ""), "html.parser").get_text(),
                    "url": item["url"],
                }
            )
        return hits

    def get_document(self, page_id):
        page = self.pages.get(page_id)
        if page is None:
            return None
        if "text" not in page:
            response = self.client.get(page["url"])
            response.raise_for_status()
            if "application/pdf" in response.headers.get("content-type", "") or response.content.startswith(b"%PDF-"):
                page["text"] = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(response.content)).pages)
            else:
                soup = BeautifulSoup(response.text, "html.parser")
                for element in soup(["script", "style", "noscript"]):
                    element.decompose()
                page["text"] = soup.get_text(" ", strip=True)
        return {"page_id": page_id, **page}

    def close(self):
        self.client.close()


class OpenSearch:
    def __init__(self):
        self.client = httpx.Client(
            base_url=os.environ["OPENSEARCH_URL"].rstrip("/") + "/",
            headers={"Authorization": f"Bearer {os.environ['OPENSEARCH_BEARER_TOKEN']}"},
            timeout=60,
        )
        self.index = os.environ.get("OPENSEARCH_INDEX", "tmg")
        self.pages = {}

    def request(self, body):
        response = self.client.post(f"{self.index}/_search", json=body, params={"search_type": "dfs_query_then_fetch"})
        response.raise_for_status()
        return response.json()["hits"]["hits"]

    def search(self, query, max_results):
        words = []
        for segment in query.split():
            words.extend(re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]*", segment) if segment.isascii() else [segment])
        words = words[:16] or [query]

        def condition(word, strict):
            body = {"query": word, "fields": ["text"], "type": "phrase" if word.isascii() else "best_fields"}
            if strict and not word.isascii():
                body["operator"] = "and"
            return {"multi_match": body}

        body = {
            "query": {
                "bool": {
                    "should": [
                        {"bool": {"must": [condition(w, True) for w in words]}},
                        {"bool": {"should": [condition(w, False) for w in words]}},
                    ]
                }
            },
            "size": max_results,
            "_source": ["title"],
            "highlight": {
                "fields": {"text": {}},
                "fragment_size": 700,
                "number_of_fragments": 1,
                "pre_tags": [""],
                "post_tags": [""],
            },
        }
        return [
            {
                "page_id": h["_id"],
                "title": h["_source"].get("title", ""),
                "snippet": " ".join(h.get("highlight", {}).get("text", [])),
            }
            for h in self.request(body)
        ]

    def get_document(self, page_id):
        if page_id not in self.pages:
            hits = self.request({"query": {"ids": {"values": [page_id]}}, "size": 1})
            if not hits:
                return None
            self.pages[page_id] = hits[0]["_source"]
        return {"page_id": page_id, **self.pages[page_id]}

    def close(self):
        self.client.close()


def execute(backend, name, arguments, ref_by_page_id):
    """Assign stable page refs within one question and return search/open/find evidence."""
    if name == "search_in_local_search_engine":
        args = {"max_results": 10, **arguments}
        hits = backend.search(args["query"], args["max_results"])
        results = []
        for hit in hits:
            page_id = str(hit["page_id"])
            ref = ref_by_page_id.setdefault(page_id, len(ref_by_page_id) + 1)
            results.append({"ref": ref, **hit})
        return {"results": results, "arguments": args}
    if name not in {"open_local_page_from_page_id", "find_keyword_in_local_page_from_page_id"}:
        raise ValueError(f"Unknown tool: {name}")
    page = backend.get_document(arguments["page_id"])
    text = " ".join(re.sub(r"<[^>]+>", " ", page["text"]).split()) if page else ""
    ref = ref_by_page_id.get(str(arguments["page_id"]))
    page_ref = {"ref": ref} if ref is not None else {}
    if name == "open_local_page_from_page_id":
        args = {"max_chars": 32000, **arguments}
        return {**page_ref, "content": text[: args["max_chars"]] if page else None, "arguments": args}
    args = {"diameter": 100, "max_matches": 20, "offset": 0, **arguments}
    keyword = " ".join(args["keyword"].split())
    matches = list(re.finditer(re.escape(keyword), text, re.IGNORECASE))
    radius = args["diameter"] // 2
    windows = [
        text[max(0, m.start() - radius) : min(len(text), m.end() + radius)].strip()
        for m in matches[args["offset"] : args["offset"] + args["max_matches"]]
    ]
    return {
        **page_ref,
        "normalized_keyword": keyword,
        "match_count": len(matches),
        "matches": windows,
        "arguments": args,
    }
