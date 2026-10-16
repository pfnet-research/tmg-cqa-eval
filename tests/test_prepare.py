import hashlib
import io
from types import SimpleNamespace

import httpx
import pytest
from pypdf import PdfWriter

from tmg_cqa_eval.citations import extract_documents
from tmg_cqa_eval.data import read_rows, write_rows
from tmg_cqa_eval.links import SourceUnavailable
from tmg_cqa_eval.prepare import extract_text, main, prepare_row


def test_prepare_preserves_source_ids_and_caches_fetches(tmp_path):
    content = b"<html><nav>Navigation</nav><main><p>Evidence</p></main></html>"
    source = {
        "source_id": "7",
        "kind": "page",
        "urls": ["https://example.com/source"],
        "pages": [],
        "body_sha256": hashlib.sha256(b"Evidence").hexdigest(),
        "title": "Title",
    }
    row = {
        "data_id": "rag-test",
        "prompt": "Instructions\n### ここからドキュメント ###\n### ここまでドキュメント ###\nQuestion",
        "sources": [source],
    }
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, content=content)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        prepared = prepare_row(row, client, tmp_path)
        assert prepare_row(row, client, tmp_path) == prepared
    assert len(requests) == 1
    assert extract_documents(prepared["prompt"]) == {
        7: "タイトル: Title\nURL: https://example.com/source\n本文:\nEvidence"
    }
    assert prepared["source_preparation"][0]["matches_original_body"]
    assert prepared["prompt"].endswith("Question")


def test_missing_url_fails_without_dropping_question(tmp_path):
    with pytest.raises(ValueError, match="Missing source URL"):
        prepare_row({"data_id": "rag-test", "sources": [{"source_id": "1", "urls": []}]}, None, tmp_path)


def test_pdf_uses_physical_page_numbers(monkeypatch):
    pages = [SimpleNamespace(extract_text=lambda text=text: text) for text in ["A", "B", "C"]]
    monkeypatch.setattr("tmg_cqa_eval.prepare.PdfReader", lambda stream: SimpleNamespace(pages=pages))
    assert extract_text(b"PDF", {"kind": "pdf", "pages": [3, 1]}) == "C\nA"


def test_empty_content_container_does_not_hide_article_text():
    content = b'<html><p>Evidence</p><div id="content"></div></html>'
    assert extract_text(content, {"kind": "page"}) == "Evidence"


def test_aes_encrypted_pdf_can_be_read_without_a_password():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("", algorithm="AES-256")
    content = io.BytesIO()
    writer.write(content)
    assert extract_text(content.getvalue(), {"kind": "pdf", "pages": [1]}) == ""


@pytest.mark.parametrize("page", [0, 3])
def test_missing_pdf_page_is_unavailable(monkeypatch, page):
    pages = [SimpleNamespace(extract_text=lambda: "Evidence")] * 2
    monkeypatch.setattr("tmg_cqa_eval.prepare.PdfReader", lambda stream: SimpleNamespace(pages=pages))
    with pytest.raises(SourceUnavailable, match="document has 2 pages"):
        extract_text(b"PDF", {"kind": "pdf", "pages": [page]})


def test_pdf_page_number_alone_requires_ocr(tmp_path, monkeypatch):
    monkeypatch.setattr("tmg_cqa_eval.prepare.extract_text", lambda content, source: "- 2 -")
    source = {
        "source_id": "4",
        "kind": "pdf",
        "urls": ["https://example.com/scanned.pdf"],
        "pages": [25],
        "body_sha256": "expected-body-hash",
    }
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"%PDF-fake"))
        ) as client,
        pytest.raises(ValueError, match="run OCR"),
    ):
        prepare_row({"data_id": "rag-test", "sources": [source]}, client, tmp_path)


def test_soft_error_page_is_not_cached_as_evidence(tmp_path):
    source = {"source_id": "0", "kind": "page", "urls": ["https://example.org/missing"], "body_sha256": "hash"}
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text="<h1>お探しのページは見つかりません。</h1>")
            )
        ) as client,
        pytest.raises(SourceUnavailable, match="error_page"),
    ):
        prepare_row({"data_id": "rag-test", "sources": [source]}, client, tmp_path)
    assert not list(tmp_path.iterdir())


def test_preparation_marks_unavailable_question_and_continues(tmp_path, monkeypatch):
    dataset = tmp_path / "input.jsonl"
    output = tmp_path / "results/prepared.jsonl"
    write_rows(dataset, [{"data_id": "missing"}, {"data_id": "good"}])

    def prepare(record, client, cache, text_dir):
        if record["data_id"] == "missing":
            raise SourceUnavailable("http_404")
        return record | {"source_preparation": []}

    monkeypatch.setattr("tmg_cqa_eval.prepare.prepare_row", prepare)
    monkeypatch.setattr(
        "sys.argv",
        ["tmg-cqa-prepare", "--dataset", str(dataset), "--output", str(output), "--cache-dir", str(tmp_path)],
    )
    main()
    rows = read_rows(output)
    assert rows[0]["source_unavailable"] == "http_404"
    assert rows[1]["source_preparation"] == []
