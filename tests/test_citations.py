import asyncio

import pytest

from tmg_cqa_eval.citations import (
    extract_documents,
    observations_from_trace,
    parse_sentences,
    score_item,
    summarize_citations,
)


def test_extract_documents_cyclic_markers():
    content = (
        "前置き\n### ここからドキュメント ###\n\n"
        "[0]\nPDF: a.pdf\n本文: A\n\n(1) タイトル: B\n本文: B\n\n"
        "{2} PDF: c.pdf\n本文: C\n\n「3」\nタイトル: D\n本文: D\n\n"
        "### ここまでドキュメント ###\n質問: x"
    )
    assert extract_documents(content) == {
        0: "PDF: a.pdf\n本文: A",
        1: "タイトル: B\n本文: B",
        2: "PDF: c.pdf\n本文: C",
        3: "タイトル: D\n本文: D",
    }


def test_extract_documents_preserves_numbered_paragraphs():
    first = "PDF: a.pdf\n本文:\nIntroduction\n(1) First item\n[0] Second item\n{9} Third item\nEvidence after the items"
    second = "タイトル: B\n本文:\nAnother introduction\n「0」 Final item\nMore evidence"
    content = f"### ここからドキュメント ###\n[0] {first}\n(1) {second}\n### ここまでドキュメント ###"
    assert extract_documents(content) == {0: first, 1: second}


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("<sentence>A<cite>0,0,1</cite></sentence>", [{"sentence": "A", "citations": [0, 1]}]),
        ("<sentence>A</sentence>", [{"sentence": "A", "citations": []}]),
        ("{sentence}A{cite}0{/cite}{/sentence}", []),
        ("Before<sentence>A<cite>0</cite></sentence>After", [{"sentence": "A", "citations": [0]}]),
        ("No sentence tags<cite>0</cite>", []),
        (
            "<|plamo:begin_sentence:plamo|>A<|plamo:begin_cite:plamo|>0<|plamo:end_cite:plamo|><|plamo:end_sentence:plamo|>",
            [{"sentence": "A", "citations": [0]}],
        ),
        (
            "<|plamo:begin_sentence:plamo|>A<cite>0</cite><|plamo:end_sentence:plamo|>",
            [{"sentence": "A", "citations": [0]}],
        ),
    ],
)
def test_sentence_tags(answer, expected):
    assert parse_sentences(answer) == expected


@pytest.mark.parametrize("numbers", [" 0", "0 ", "0, 2", "0 ,2", "0,\t2", "0,\n2"])
def test_citation_numbers_with_whitespace_are_not_extracted(numbers):
    answer = f"<sentence>A<cite>{numbers}</cite></sentence>"
    assert [sentence["citations"] for sentence in parse_sentences(answer)] == [[]]


@pytest.mark.parametrize("variant", ["rag", "tool"])
@pytest.mark.parametrize("model", ["plamo-test", "gemma-test"])
def test_all_models_and_variants_accept_general_tags(variant, model):
    class Judge:
        async def support(self, question, sentence, snippet):
            return float(bool(snippet))

        async def relevance(self, question, sentence, snippet):
            return float(bool(snippet))

    row = {
        "index": 0,
        "model": model,
        "question": "Q",
        "response": "<sentence>A<cite>1</cite></sentence>",
        "tool_trace": [{"error": "Failed"}, {"result": {"ref": 1, "content": "A"}}],
    }
    result = asyncio.run(score_item(Judge(), row, {1: "A"} if variant == "rag" else None))
    assert result["n_sentences"] == result["n_citations"] == 1
    assert result["citation_f1"] == 1.0
    assert "strict_tags" not in result and "nonconforming_tags" not in result
    assert "nonconforming_tag_rate" not in summarize_citations([result], variant)


@pytest.mark.parametrize("variant", ["rag", "tool"])
def test_invalid_citations_and_macro_f1(variant):
    class Judge:
        async def support(self, question, sentence, snippet):
            return float(bool(snippet))

        async def relevance(self, question, sentence, snippet):
            return float(bool(snippet))

    trace = [{"result": {"ref": 1, "content": "A"}}, {"error": "Failed"}, {"result": {"ref": 2, "matches": ["Other"]}}]
    assert observations_from_trace(trace) == {1: "A", 2: "Other"}
    row = {"index": 0, "question": "Q", "response": "<sentence>A<cite>0,1</cite></sentence>", "tool_trace": trace}

    async def score():
        documents = {0: "A"} if variant == "rag" else None
        valid = await score_item(Judge(), row, documents)
        empty = await score_item(Judge(), row | {"response": "No citation tags"}, documents)
        return valid, empty

    valid, empty = asyncio.run(score())
    assert valid["citation_recall"] == 1
    assert valid["citation_precision"] == 0.5
    assert valid["n_invalid_citations"] == 1
    summary = summarize_citations([valid, empty], variant)
    assert summary["citation_f1"] == 1 / 3
    assert summary["invalid_citation_rate"] == 0.5


def test_page_evidence_combines_search_open_and_find_without_duplicates():
    trace = [
        {
            "result": {
                "results": [
                    {"ref": 1, "title": "Title", "snippet": "Snippet"},
                    {"ref": 2, "snippet": "Other"},
                    {"ref": 3, "snippet": ""},
                ]
            }
        },
        {"error": "Failed"},
        {"result": {"ref": 1, "content": "Body"}},
        {"result": {"ref": 1, "matches": ["Match", "Match"]}},
        {"result": {"results": [{"ref": 1, "title": "Title", "snippet": "Snippet"}]}},
        {"result": {"ref": 2, "content": None}},
        {"result": {"content": "No page ref"}},
    ]
    assert observations_from_trace(trace) == {1: "Title\nSnippet\n\nBody\n\nMatch", 2: "Other"}


@pytest.mark.parametrize("variant", ["rag", "tool"])
def test_evidence_limits_match_each_variant(variant):
    text = "x" * 9000 + "Evidence at the end"

    class Judge:
        async def support(self, question, sentence, snippet):
            return float("Evidence at the end" in snippet)

        async def relevance(self, question, sentence, snippet):
            return float("Evidence at the end" in snippet)

    row = {
        "index": 0,
        "question": "Q",
        "response": "<sentence>A<cite>1</cite></sentence>",
        "tool_trace": [{"result": {"ref": 1, "content": text}}],
    }
    result = asyncio.run(score_item(Judge(), row, {1: text} if variant == "rag" else None))
    assert result["citation_f1"] == (0.0 if variant == "rag" else 1.0)
