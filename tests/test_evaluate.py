import asyncio
import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI

from tmg_cqa_eval import evaluate
from tmg_cqa_eval.citations import Judge
from tmg_cqa_eval.data import read_rows, write_rows
from tmg_cqa_eval.evaluate import generate_one, judge_answer, run, summarize


def test_accuracy_includes_failed_generations():
    result = summarize([{"verdict": "consistent"}, {"verdict": "generation_failed"}], "rag", "judge")
    assert result["accuracy"] == 0.5
    assert result["accuracy_given_judged"] == 1


@pytest.mark.parametrize(
    "variant,model,citation_mode,expected",
    [
        ("rag", "gemma-test", "general", "<sentence>A<cite>1</cite></sentence>"),
        ("rag", "plamo-3.0-prime", "general", "<sentence>A<cite>1</cite></sentence>"),
        (
            "rag",
            "custom-served-name",
            "plamo",
            "<|plamo:begin_sentence:plamo|>A<|plamo:begin_cite:plamo|>1<|plamo:end_cite:plamo|><|plamo:end_sentence:plamo|>",
        ),
        (
            "rag",
            "PLaMo-test",
            "plamo",
            "<|plamo:begin_sentence:plamo|>A<|plamo:begin_cite:plamo|>1<|plamo:end_cite:plamo|><|plamo:end_sentence:plamo|>",
        ),
        ("tool", "gemma-test", "general", "<sentence>A<cite>1</cite></sentence>"),
        ("tool", "plamo-3.0-prime", "general", "<sentence>A<cite>1</cite></sentence>"),
        (
            "tool",
            "custom-served-name",
            "plamo",
            "<|plamo:begin_sentence:plamo|>A<|plamo:begin_cite:plamo|>1<|plamo:end_cite:plamo|><|plamo:end_sentence:plamo|>",
        ),
    ],
)
def test_generation_citation_instruction(monkeypatch, variant, model, citation_mode, expected):
    monkeypatch.setattr("tmg_cqa_eval.evaluate.BraveSearch", lambda: SimpleNamespace(close=lambda: None))
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer", tool_calls=None), finish_reason="stop")]
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    args = Namespace(
        variant=variant,
        model=model,
        citation_mode=citation_mode,
        search="brave",
        max_tokens=128,
        max_tool_steps=1,
        temperature=0,
        top_p=1,
        top_k=-1,
        min_p=0,
        repetition_penalty=1,
        presence_penalty=0,
        frequency_penalty=0,
    )
    prompt = (
        "<|plamo:begin_sentence:plamo|>A<|plamo:begin_cite:plamo|>1<|plamo:end_cite:plamo|><|plamo:end_sentence:plamo|>"
    )
    record = {"data_id": "test", "question": "Q", "reference_answer": "A", "prompt": prompt}
    asyncio.run(generate_one(client, args, record, 0))
    assert create.call_args.kwargs["messages"][0]["content"] == expected
    kwargs = create.call_args.kwargs["extra_body"].get("chat_template_kwargs")
    assert kwargs == ({"enable_citation": True} if citation_mode == "plamo" else None)


def test_invalid_judgment_retains_example():
    judge = SimpleNamespace(query=AsyncMock(return_value="not JSON"))
    row = {"index": 0, "question": "Q", "reference_answer": "A", "response": "Candidate"}
    result = asyncio.run(judge_answer(judge, Namespace(variant="rag"), row))
    assert result["verdict"] == "judge_error"
    assert summarize([result], "rag", "judge")["n_total"] == 1


def test_tool_budget_forces_final_answer_and_preserves_order(monkeypatch):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        call = {
            "id": "call-0",
            "type": "function",
            "function": {"name": "search_in_local_search_engine", "arguments": '{"query":"Q"}'},
        }
        message = (
            {"role": "assistant", "content": "Answer"}
            if len(requests) == 2
            else {
                "role": "assistant",
                "content": None,
                "tool_calls": [call],
            }
        )
        return httpx.Response(200, json={"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]})

    monkeypatch.setattr(
        "tmg_cqa_eval.evaluate.BraveSearch",
        lambda: SimpleNamespace(
            search=lambda query, max_results: [{"page_id": "page", "snippet": "Evidence"}],
            close=lambda: None,
        ),
    )
    args = Namespace(
        model="plamo-example",
        citation_mode="plamo",
        variant="tool",
        search="brave",
        max_tokens=20000,
        max_tool_steps=1,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.01,
        repetition_penalty=1.1,
        presence_penalty=0.2,
        frequency_penalty=-0.1,
    )
    record = {
        "data_id": "tool-0",
        "question": "Q",
        "reference_answer": "A",
        "prompt": "Answer Q and cite each page's ref.",
    }

    async def generate():
        async with AsyncOpenAI(
            api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        ) as client:
            return await generate_one(client, args, record, 0)

    result = asyncio.run(generate())
    for request in requests:
        assert request["temperature"] == 0.6
        assert request["top_p"] == 0.95 and request["top_k"] == 20
        assert request["min_p"] == 0.01 and request["repetition_penalty"] == 1.1
        assert request["presence_penalty"] == 0.2 and request["frequency_penalty"] == -0.1
        assert request["chat_template_kwargs"] == {"enable_citation": True}
    assert result["response"] == "Answer" and result["budget_exhausted"]
    assert result["tool_trace"][0]["result"]["results"][0]["snippet"] == "Evidence"
    assert result["tool_trace"][0]["result"]["results"][0]["ref"] == 1
    assert requests[0]["max_completion_tokens"] == 16384
    assert requests[1]["tool_choice"] == "none"
    assert requests[1]["messages"][-1]["content"] == "Give the single best final answer now."
    assert requests[0]["messages"][0]["content"] == record["prompt"]


@pytest.mark.parametrize("temperature,top_p,top_k", [(0.6, 0.95, 20), (0.0, 1.0, -1)])
def test_rag_decode_settings_survive_context_retry_and_leave_judge_greedy(temperature, top_p, top_k):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                400, json={"error": {"message": "maximum context length is 8192 tokens with 4096 input tokens"}}
            )
        return httpx.Response(
            200, json={"choices": [{"index": 0, "message": {"content": "Answer"}, "finish_reason": "stop"}]}
        )

    args = Namespace(
        model="plamo-example",
        variant="rag",
        citation_mode="plamo",
        search="brave",
        max_tokens=8192,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=0.01,
        repetition_penalty=1.1,
        presence_penalty=0.2,
        frequency_penalty=-0.1,
    )
    record = {"data_id": "rag-0", "question": "Q", "reference_answer": "A", "prompt": "Prompt"}

    async def generate_and_judge():
        async with AsyncOpenAI(
            api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        ) as client:
            result = await generate_one(client, args, record, 0)
            await Judge(client, "judge", asyncio.Semaphore(1), "rag", {}).query("Judge this answer")
            return result

    assert asyncio.run(generate_and_judge())["response"] == "Answer"
    assert [request["max_tokens"] for request in requests[:2]] == [8192, 4032]
    for request in requests[:2]:
        assert request["temperature"] == temperature
        assert request["top_p"] == top_p and request["top_k"] == top_k
        assert request["min_p"] == 0.01 and request["repetition_penalty"] == 1.1
        assert request["presence_penalty"] == 0.2 and request["frequency_penalty"] == -0.1
        assert request["chat_template_kwargs"] == {"enable_citation": True}
    assert requests[2]["temperature"] == 0
    assert "top_k" not in requests[2]


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], (0.0, 1.0, -1)),
        (["--decode", "reasoning"], (0.6, 0.95, 20)),
        (["--decode", "reasoning", "--temperature", "0.8", "--top-k", "40"], (0.8, 0.95, 40)),
    ],
)
@pytest.mark.parametrize("variant", ["rag", "tool"])
def test_cli_resolves_decode_preset_and_overrides(monkeypatch, flags, expected, variant):
    execute = AsyncMock()
    monkeypatch.setattr(evaluate, "run", execute)
    monkeypatch.setattr(
        "sys.argv", ["tmg-cqa-eval", "--variant", variant, "--dataset", "data.jsonl", "--output", "results", *flags]
    )
    evaluate.main()
    args = execute.call_args.args[0]
    assert (args.temperature, args.top_p, args.top_k) == expected
    assert args.max_tokens == 16384
    assert (args.min_p, args.repetition_penalty, args.presence_penalty, args.frequency_penalty) == (0.0, 1.0, 0.0, 0.0)


@pytest.mark.parametrize("variant", ["rag", "tool"])
def test_unavailable_sources_skip_generation_scoring_and_denominators(tmp_path, monkeypatch, variant):
    dataset = tmp_path / "data.jsonl"
    row = {"question": "Q", "reference_answer": "A", "prompt": "Prompt"}
    records = [row | {"data_id": data_id} for data_id in ["missing", "good"]]
    write_rows(dataset, records)
    args = Namespace(
        dataset=dataset,
        output=tmp_path / "results",
        limit=None,
        variant=variant,
        stage="all",
        concurrency=1,
        base_url="http://localhost/v1",
        model="model",
        judge_base_url="http://localhost/v1",
        judge_model="judge",
        judge_quantizations="",
    )
    skipped = [{"data_id": "missing", "reason": "source_unavailable", "sources": []}]
    generate = AsyncMock(
        side_effect=lambda client, args, record, index: (
            record | {"index": index, "response": "A", "finish_reason": "stop"}
        )
    )
    judge = AsyncMock(side_effect=lambda judge, args, row: row | {"verdict": "consistent"})
    citation = AsyncMock(
        return_value={
            "citation_recall": 1,
            "citation_precision": 1,
            "citation_f1": 1,
            "n_sentences": 1,
            "n_citations": 1,
            "n_invalid_citations": 0,
        }
    )
    monkeypatch.setattr("tmg_cqa_eval.evaluate.check_sources", AsyncMock(return_value=({}, skipped)))
    monkeypatch.setattr("tmg_cqa_eval.evaluate.generate_one", generate)
    monkeypatch.setattr("tmg_cqa_eval.evaluate.judge_answer", judge)
    monkeypatch.setattr("tmg_cqa_eval.citations.score_item", citation)
    asyncio.run(run(args))
    result = json.loads((args.output / "result.json").read_text())
    assert generate.call_args.args[2]["data_id"] == "good"
    assert judge.call_args.args[2]["data_id"] == "good"
    assert citation.call_args.args[1]["data_id"] == "good"
    assert generate.call_args.args[3] == 1
    assert generate.await_count == judge.await_count == citation.await_count == 1
    assert result["accuracy"] == result["citation_f1"] == 1
    assert result["n_input"] == 2 and result["n_total"] == result["n_skipped_sources"] == 1
    assert read_rows(args.output / "skipped.jsonl") == skipped

    skipped.append({"data_id": "good", "reason": "source_unavailable", "sources": []})
    args.stage = "score"
    judge.reset_mock()
    citation.reset_mock()
    asyncio.run(run(args))
    result = json.loads((args.output / "result.json").read_text())
    judge.assert_not_awaited()
    citation.assert_not_awaited()
    assert result["n_total"] == result["n_citation_attempted"] == 0
    assert result["n_skipped_sources"] == 2
