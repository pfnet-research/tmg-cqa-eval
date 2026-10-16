"""Generate and score Tokyo-Metropolitan Government Complex QA."""

import argparse
import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from openai import APIError, AsyncOpenAI, BadRequestError

from . import citations, prompts
from .data import read_rows, write_rows
from .links import check_sources
from .search import BraveSearch, OpenSearch, execute

TOOLS = json.loads(Path(__file__).with_name("tools.json").read_text())
DECODE_DEFAULTS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


def strip_citations(text, variant):
    if variant == "tool":
        for pattern in [
            r"<\|plamo:begin_cite:plamo\|>.*?<\|plamo:end_cite:plamo\|>",
            r"\{cite\}[^{]*\{/cite\}",
            r"\{/?sentence\}",
        ]:
            text = re.sub(pattern, "", text, flags=re.DOTALL)
    for pattern in [r"<\|plamo:[^|]+:plamo\|>", r"<cite>[^<]*</cite>", r"</?sentence>", r"\[(\d+)\]"]:
        text = re.sub(pattern, "", text)
    return text.strip()


def truncate(text, limit):
    return text if len(text) <= limit else text[:limit] + "…"


async def generate_one(client, args, record, index):
    row = {
        "index": index,
        "data_id": record["data_id"],
        "question": record["question"],
        "reference_answer": record["reference_answer"],
        "model": args.model,
    }
    plamo_citation = args.citation_mode == "plamo"
    extra = {"chat_template_kwargs": {"enable_citation": True}} if plamo_citation else {}
    prompt = record["prompt"]
    if not plamo_citation:
        for plamo_tag, general_tag in zip(citations.PLAMO_CITATION_TAGS, citations.GENERAL_CITATION_TAGS, strict=True):
            prompt = prompt.replace(plamo_tag, general_tag)
    messages = [{"role": "user", "content": prompt}]
    trace = []
    ref_by_page_id = {}
    backend = (BraveSearch if args.search == "brave" else OpenSearch)() if args.variant == "tool" else None
    extra.update(top_k=args.top_k, min_p=args.min_p, repetition_penalty=args.repetition_penalty)
    options = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "extra_body": extra,
    }
    try:
        if backend is None:
            max_tokens = args.max_tokens
            for _ in range(5):
                try:
                    response = await client.chat.completions.create(**options, messages=messages, max_tokens=max_tokens)
                    break
                except BadRequestError as exc:
                    limit = re.search(r"maximum context length is (\d+) tokens", str(exc))
                    input_tokens = re.search(r"(\d+) input tokens", str(exc))
                    if not limit or not input_tokens:
                        raise
                    max_tokens = int(limit[1]) - int(input_tokens[1]) - 64
                    if max_tokens < 1024:
                        return row | {"response": "[INPUT_TOO_LONG]", "finish_reason": "input_too_long"}
            else:
                return row | {"response": "[GENERATION_FAILED]", "finish_reason": "error"}
        else:
            for step in range(args.max_tool_steps + 1):
                exhausted = step == args.max_tool_steps
                if exhausted:
                    messages.append({"role": "user", "content": "Give the single best final answer now."})
                response = await client.chat.completions.create(
                    **options,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="none" if exhausted else "auto",
                    max_completion_tokens=min(args.max_tokens, 16384),
                )
                message = response.choices[0].message
                if not message.tool_calls:
                    break
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or None,
                        "reasoning": getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None),
                        "tool_calls": [call.model_dump() for call in message.tool_calls],
                    }
                )
                for call in message.tool_calls:
                    name = call.function.name
                    arguments = call.function.arguments
                    try:
                        arguments = json.loads(arguments)
                        result = await asyncio.to_thread(execute, backend, name, arguments, ref_by_page_id)
                        trace.append({"name": name, "arguments": arguments, "result": result})
                    except Exception as exc:  # noqa: BLE001 - failed tools remain in the ordered trace
                        result = {"error": f"{type(exc).__name__}: {exc}"}
                        trace.append({"name": name, "arguments": arguments, **result})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": f"Tool call failed: {result['error']}"
                            if "error" in result
                            else json.dumps(result, ensure_ascii=False),
                        }
                    )
            row.update(tool_trace=trace, tool_call_count=sum("result" in t for t in trace), budget_exhausted=exhausted)
        choice = response.choices[0]
        return row | {
            "response": (choice.message.content or "").strip(),
            "finish_reason": choice.finish_reason,
            "reasoning_content": getattr(choice.message, "reasoning_content", None)
            or getattr(choice.message, "reasoning", None)
            or "",
        }
    except BadRequestError as exc:
        if args.variant != "tool" or "maximum context length" not in str(exc):
            raise
        return row | {
            "response": "",
            "finish_reason": "context_exhausted",
            "tool_trace": [],
            "tool_call_count": None,
            "budget_exhausted": False,
        }
    finally:
        if backend is not None:
            backend.close()


async def judge_answer(judge, args, row):
    finish = row.get("finish_reason")
    if finish in {"input_too_long", "context_exhausted"}:
        return row | {"verdict": finish}
    if finish == "error" or not row["response"].strip():
        return row | {"verdict": "generation_failed"}
    try:
        output = await judge.query(
            prompts.JUDGE_PROMPT.format(
                question=truncate(row["question"].strip(), 2000),
                reference=truncate(row["reference_answer"].strip(), 6000),
                candidate=truncate(strip_citations(row["response"], args.variant), 8000),
            )
        )
        result = json.loads(output[output.find("{") : output.rfind("}") + 1])
        if result["verdict"] not in {"consistent", "contradictory", "insufficient"}:
            raise ValueError(f"Invalid judge verdict: {result['verdict']}")
        return row | result
    except (APIError, ValueError, KeyError, TypeError) as exc:
        return row | {"verdict": "judge_error", "conflicts": [], "reason": f"{type(exc).__name__}: {exc}"}


def summarize(rows, variant, model):
    n = len(rows)
    counts = Counter(row["verdict"] for row in rows)
    judged = sum(counts[k] for k in ["consistent", "contradictory", "insufficient"])
    summary = {
        "n_total": n,
        "n_judged": judged,
        "judge_model": model,
        "accuracy": counts["consistent"] / n if n else 0,
        "accuracy_given_judged": counts["consistent"] / judged if judged else 0,
        "contradictory_rate": counts["contradictory"] / n if n else 0,
        "insufficient_rate": counts["insufficient"] / n if n else 0,
    }
    categories = [
        "consistent",
        "contradictory",
        "insufficient",
        "generation_failed",
        "judge_error",
        "input_too_long" if variant == "rag" else "context_exhausted",
    ]
    summary.update({f"n_{key}": counts[key] for key in categories})
    if variant == "tool":
        attempted = [
            r for r in rows if r.get("finish_reason") not in {"context_exhausted", "error"} and r["response"].strip()
        ]
        counted = [r for r in rows if r.get("tool_call_count") is not None]
        summary.update(
            no_tool_call_rate=sum(not r.get("tool_trace") for r in attempted) / len(attempted) if attempted else 0,
            mean_tool_calls=sum(r["tool_call_count"] for r in counted) / len(counted) if counted else 0,
            budget_exhausted_rate=sum(r.get("budget_exhausted", False) for r in attempted) / len(attempted)
            if attempted
            else 0,
        )
    return summary


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    records = read_rows(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]
    checks, skipped = await check_sources(records)
    skipped_ids = {row["data_id"] for row in skipped}
    write_rows(args.output / "link_checks.jsonl", checks.values())
    write_rows(args.output / "skipped.jsonl", skipped)
    eligible = [(index, row) for index, row in enumerate(records) if row["data_id"] not in skipped_ids]
    if args.variant == "rag" and any("sources" in row and "source_preparation" not in row for _, row in eligible):
        raise ValueError("Run tmg-cqa-prepare on the link-only RAG dataset before evaluation.")
    predictions = args.output / "predictions.jsonl"
    semaphore = asyncio.Semaphore(args.concurrency)
    if args.stage in {"generate", "all"}:
        async with AsyncOpenAI(
            base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=1800
        ) as client:

            async def one(index, record):
                async with semaphore:
                    try:
                        return await generate_one(client, args, record, index)
                    except Exception as exc:  # noqa: BLE001 - retain failed generations in the denominator
                        return {
                            "index": index,
                            "data_id": record["data_id"],
                            "question": record["question"],
                            "reference_answer": record["reference_answer"],
                            "model": args.model,
                            "response": "" if args.variant == "tool" else "[GENERATION_FAILED]",
                            "finish_reason": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "tool_trace": [],
                            "tool_call_count": None,
                        }

            rows = await asyncio.gather(*(one(index, row) for index, row in eligible))
        write_rows(predictions, rows)
    if args.stage in {"score", "all"}:
        rows = [row for row in read_rows(predictions) if row["data_id"] not in skipped_ids]
        extra = (
            {"provider": {"quantizations": args.judge_quantizations.split(","), "data_collection": "deny"}}
            if args.judge_quantizations
            else {}
        )
        async with AsyncOpenAI(
            base_url=args.judge_base_url,
            api_key=os.environ.get("JUDGE_API_KEY", os.environ.get("OPENROUTER_API_KEY", "EMPTY")),
            timeout=600,
        ) as client:
            judge = citations.Judge(client, args.judge_model, semaphore, args.variant, extra)
            judgments = await asyncio.gather(*(judge_answer(judge, args, row) for row in rows))
            items = await asyncio.gather(
                *(
                    citations.score_item(
                        judge,
                        row,
                        citations.extract_documents(records[row["index"]]["prompt"]) if args.variant == "rag" else None,
                    )
                    for row in rows
                )
            )
        write_rows(args.output / "judgments.jsonl", judgments)
        write_rows(args.output / "citation_judgments.jsonl", items)
        result = (
            summarize(judgments, args.variant, args.judge_model)
            | citations.summarize_citations(items, args.variant)
            | {"n_input": len(records), "n_skipped_sources": len(skipped)}
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    (args.output / "run.json").write_text(
        json.dumps(
            vars(args)
            | {
                "dataset": str(args.dataset),
                "output": str(args.output),
                "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "links_checked_at": datetime.now(UTC).isoformat(),
                "skipped_data_ids": sorted(skipped_ids),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["rag", "tool"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["generate", "score", "all"], default="all")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--judge-model", default="google/gemma-4-31b-it")
    parser.add_argument("--judge-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-quantizations", default="bf16")
    parser.add_argument("--search", choices=["brave", "opensearch"], default="brave")
    parser.add_argument(
        "--citation-mode",
        choices=["general", "plamo"],
        default="general",
        help="Citation format for both variants; use plamo for models trained in PLaMo 3.1 citation mode",
    )
    parser.add_argument("--decode", choices=["greedy", "reasoning"], default="greedy", help="Answer-generation preset")
    for name, default in DECODE_DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(default), help=f"Override {name} for generation")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Total output tokens, including reasoning")
    parser.add_argument("--max-tool-steps", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    defaults = DECODE_DEFAULTS | (
        {"temperature": 0.6, "top_p": 0.95, "top_k": 20} if args.decode == "reasoning" else {}
    )
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
