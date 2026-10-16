"""Sentence citation metrics, matching the RAG and tool benchmark definitions."""

import asyncio
import re

from openai import APIError

from . import prompts

RAG_INDEX_PATTERN = re.compile(r"^(?:\((\d+)\)|\[(\d+)\]|\{(\d+)\}|「(\d+)」)\s*(?=PDF: |タイトル: )", re.MULTILINE)
DOCUMENT_START_MARKER = "### ここからドキュメント ###"
DOCUMENT_END_MARKER = "### ここまでドキュメント ###"

MAX_SNIPPET_CHARS = 8000
MAX_COMBINED_SNIPPET_CHARS = 12000

PLAMO_BEGIN_SENTENCE = "<|plamo:begin_sentence:plamo|>"
PLAMO_END_SENTENCE = "<|plamo:end_sentence:plamo|>"
PLAMO_BEGIN_CITE = "<|plamo:begin_cite:plamo|>"
PLAMO_END_CITE = "<|plamo:end_cite:plamo|>"
GENERAL_CITATION_TAGS = ("<sentence>", "</sentence>", "<cite>", "</cite>")
PLAMO_CITATION_TAGS = (PLAMO_BEGIN_SENTENCE, PLAMO_END_SENTENCE, PLAMO_BEGIN_CITE, PLAMO_END_CITE)
PLAMO_SENTENCE_PATTERN = re.compile(
    rf"{re.escape(PLAMO_BEGIN_SENTENCE)}(.*?){re.escape(PLAMO_END_SENTENCE)}",
    re.DOTALL,
)
PLAMO_CITE_PATTERN = re.compile(rf"{re.escape(PLAMO_BEGIN_CITE)}([0-9]+(?:,[0-9]+)*){re.escape(PLAMO_END_CITE)}")
BRACKET_LABEL_PATTERN = re.compile(r"\[\[\s*([^\]]+?)\s*\]\]")


def extract_documents(user_content):
    if DOCUMENT_START_MARKER not in user_content or DOCUMENT_END_MARKER not in user_content:
        return {}
    _, rest = user_content.split(DOCUMENT_START_MARKER, 1)
    document_block, _ = rest.split(DOCUMENT_END_MARKER, 1)
    index_positions = []
    for match in RAG_INDEX_PATTERN.finditer(document_block):
        index_str = next(g for g in match.groups() if g is not None)
        index_positions.append((match.start(), match.end(), int(index_str)))
    documents = {}
    for i, (_, header_end, index_value) in enumerate(index_positions):
        end = index_positions[i + 1][0] if i + 1 < len(index_positions) else len(document_block)
        doc_text = document_block[header_end:end].strip()
        if doc_text:
            documents[index_value] = doc_text
    return documents


class Judge:
    def __init__(self, client, model, semaphore, variant, extra_body=None):
        self.client, self.model, self.semaphore = client, model, semaphore
        self.extra_body = extra_body or {}
        self.need_prompt = prompts.RAG_NEED_CITATION_PROMPT if variant == "rag" else prompts.NEED_CITATION_PROMPT
        self.support_prompt = prompts.RAG_SUPPORT_PROMPT if variant == "rag" else prompts.SUPPORT_PROMPT
        self.relevant_prompt = prompts.RAG_RELEVANT_PROMPT if variant == "rag" else prompts.RELEVANT_PROMPT

    async def query(self, prompt):
        async with self.semaphore:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0,
                extra_body=self.extra_body,
            )
        return completion.choices[0].message.content or ""

    async def support(self, question, sentence, snippet):
        if not snippet:
            return 0.0
        return await self.score(
            self.support_prompt.format(question=question, sentence=sentence, snippet=snippet), support_score
        )

    async def need_citation(self, question, answer, sentence):
        return await self.score(
            self.need_prompt.format(question=question, answer=answer, sentence=sentence), need_citation_score
        )

    async def relevance(self, question, sentence, snippet):
        if not snippet:
            return 0.0
        return await self.score(
            self.relevant_prompt.format(question=question, sentence=sentence, snippet=snippet), relevance_score
        )

    async def score(self, prompt, parse):
        try:
            return parse(await self.query(prompt)) or 0.0
        except APIError:
            return 0.0


def normalize_citation_tags(response):
    for from_tag, to_tag in zip(GENERAL_CITATION_TAGS, PLAMO_CITATION_TAGS, strict=True):
        response = response.replace(from_tag, to_tag)
    return response


def parse_sentences(response):
    response = normalize_citation_tags(response)
    sentences = []
    for match in PLAMO_SENTENCE_PATTERN.finditer(response):
        raw = match.group(1).strip()
        cited = [int(index) for cite in PLAMO_CITE_PATTERN.finditer(raw) for index in cite[1].split(",")]
        if text := PLAMO_CITE_PATTERN.sub("", raw).strip():
            sentences.append({"sentence": text, "citations": list(dict.fromkeys(cited))})
    return sentences


def _hit_text(hit):
    title = str(hit.get("title") or "").strip()
    snippet = str(hit.get("snippet") or "").strip()
    return f"{title}\n{snippet}".strip()


def observations_from_trace(tool_trace):
    """Everything the episode showed for each ref, in the order it was shown."""
    pieces = {}
    for call in tool_trace:
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        for hit in result.get("results") or []:
            if isinstance(hit, dict) and isinstance(hit.get("ref"), int):
                pieces.setdefault(hit["ref"], []).append(_hit_text(hit))
        ref = result.get("ref")
        if isinstance(ref, int):
            if result.get("content"):
                pieces.setdefault(ref, []).append(str(result["content"]))
            pieces.setdefault(ref, []).extend(str(match) for match in result.get("matches") or [])
    observations = {}
    for ref, texts in pieces.items():
        if text := "\n\n".join(dict.fromkeys(text for text in texts if text)):
            observations[ref] = text
    return observations


def first_bracket_label(text):
    match = BRACKET_LABEL_PATTERN.search(text)
    return match.group(1).strip().lower() if match else None


def need_citation_score(text):
    label = first_bracket_label(text)
    if label is None:
        return None
    if "yes" in label:
        return 0.0
    if "no" in label:
        return 1.0
    return None


def support_score(text):
    label = first_bracket_label(text)
    if label == "fully supported":
        return 1.0
    if label == "partially supported":
        return 0.5
    if label == "no support":
        return 0.0
    return None


def relevance_score(text):
    label = first_bracket_label(text)
    if label is None:
        return None
    if "unrelevant" in label or "irrelevant" in label:
        return 0.0
    if "relevant" in label:
        return 1.0
    return None


def plain_answer(response):
    text = normalize_citation_tags(response)
    text = PLAMO_CITE_PATTERN.sub("", text)
    return text.replace(PLAMO_BEGIN_SENTENCE, "").replace(PLAMO_END_SENTENCE, "").strip()


async def score_item(judge, row, documents=None):
    question = row["question"]
    response = row["response"]
    observations = documents if documents is not None else observations_from_trace(row.get("tool_trace") or [])
    result = {
        "index": row["index"],
        "citation_recall": 0.0,
        "citation_precision": 0.0,
        "citation_f1": 0.0,
        "n_sentences": 0,
        "n_citations": 0,
        "n_invalid_citations": 0,
        "n_tool_calls": len(row.get("tool_trace") or []),
        "sentences": [],
    }
    if not response.strip() or row.get("finish_reason") in {"error", "input_too_long", "context_exhausted"}:
        result["skipped"] = row.get("finish_reason", "empty")
        return result
    sentences = parse_sentences(response)
    result["n_sentences"] = len(sentences)
    if not sentences:
        return result
    answer = plain_answer(response)

    async def score_sentence(item):
        sentence, cited = item["sentence"], item["citations"]
        evidence = {c: text[:MAX_SNIPPET_CHARS] if documents is not None else text for c, text in observations.items()}
        separator = "\n" if documents is not None else "\n\n"
        snippet = separator.join(evidence[c] for c in cited if c in evidence)
        if documents is not None:
            snippet = snippet[:MAX_COMBINED_SNIPPET_CHARS]
        recall = (
            judge.support(question, sentence, snippet) if cited else judge.need_citation(question, answer, sentence)
        )
        recall_score, *relevance = await asyncio.gather(
            recall,
            *(judge.relevance(question, sentence, evidence.get(c, "")) for c in cited),
        )
        scored = item | {"recall_score": recall_score}
        if cited:
            scored["relevance_scores"] = relevance
        return scored

    result["sentences"] = await asyncio.gather(*(score_sentence(item) for item in sentences))
    cited = [c for item in sentences for c in item["citations"]]
    result["n_citations"] = len(cited)
    result["n_invalid_citations"] = sum(c not in observations for c in cited)
    relevance = [score for item in result["sentences"] for score in item.get("relevance_scores", [])]
    recall = sum(item["recall_score"] for item in result["sentences"]) / len(sentences)
    precision = sum(relevance) / len(relevance) if relevance else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    result["citation_recall"] = recall
    result["citation_precision"] = precision
    result["citation_f1"] = f1
    return result


def summarize_citations(items, variant):
    excluded = "input_too_long" if variant == "rag" else "context_exhausted"
    attempted = [item for item in items if item.get("skipped") != excluded]
    scored = [item for item in attempted if item["n_sentences"] > 0]

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "citation_recall": mean([item["citation_recall"] for item in attempted]),
        "citation_precision": mean([item["citation_precision"] for item in attempted]),
        "citation_f1": mean([item["citation_f1"] for item in attempted]),
        "citation_recall_given_scored": mean([item["citation_recall"] for item in scored]),
        "citation_precision_given_scored": mean([item["citation_precision"] for item in scored]),
        "citation_f1_given_scored": mean([item["citation_f1"] for item in scored]),
        "n_citation_attempted": len(attempted),
        "n_citation_scored": len(scored),
        "mean_sentences_per_answer": mean([float(item["n_sentences"]) for item in scored]),
        "mean_citations_per_answer": mean([float(item["n_citations"]) for item in scored]),
        "invalid_citation_rate": (
            sum(item["n_invalid_citations"] for item in scored) / total
            if (total := sum(item["n_citations"] for item in scored))
            else 0.0
        ),
    }
