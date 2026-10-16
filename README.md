# Tokyo-Metropolitan Government Complex QA evaluation

Generate answers and score their accuracy and citations for the
[Tokyo-Metropolitan Government Complex QA dataset](https://huggingface.co/datasets/pfnet/tmg-cqa).
RAG answers from supplied documents; Tool searches the Web with Brave Search.

Both variants use the same citation parser for all models: PLaMo sentence/cite tokens
or `<sentence>`/`<cite>` tags, with comma-separated citation numbers without spaces
(e.g. `<cite>0,2</cite>`). Brace tags and text outside sentence tags are not extracted.

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and an OpenAI-compatible API.
Tool models must support function calling.

```bash
uv sync
export OPENAI_API_KEY=your-model-key
export OPENROUTER_API_KEY=your-judge-key
```

The default judge is `google/gemma-4-31b-it` on OpenRouter (BF16, data collection denied).
For a local judge, set `JUDGE_API_KEY` and pass
`--judge-base-url URL --judge-model MODEL --judge-quantizations ''`.

## Run

Use local `rag.jsonl` and `tool.jsonl` from the dataset (`.jsonl.zst` also works).

RAG preparation downloads the linked documents and extracts their PDF pages or HTML text.
Scanned PDFs require OCR: save each source's requested pages as `<body_sha256>.txt`
and pass `--text-dir DIR` to `tmg-cqa-prepare`. Check the extracted text and tables.
Downloads and prepared prompts stay local under `results/`.

For both variants, `--citation-mode general` (the default) uses plain `<sentence>`/`<cite>` tags,
including with PLaMo 3.0. Use `--citation-mode plamo` for models trained in PLaMo 3.1
citation mode.

The following runs generation and scoring on the first three questions of each variant.
Set the endpoints and model IDs to those returned by your servers' `/v1/models`.
The Tool server needs `--enable-auto-tool-choice --tool-call-parser gemma4`.
Set `BRAVE_API_KEY` for Web search.

```bash
export OPENAI_API_KEY=EMPTY JUDGE_API_KEY=EMPTY
export MODEL_URL=http://localhost:8000/v1 MODEL=gemma-4-31B-it
export JUDGE_URL=http://localhost:8000/v1 JUDGE_MODEL=gemma-4-31B-it

uv run tmg-cqa-prepare --dataset data/rag.jsonl \
  --output results/smoke/rag-prepared.jsonl --limit 3
uv run tmg-cqa-eval --variant rag --dataset results/smoke/rag-prepared.jsonl \
  --model "$MODEL" --base-url "$MODEL_URL" --decode reasoning \
  --judge-model "$JUDGE_MODEL" --judge-base-url "$JUDGE_URL" --judge-quantizations '' \
  --limit 3 --concurrency 1 --max-tokens 8192 --output results/smoke/rag
uv run tmg-cqa-eval --variant tool --dataset data/tool.jsonl \
  --model "$MODEL" --base-url "$MODEL_URL" --decode reasoning \
  --judge-model "$JUDGE_MODEL" --judge-base-url "$JUDGE_URL" --judge-quantizations '' \
  --limit 3 --concurrency 1 --max-tokens 8192 --max-tool-steps 8 --output results/smoke/tool

cat results/smoke/rag/result.json results/smoke/tool/result.json
```

`EMPTY` is for servers without API authentication. `--judge-quantizations ''` disables
OpenRouter-specific provider options. For a full run, remove `--limit 3` and
`--max-tool-steps 8` (the default is 30 rounds) and `--max-tokens 8192`.
Both variants default to 16,384 output tokens, including reasoning; Tool applies this
limit per turn. Scores depend on the model and sources.

`BRAVE_SEARCH_API_KEY` also works. To use OpenSearch, pass `--search opensearch`
and set `OPENSEARCH_URL`, `OPENSEARCH_BEARER_TOKEN` and optionally `OPENSEARCH_INDEX`
(default: `tmg`). Tool citations use page refs, assigned from 1 in first-seen search order
within each question. Repeated searches, page reads and keyword matches reuse the same ref.
Scoring combines the observed evidence for each ref; failed calls add no evidence.
PLaMo citation mode requires a citation-capable chat template.

Select `--decode reasoning` for temperature 0.6, top-p 0.95 and top-k 20, or
`--decode greedy` (the default) for temperature 0, top-p 1 and top-k -1.
Both presets use min-p 0, repetition penalty 1, and presence/frequency penalties 0.
Choose the preset explicitly for your model; it is not inferred from the model name.
Override individual values with `--temperature`, `--top-p`, `--top-k`, `--min-p`,
`--repetition-penalty`, `--presence-penalty` and `--frequency-penalty`.
These options apply to RAG generation and every Tool turn, including the final answer.
The judge always uses temperature 0. Resolved generation settings are saved in `run.json`.
Use the same decode settings when comparing scores; earlier runs used temperature 0.

See `uv run tmg-cqa-eval --help` and `uv run tmg-cqa-prepare --help` for all options.
Generation and scoring can also run separately with `--stage generate` or `--stage score`;
use the same dataset and limit for both stages.

## Results

`result.json` contains accuracy (reference consistency), citation recall (sentence
support), precision (citation relevance) and F1 (averaged per answer).
Generation failures count against accuracy; context-limit failures are excluded from
citation averages. Predictions, judgments and run settings are saved alongside the scores.
Record model/judge settings, input hashes, token limits and search backend when comparing runs.

Source links were collected as of **2026-10-01** and may break later. The evaluator checks
them before generation and scoring, including error pages returned with HTTP 200.
Questions with unavailable sources are skipped and excluded from all score denominators.
RAG preparation also skips questions whose requested PDF pages are unavailable.
`skipped.jsonl` and `link_checks.jsonl` record the checks;
`result.json` reports input, skipped and evaluated counts.

## Development

- `src/tmg_cqa_eval/`: evaluation, document preparation, search and scoring code.
- `tests/`: unit tests.
- `results/`: local downloads and evaluation outputs (not tracked).

```bash
uv run ruff format .
uv run ruff check .
uv run pytest -q
```

Code and dataset: Apache-2.0. Linked source documents retain their own terms.
