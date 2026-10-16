"""Read JSONL datasets and write evaluation records."""

import json

import zstandard


def read_rows(path):
    opener = zstandard.open if path.suffix == ".zst" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
