import zstandard

from tmg_cqa_eval.data import read_rows, write_rows


def test_jsonl_and_compressed_dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [{"data_id": "rag-0", "question": "東京都の予算は？"}]
    write_rows(path, rows)
    assert read_rows(path) == rows
    compressed = path.with_suffix(".jsonl.zst")
    compressed.write_bytes(zstandard.ZstdCompressor().compress(path.read_bytes()))
    assert read_rows(compressed) == rows
