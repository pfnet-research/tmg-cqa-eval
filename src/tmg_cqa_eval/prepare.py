"""Fetch RAG evidence locally from a link-only dataset before evaluation."""

import argparse
import hashlib
import io
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .citations import DOCUMENT_END_MARKER, DOCUMENT_START_MARKER
from .data import read_rows, write_rows
from .links import SourceUnavailable, response_error


def extract_text(content, source):
    if source["kind"] == "pdf":
        reader = PdfReader(io.BytesIO(content))
        if any(page < 1 or page > len(reader.pages) for page in source["pages"]):
            raise SourceUnavailable(
                f"Requested PDF pages {source['pages']} are unavailable; document has {len(reader.pages)} pages"
            )
        return "\n".join(reader.pages[page - 1].extract_text() or "" for page in source["pages"]).strip()
    soup = BeautifulSoup(content, "html.parser")
    for element in soup(["script", "style", "noscript", "nav", "footer"]):
        element.decompose()
    main = soup.select_one("main, article, #content, #contents") or soup
    if not main.get_text(strip=True):
        main = soup
    return main.get_text("\n", strip=True)


def prepare_row(row, client, cache, text_dir=None):
    documents = []
    manifest = []
    for source in row["sources"]:
        urls = source["urls"]
        if not urls:
            raise SourceUnavailable(f"Missing source URL for {row['data_id']} source {source['source_id']}")
        url = urls[0]
        override = text_dir / f"{source['body_sha256']}.txt" if text_dir else None
        fetched_hash = None
        if override and override.exists():
            text = override.read_text().strip()
            extraction = "local_text_override"
        else:
            cached = cache / f"{hashlib.sha256(url.encode()).hexdigest()}.bin"
            if not cached.exists():
                try:
                    response = client.get(url)
                except httpx.HTTPError as exc:
                    raise SourceUnavailable(f"Cannot fetch {url}: {type(exc).__name__}") from exc
                if error := response_error(response, source["kind"]):
                    raise SourceUnavailable(f"Cannot fetch {url}: {error}")
                cached.write_bytes(response.content)
            content = cached.read_bytes()
            fetched_hash = hashlib.sha256(content).hexdigest()
            text = extract_text(content, source)
            extraction = "pypdf" if source["kind"] == "pdf" else "beautifulsoup"
        if not text or (source["kind"] == "pdf" and sum(character.isalnum() for character in text) < 20):
            raise ValueError(
                f"No text for {row['data_id']} source {source['source_id']}; "
                f"run OCR and supply --text-dir with {source['body_sha256']}.txt"
            )
        actual_hash = hashlib.sha256(text.encode()).hexdigest()
        manifest.append(
            {
                "source_id": source["source_id"],
                "url": url,
                "pages": source["pages"],
                "fetched_file_sha256": fetched_hash,
                "extraction": extraction,
                "body_sha256": actual_hash,
                "original_body_sha256": source["body_sha256"],
                "matches_original_body": actual_hash == source["body_sha256"],
            }
        )
        header = (
            f"PDF: {url}\nページ: {source['pages']}"
            if source["kind"] == "pdf"
            else f"タイトル: {source['title']}\nURL: {url}"
        )
        documents.append(f"({source['source_id']}) {header}\n本文:\n{text}")
    before, rest = row["prompt"].split(DOCUMENT_START_MARKER, 1)
    after = rest.split(DOCUMENT_END_MARKER, 1)[1]
    return row | {
        "prompt": f"{before}{DOCUMENT_START_MARKER}\n" + "\n\n".join(documents) + f"\n{DOCUMENT_END_MARKER}{after}",
        "source_preparation": manifest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/sources"))
    parser.add_argument("--text-dir", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        prepared = []
        for row in rows:
            try:
                prepared.append(prepare_row(row, client, args.cache_dir, args.text_dir))
            except SourceUnavailable as exc:
                prepared.append(row | {"source_unavailable": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, prepared)
    skipped = sum("source_unavailable" in row for row in prepared)
    print(f"Prepared {len(prepared) - skipped} questions; skipped {skipped} unavailable sources.")


if __name__ == "__main__":
    main()
