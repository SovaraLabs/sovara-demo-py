#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import traceback

from sovara_demo.env import load_repo_env

load_repo_env()

from sovara_demo.pageindex import PageIndexClient
from sovara_demo.pageindex.utils import remove_fields

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
PDF_DIR = os.path.join(REPO_ROOT, "pdfs")
SAMPLES_PATH = os.path.join(REPO_ROOT, "data", "financebench", "raw", "financebench_open_source.jsonl")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "pageindex")
INDEX_PATH = os.path.join(OUTPUT_DIR, "financebench_pageindex.jsonl")
ERROR_PATH = os.path.join(OUTPUT_DIR, "financebench_pageindex_errors.jsonl")


def load_done_doc_ids(path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(path):
        return done

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["doc_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def load_required_doc_ids(samples_path: str) -> set[str]:
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Missing samples JSONL: {samples_path}")

    doc_ids = set()
    with open(samples_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            doc_name = sample.get("doc_name")
            if not doc_name:
                continue
            doc_ids.add(os.path.splitext(os.path.basename(doc_name))[0])
    return doc_ids


def pdf_paths(required_doc_ids: set[str]) -> list[str]:
    if not os.path.isdir(PDF_DIR):
        raise FileNotFoundError(f"Missing PDF directory: {PDF_DIR}")

    pdfs_by_doc_id = {
        os.path.splitext(name)[0]: os.path.join(PDF_DIR, name)
        for name in os.listdir(PDF_DIR)
        if name.lower().endswith(".pdf")
    }
    missing = sorted(required_doc_ids - set(pdfs_by_doc_id))
    if missing:
        raise FileNotFoundError(
            "Missing PDFs for referenced FinanceBench docs: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )

    return [pdfs_by_doc_id[doc_id] for doc_id in sorted(required_doc_ids)]


def relative_to_repo(path: str) -> str:
    return os.path.relpath(os.path.abspath(path), REPO_ROOT)


def index_pdf(pdf_path: str) -> dict:
    model = os.getenv("PAGEINDEX_MODEL") or None
    client = PageIndexClient(model=model)
    pageindex_doc_id = client.index(pdf_path, mode="pdf")
    doc = client.documents[pageindex_doc_id]
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    return {
        "doc_id": doc_id,
        "pageindex_doc_id": pageindex_doc_id,
        "type": doc.get("type", "pdf"),
        "source_pdf": relative_to_repo(pdf_path),
        "doc_name": doc.get("doc_name") or os.path.basename(pdf_path),
        "doc_description": doc.get("doc_description", ""),
        "page_count": doc.get("page_count", 0),
        "structure": remove_fields(doc.get("structure", []), fields=["text"]),
        "pages": doc.get("pages", []),
    }


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    required_doc_ids = load_required_doc_ids(SAMPLES_PATH)
    paths = pdf_paths(required_doc_ids)
    done = load_done_doc_ids(INDEX_PATH)

    print(
        f"indexing {len(paths)} PDFs referenced by {os.path.relpath(SAMPLES_PATH, REPO_ROOT)}",
        flush=True,
    )

    for index, pdf_path in enumerate(paths, 1):
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        if doc_id in done:
            print(f"[{index}/{len(paths)}] skip {doc_id}", flush=True)
            continue

        print(f"[{index}/{len(paths)}] index {doc_id}", flush=True)
        try:
            record = index_pdf(pdf_path)
            append_jsonl(INDEX_PATH, record)
            done.add(doc_id)
        except Exception as exc:
            append_jsonl(
                ERROR_PATH,
                {
                    "doc_id": doc_id,
                    "source_pdf": relative_to_repo(pdf_path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"ERROR {doc_id}: {exc}", flush=True)

    print(f"wrote {INDEX_PATH}")
    if os.path.exists(ERROR_PATH):
        print(f"errors {ERROR_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
