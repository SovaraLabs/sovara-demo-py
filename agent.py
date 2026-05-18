#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import asyncio
import concurrent.futures
import json
import os
from decimal import Decimal, getcontext

from sovara import log_input, log_output
from sovara_demo.env import load_repo_env
from sovara_demo.pageindex.retrieve import get_document as pageindex_get_document
from sovara_demo.pageindex.retrieve import get_document_outline as pageindex_get_document_outline
from sovara_demo.pageindex.retrieve import get_page_content as pageindex_get_page_content
from sovara_demo.pageindex.retrieve import get_section_content as pageindex_get_section_content
from sovara_demo.pageindex.retrieve import search_document_structure as pageindex_search_document_structure
from sovara_demo.pageindex.retrieve import search_page_content as pageindex_search_page_content


load_repo_env()

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
DEFAULT_INDEX_ROOT = os.path.join(REPO_ROOT, "data", "pageindex")
DEFAULT_INDEX_FILENAME = "financebench_pageindex.jsonl"
DEFAULT_AGENT_MODEL = os.getenv("FINANCEBENCH_AGENT_MODEL", "claude-haiku-4-5")
DEFAULT_MAX_TURNS_ENV = os.getenv("FINANCEBENCH_AGENT_MAX_TURNS")
DEFAULT_MAX_TURNS = int(DEFAULT_MAX_TURNS_ENV) if DEFAULT_MAX_TURNS_ENV else None


# FinanceBench companies (updated with actual company names from dataset)
FINANCEBENCH_COMPANIES = {
    "3M": "MMM",
    "AES Corporation": "AES",
    "AMD": "AMD",
    "Activision Blizzard": "ATVI",
    "Adobe": "ADBE",
    "Amazon": "AMZN",
    "Amcor": "AMCR",
    "American Express": "AXP",
    "American Water Works": "AWK",
    "Best Buy": "BBY",
    "Block": "SQ",
    "Boeing": "BA",
    "CVS Health": "CVS",
    "Coca-Cola": "KO",
    "Corning": "GLW",
    "Costco": "COST",
    "Foot Locker": "FL",
    "General Mills": "GIS",
    "JPMorgan": "JPM",
    "Johnson & Johnson": "JNJ",
    "Kraft Heinz": "KHC",
    "Lockheed Martin": "LMT",
    "MGM Resorts": "MGM",
    "Microsoft": "MSFT",
    "Netflix": "NFLX",
    "Nike": "NKE",
    "Paypal": "PYPL",
    "PepsiCo": "PEP",
    "Pfizer": "PFE",
    "Ulta Beauty": "ULTA",
    "Verizon": "VZ",
    "Walmart": "WMT",
}


# Datamule resolves tickers through a listed-company snapshot. Delisted or
# renamed FinanceBench companies need CIK fallbacks so benchmark ingestion stays
# keyed by the original FinanceBench ticker.
TICKER_CIK_FALLBACKS = {
    "ATVI": "0000718877",
    "SQ": "0001512673",
}


AGENT_SYSTEM_PROMPT = """
You answer FinanceBench questions using a PageIndex document index.

Tool use:
- Work only with the single document named in the user prompt.
- Call get_document first to confirm the document metadata.
- Use search_document_structure to find relevant sections by title/summary.
- Use get_section_content for a matching node_id or title_query when the section is narrow.
- Use search_page_content to find exact table rows or line items in page text.
- Use get_page_content only as a low-level fallback with focused page ranges. Do not fetch the full document.
- Use calculate for arithmetic involving ratios, percentages, percentage-point
  changes, basis-point changes, growth rates, per-share values, averages, or unit
  conversions. Pass formulas using source values, not rounded displayed values.
  When a formula adds or subtracts multiple source values, pass the expanded
  expression to calculate; do not precompute subtotals mentally.
- Bash, Read, Grep, Glob, and LS are available for local inspection or fallback
  search when the MCP document tools are not enough. Do not modify files.

Answering:
- Base the answer only on retrieved document text.
- Preserve requested units and signs.
- Show short arithmetic when the question requires calculation, and base numeric
  results on calculate outputs rather than mental arithmetic.
- Any calculated number in the final answer must match the calculate expression
  and its output. If the displayed inputs are components, the calculate expression
  must use those same components.
- If a value is in millions and the question asks for billions, convert it.
- For yes/no questions, answer yes or no first, then give concise arithmetic/context.
- If retrieved runtime priors give a more specific source-selection or answer-shaping
  rule than the question metadata, follow the prior.
- Use question metadata, when provided, to choose answer depth:
  - Information extraction: answer with the directly requested fact(s) and the
    minimal supporting context needed to identify them. Do not expand into a
    broader analysis or adjacent factors unless the question asks for that.
  - Calculation: show the formula, inputs, arithmetic, and final value.
  - Comparison: compare only the requested entities, periods, or metrics, then
    state the conclusion.
  - Explanation: explain the causal driver(s) or rationale, with scope anchored
    to the question. If the source identifies a primary driver and the question
    does not ask for a full bridge, answer with that primary driver or drivers
    and do not add supporting breakdowns, secondary drivers, or offsets.
- For direct-driver questions, the final answer should usually be one sentence.
  Do not include calculated change magnitudes, percentages, support amounts, or
  segment losses unless the question asks for them.
- Return the final answer directly and keep it concise.
"""


def ticker_for_company(company: str) -> str:
    return FINANCEBENCH_COMPANIES[company]


def doc_id_from_name(doc_name: str) -> str:
    return os.path.splitext(os.path.basename(doc_name))[0]


def _unique_paths(paths: list[str]) -> list[str]:
    seen = set()
    unique = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def index_paths(index_root: str, company: str | None = None, ticker: str | None = None) -> list[str]:
    if os.path.isfile(index_root):
        return [index_root]

    resolved_ticker = ticker
    if resolved_ticker is None and company:
        resolved_ticker = ticker_for_company(company)

    candidates = []
    if resolved_ticker:
        candidates.extend(
            [
                os.path.join(index_root, resolved_ticker, DEFAULT_INDEX_FILENAME),
                os.path.join(index_root, resolved_ticker, "index.jsonl"),
            ]
        )

    candidates.append(os.path.join(index_root, DEFAULT_INDEX_FILENAME))
    existing = [path for path in candidates if os.path.exists(path)]
    if existing:
        return _unique_paths(existing)
    return _unique_paths(candidates[:1])


def sample_index_paths(sample: dict, index_root: str = DEFAULT_INDEX_ROOT) -> list[str]:
    doc_id = doc_id_from_name(sample["doc_name"])
    company = sample.get("company")
    ticker = ticker_for_company(company) if company else None

    if os.path.isfile(index_root):
        return [index_root]

    candidates = []
    if ticker:
        candidates.extend(
            [
                os.path.join(index_root, ticker, f"{doc_id}.json"),
                os.path.join(index_root, ticker, f"{doc_id}.jsonl"),
                os.path.join(index_root, ticker, DEFAULT_INDEX_FILENAME),
                os.path.join(index_root, ticker, "index.jsonl"),
            ]
        )
    candidates.extend(
        [
            os.path.join(index_root, f"{doc_id}.json"),
            os.path.join(index_root, f"{doc_id}.jsonl"),
            os.path.join(index_root, DEFAULT_INDEX_FILENAME),
        ]
    )

    existing = [path for path in candidates if os.path.exists(path)]
    if existing:
        return _unique_paths(existing)
    return _unique_paths(candidates)


def _record_doc_id(record: dict) -> str:
    if record.get("doc_id"):
        return record["doc_id"]
    if record.get("doc_name"):
        return doc_id_from_name(record["doc_name"])
    if record.get("source_pdf"):
        return doc_id_from_name(record["source_pdf"])
    raise KeyError("Indexed record has no doc_id, doc_name, or source_pdf")


def _normalize_record(record: dict) -> dict:
    normalized = dict(record)
    source_pdf = normalized.get("source_pdf")
    if source_pdf and "path" not in normalized:
        if os.path.isabs(source_pdf):
            normalized["path"] = source_pdf
        else:
            normalized["path"] = os.path.join(REPO_ROOT, source_pdf)
    return normalized


def load_indexed_documents(index_root: str = DEFAULT_INDEX_ROOT, company: str | None = None) -> dict:
    documents = {}
    for path in index_paths(index_root, company=company):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = _normalize_record(json.loads(line))
                documents[_record_doc_id(record)] = record
    return documents


def _records_from_path(path: str) -> list[dict]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [_normalize_record(record) for record in payload]
        return [_normalize_record(payload)]

    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(_normalize_record(json.loads(line)))
    return records


def load_indexed_document_for_sample(sample: dict, index_root: str = DEFAULT_INDEX_ROOT) -> tuple[str, dict]:
    doc_id = doc_id_from_name(sample["doc_name"])
    checked_paths = []
    for path in sample_index_paths(sample, index_root=index_root):
        checked_paths.append(path)
        if not os.path.exists(path):
            continue
        for record in _records_from_path(path):
            if _record_doc_id(record) == doc_id:
                return doc_id, record

    raise KeyError(
        f"Could not find preprocessed document {doc_id!r}. "
        f"Checked: {', '.join(checked_paths)}"
    )


def _tool_response(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


getcontext().prec = 50


def _format_decimal(value: Decimal) -> str:
    if value.is_nan() or value.is_infinite():
        raise ValueError("Calculation result is not finite")
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _decimal_from_node(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _decimal_from_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_decimal_from_node(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _decimal_from_node(node.operand)
    if isinstance(node, ast.BinOp):
        left = _decimal_from_node(node.left)
        right = _decimal_from_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(f"Unsupported arithmetic expression: {ast.dump(node, include_attributes=False)}")


def calculate_expression(expression: str) -> str:
    parsed = ast.parse(expression, mode="eval")
    result = _decimal_from_node(parsed)
    return f"{expression} = {_format_decimal(result)}"


def _build_mcp_server(documents: dict, doc_id: str):
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool(
        "get_document",
        "Get metadata for the FinanceBench document currently being answered.",
        {"type": "object", "properties": {}},
    )
    async def get_document_tool(args: dict) -> dict:
        return _tool_response(pageindex_get_document(documents, doc_id))

    @tool(
        "get_document_outline",
        "Get a compact PageIndex outline with section titles and page spans.",
        {
            "type": "object",
            "properties": {
                "depth": {
                    "type": "integer",
                    "description": "Maximum nested outline depth to return. Defaults to 4.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of outline nodes to return. Defaults to 100.",
                },
                "include_summary": {
                    "type": "boolean",
                    "description": "Include short summaries for outline nodes. Defaults to false.",
                },
            },
        },
    )
    async def get_document_outline_tool(args: dict) -> dict:
        return _tool_response(
            pageindex_get_document_outline(
                documents,
                doc_id,
                depth=args.get("depth", 4),
                limit=args.get("limit", 100),
                include_summary=args.get("include_summary", False),
            )
        )

    @tool(
        "search_document_structure",
        "Search PageIndex section titles and summaries. Returns compact matching nodes with node_id and page spans.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or phrases, for example 'balance sheet' or 'cash flow|capital expenditures'.",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to search. Defaults to ['title', 'summary'].",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum matches to return. Defaults to 10.",
                },
                "context_chars": {
                    "type": "integer",
                    "description": "Approximate snippet size. Defaults to 240.",
                },
            },
            "required": ["query"],
        },
    )
    async def search_document_structure_tool(args: dict) -> dict:
        return _tool_response(
            pageindex_search_document_structure(
                documents,
                doc_id,
                query=args["query"],
                fields=args.get("fields"),
                limit=args.get("limit", 10),
                context_chars=args.get("context_chars", 240),
            )
        )

    @tool(
        "get_section_content",
        "Get page text for a PageIndex section by node_id or title query. Large sections are capped by max_pages.",
        {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "PageIndex node_id from search_document_structure or get_document_outline.",
                },
                "title_query": {
                    "type": "string",
                    "description": "Title keywords to find a section when node_id is unknown.",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum section pages to return. Defaults to 5.",
                },
            },
        },
    )
    async def get_section_content_tool(args: dict) -> dict:
        return _tool_response(
            pageindex_get_section_content(
                documents,
                doc_id,
                node_id=args.get("node_id"),
                title_query=args.get("title_query"),
                max_pages=args.get("max_pages", 5),
            )
        )

    @tool(
        "search_page_content",
        "Search actual page text and return compact snippets with physical page numbers.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to find in page content, for example 'Total current assets'.",
                },
                "pages": {
                    "type": "string",
                    "description": "Optional physical page range such as '5-8' or '47,49,51'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum matching pages to return. Defaults to 10.",
                },
                "context_chars": {
                    "type": "integer",
                    "description": "Approximate snippet size. Defaults to 240.",
                },
            },
            "required": ["query"],
        },
    )
    async def search_page_content_tool(args: dict) -> dict:
        return _tool_response(
            pageindex_search_page_content(
                documents,
                doc_id,
                query=args["query"],
                pages=args.get("pages"),
                limit=args.get("limit", 10),
                context_chars=args.get("context_chars", 240),
            )
        )

    @tool(
        "get_page_content",
        "Get text for specific physical pages. Use narrow ranges such as '57', '57-59', or '57,60'.",
        {
            "type": "object",
            "properties": {
                "pages": {
                    "type": "string",
                    "description": "Physical page numbers, for example '57', '57-59', or '57,60'.",
                },
            },
            "required": ["pages"],
        },
    )
    async def get_page_content_tool(args: dict) -> dict:
        return _tool_response(pageindex_get_page_content(documents, doc_id, args["pages"]))

    @tool(
        "calculate",
        "Evaluate a deterministic arithmetic expression. Use source values directly; for sums, pass the expanded expression instead of a precomputed subtotal.",
        {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression using numbers, parentheses, +, -, *, and /. Use original source values, for example: (120 + 40 + 80 + 5) / 190.",
                },
            },
            "required": ["expression"],
        },
    )
    async def calculate_tool(args: dict) -> dict:
        try:
            return _tool_response(calculate_expression(args["expression"]))
        except Exception as exc:
            return _tool_response(f"Calculation error: {exc}")

    return create_sdk_mcp_server(
        name="financebench",
        version="1.0.0",
        tools=[
            get_document_tool,
            get_document_outline_tool,
            search_document_structure_tool,
            get_section_content_tool,
            search_page_content_tool,
            get_page_content_tool,
            calculate_tool,
        ],
    )


def _print_verbose_message(message) -> None:
    from claude_agent_sdk.types import AssistantMessage, ResultMessage, UserMessage

    if isinstance(message, AssistantMessage):
        for block in message.content:
            if hasattr(block, "text") and block.text.strip():
                print(block.text, flush=True)
            elif hasattr(block, "name"):
                print(f"[tool call] {block.name} {getattr(block, 'input', {})}", flush=True)
    elif isinstance(message, UserMessage) and isinstance(message.content, list):
        for block in message.content:
            content = getattr(block, "content", None)
            if content:
                preview = str(content)
                if len(preview) > 600:
                    preview = preview[:600] + "..."
                print(f"[tool result] {preview}", flush=True)
    elif isinstance(message, ResultMessage):
        cost = getattr(message, "total_cost_usd", None)
        turns = getattr(message, "num_turns", None)
        if cost is not None and turns is not None:
            print(f"[result] turns={turns} cost=${cost:.4f}", flush=True)


async def answer_question_async(
    question: str,
    doc_id: str,
    documents: dict,
    model: str | None = None,
    max_turns: int | None = DEFAULT_MAX_TURNS,
    verbose: bool = False,
    question_type: str | None = None,
    question_reasoning: str | None = None,
) -> str:
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
    from claude_agent_sdk.types import AssistantMessage, ResultMessage

    mcp_server = _build_mcp_server(documents, doc_id)
    options = ClaudeAgentOptions(
        system_prompt={"type": "preset", "preset": "claude_code", "append": AGENT_SYSTEM_PROMPT},
        allowed_tools=[
            "Bash",
            "Read",
            "Grep",
            "Glob",
            "LS",
            "mcp__financebench__get_document",
            "mcp__financebench__get_document_outline",
            "mcp__financebench__search_document_structure",
            "mcp__financebench__get_section_content",
            "mcp__financebench__search_page_content",
            "mcp__financebench__get_page_content",
            "mcp__financebench__calculate",
        ],
        mcp_servers={"financebench": mcp_server},
        permission_mode="default",
        model=model or DEFAULT_AGENT_MODEL,
        max_turns=max_turns,
        max_buffer_size=100_000_000,
        cwd=REPO_ROOT,
    )

    metadata_lines = []
    if question_type:
        metadata_lines.append(f"Question type: {question_type}")
    if question_reasoning:
        metadata_lines.append(f"Question reasoning: {question_reasoning}")

    prompt_parts = [
        f"Document id: {doc_id}",
        *metadata_lines,
        f"Question: {question}",
        "",
        "Answer the question from this document.",
        "Keep the final answer concise.",
        (
            "If the document states a primary driver or direct driver clause for a "
            "requested change, answer those driver(s) directly; do not add supporting "
            "breakdowns, secondary drivers, offsets, or a full bridge unless the "
            "question explicitly asks for one."
        ),
        (
            "If the direct driver clause answers the question, the final answer should "
            "be the driver sentence only; do not add calculated change magnitudes, "
            "supporting amounts, segment details, or surrounding context unless "
            "requested."
        ),
        (
            "For information-extraction questions about what drove a change, extract "
            "only the directly requested driver factors. Do not include opposite-"
            "direction offsets, bridge detail, or broader analysis unless requested."
        ),
        (
            "Before returning a direct-driver answer, remove unnecessary text that "
            "explains how much the metric changed, why the change was not larger, or "
            "how related segments/rows performed."
        ),
        (
            "If the question conditionally asks whether a metric is not useful and "
            "the metric is structurally valid, do not add a metric-usefulness caveat; "
            "answer the main question only."
        ),
        (
            "If the metric is structurally not useful, state that directly and do not "
            "calculate proxy or alternative metrics unless requested."
        ),
    ]
    prompt = "\n".join(prompt_parts)
    log_input(question)
    assistant_text = []
    final_answer = ""

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if verbose:
                _print_verbose_message(message)
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text") and block.text.strip():
                        assistant_text.append(block.text)
            elif isinstance(message, ResultMessage):
                if getattr(message, "is_error", False):
                    raise RuntimeError(message.result or "Claude agent returned an error")
                final_answer = message.result or ""

    answer = final_answer.strip() if final_answer.strip() else "\n".join(assistant_text).strip()
    log_output(answer)
    return answer


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def answer_question(
    question: str,
    doc_id: str,
    documents: dict,
    model: str | None = None,
    max_turns: int | None = DEFAULT_MAX_TURNS,
    verbose: bool = False,
    question_type: str | None = None,
    question_reasoning: str | None = None,
) -> str:
    return _run_async(
        answer_question_async(
            question,
            doc_id,
            documents,
            model,
            max_turns,
            verbose,
            question_type,
            question_reasoning,
        )
    )


def answer_financebench_sample(
    sample: dict,
    index_root: str = DEFAULT_INDEX_ROOT,
    model: str | None = None,
    max_turns: int | None = DEFAULT_MAX_TURNS,
    verbose: bool = False,
) -> str:
    doc_id, document = load_indexed_document_for_sample(sample, index_root=index_root)
    documents = {doc_id: document}
    return answer_question(
        question=sample["question"],
        doc_id=doc_id,
        documents=documents,
        model=model,
        max_turns=max_turns,
        verbose=verbose,
        question_type=sample.get("question_type"),
        question_reasoning=sample.get("question_reasoning"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the FinanceBench Claude agent on a single question.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--doc-name", required=True)
    parser.add_argument("--company", required=True)
    parser.add_argument("--index-root", default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sample = {
        "company": args.company,
        "doc_name": args.doc_name,
        "question": args.question,
    }
    answer = answer_financebench_sample(
        sample,
        index_root=args.index_root,
        model=args.model,
        max_turns=args.max_turns,
        verbose=args.verbose,
    )
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
