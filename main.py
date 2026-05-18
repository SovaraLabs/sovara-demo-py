#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from openai import AsyncOpenAI
import sovara

from sovara_demo.env import load_repo_env


load_repo_env()

from agent import DEFAULT_AGENT_MODEL
from agent import DEFAULT_INDEX_ROOT
from agent import DEFAULT_MAX_TURNS
from agent import answer_financebench_sample
from agent import ticker_for_company


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
DEFAULT_SAMPLES_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "financebench",
    "raw",
    "financebench_open_source.jsonl",
)


async def get_completion_async(
    prompt,
    model="gpt-4o-2024-11-20",
    max_retries=3,
) -> str:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    retries = 0
    while retries < max_retries:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception:
            retries += 1
            if retries == max_retries:
                raise
            time.sleep(1)
    raise RuntimeError("max_retries must be positive")


# Adapted from VectifyAI/Mafin2.5-FinanceBench eval.py:
# https://github.com/VectifyAI/Mafin2.5-FinanceBench/blob/main/eval.py
async def check_answer_equivalence(answer, gold_answer, query=None, model="gpt-4o-2024-11-20") -> bool:
    query_prompt = f"- Query: {query}" if query else ""
    prompt = f"""
You are an expert evaluator for finance benchmark answers.

Decide whether the candidate answer and the gold answer are equivalent for the
same finance question. Treat values as equivalent when they express the same
amount after unit conversion, rounding, formatting, or harmless wording changes.
Treat them as different when the candidate gives the wrong value, wrong sign,
wrong period, wrong entity, or omits a required calculation.

Do not mark an answer different merely because it includes extra context,
intermediate values, or source wording. If the candidate contains the required
gold conclusion/value/fact and the extra context does not contradict it, mark it
equivalent.

If the answer uses the source company's fiscal-year label but identifies the
same period/date intended by the question or gold answer, treat the period as
equivalent.

{query_prompt}
- Candidate answer: {answer}
- Gold answer: {gold_answer}

Respond with exactly one word: true or false.
"""
    response = await get_completion_async(prompt, model=model)
    if "true" in response.lower():
        return True
    if "false" in response.lower():
        return False
    return False


def load_sample(samples_path: str, sample_id: int) -> dict:
    if sample_id < 0:
        raise ValueError("--sample-id must be non-negative")

    seen = 0
    with open(samples_path, "r", encoding="utf-8") as handle:
        for _, line in enumerate(handle):
            if not line.strip():
                continue
            if seen == sample_id:
                return json.loads(line)
            seen += 1

    raise IndexError(f"--sample-id must be between 0 and {seen - 1}")


def run_sample(args) -> dict:
    sample = load_sample(args.samples_path, args.sample_id)
    gold_answer = sample.get("answer")
    if gold_answer is None:
        raise ValueError(f"FinanceBench sample {args.sample_id} has no gold answer")

    run_name = args.run_name or f"financebench/sample_{args.sample_id}"
    equivalent = None

    with sovara.run(run_name) as run_id:
        agent_answer = answer_financebench_sample(
            sample,
            index_root=args.index_root,
            model=args.agent_model,
            max_turns=args.max_turns,
            verbose=args.verbose,
        )
        with sovara.disable_tracing():
            equivalent = asyncio.run(
                check_answer_equivalence(
                    agent_answer,
                    gold_answer,
                    query=sample["question"],
                    model=args.eval_model,
                )
            )
        result = {
            "run_id": run_id,
            "sample_id": args.sample_id,
            "financebench_id": sample.get("financebench_id"),
            "company": sample["company"],
            "ticker": ticker_for_company(sample["company"]),
            "doc_name": sample["doc_name"],
            "question": sample["question"],
            "gold_answer": gold_answer,
            "agent_answer": agent_answer,
            "equivalent": equivalent,
            "validator_skipped": False,
            "success": True,
            "error": None,
            "timed_out": False,
            "agent_model": args.agent_model or DEFAULT_AGENT_MODEL,
            "eval_model": args.eval_model,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.queue_for_annotation:
            sovara.queue_for_annotation()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one FinanceBench sample through the agent.")
    parser.add_argument("--run-name", default=None, help="Sovara run name. Defaults to financebench/sample_<id>.")
    parser.add_argument("--sample-id", type=int, required=True, help="Zero-based FinanceBench sample id.")
    parser.add_argument("--samples-path", default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--index-root", default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--eval-model", default="gpt-4o-2024-11-20")
    parser.add_argument(
        "--queue-for-annotation",
        action="store_true",
        help="Ask the annotation agent to queue the run after validation.",
    )
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_sample(args)


if __name__ == "__main__":
    main()
