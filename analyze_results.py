#!/usr/bin/env python3
"""Analyze FinanceBench results from a run_benchmark.sh output directory."""

import argparse
import json
import os
import sys


def load_results(results_dir: str) -> list[dict]:
    """Load all JSON result files from a directory."""
    if not os.path.isdir(results_dir):
        print(f"Error: directory '{results_dir}' not found")
        sys.exit(1)

    results = []
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(results_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            results.append(json.load(handle))
    return results


def is_correct(result: dict) -> bool:
    """Support this repo's equivalent field, with text2sql compatibility."""
    if "equivalent" in result:
        return bool(result.get("equivalent"))
    return bool(result.get("is_correct"))


def analyze(results: list[dict]) -> None:
    """Print analysis of benchmark results."""
    if not results:
        print("No results found.")
        return

    total = len(results)
    correct = [result for result in results if is_correct(result)]
    failed = [result for result in results if not result.get("success", False)]
    timed_out = [result for result in results if result.get("timed_out", False)]

    accuracy = len(correct) / total * 100 if total else 0

    print("=" * 72)
    print("FINANCEBENCH RESULTS")
    print("=" * 72)
    print(f"Total samples:    {total}")
    print(f"Equivalent:       {len(correct)}/{total} ({accuracy:.1f}%)")
    print(f"Succeeded:        {total - len(failed)}/{total}")
    print(f"Failed:           {len(failed)}")
    print(f"Timed out:        {len(timed_out)}")
    print()

    print(f"{'ID':>4}  {'Status':<12}  {'Company':<24}  {'Doc'}")
    print("-" * 72)
    for result in sorted(results, key=lambda item: item.get("sample_id", 0)):
        sample_id = result.get("sample_id", "?")
        if result.get("timed_out"):
            status = "TIMEOUT"
        elif not result.get("success", False):
            status = "FAILED"
        elif is_correct(result):
            status = "EQUIVALENT"
        else:
            status = "WRONG"
        company = str(result.get("company") or "")[:24]
        doc_name = result.get("doc_name") or ""
        print(f"{sample_id:>4}  {status:<12}  {company:<24}  {doc_name}")

    if timed_out:
        print(f"\nTimed out samples: {[result.get('sample_id') for result in timed_out]}")

    non_timeout_failures = [
        result for result in failed if not result.get("timed_out")
    ]
    if non_timeout_failures:
        print("\nFailed samples (non-timeout):")
        for result in non_timeout_failures:
            print(
                f"  sample {result.get('sample_id')}: "
                f"{result.get('error', 'unknown error')}"
            )

    wrong = [
        result
        for result in results
        if result.get("success") and not is_correct(result) and not result.get("timed_out")
    ]
    if wrong:
        print("\nWrong answer samples:")
        for result in sorted(wrong, key=lambda item: item.get("sample_id", 0)):
            print(f"\n  sample {result.get('sample_id')}:")
            print(f"    question: {result.get('question')}")
            print(f"    gold:     {result.get('gold_answer')}")
            print(f"    agent:    {result.get('agent_answer')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze FinanceBench results")
    parser.add_argument("results_dir", help="Directory containing result JSON files")
    args = parser.parse_args()
    analyze(load_results(args.results_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
