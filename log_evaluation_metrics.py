"""Log batch-level metrics from the sample results emitted by main.py."""

import argparse
import json
from pathlib import Path
from statistics import mean


def summarize(eval_id, log_dir, start, stop):
    results = []
    decoder = json.JSONDecoder()
    for sample_id in range(start, stop):
        text = (Path(log_dir) / f"sample_{sample_id}.log").read_text()
        result = None
        for offset, char in enumerate(text):
            if char != "{" or (offset and text[offset - 1] != "\n"):
                continue
            try:
                value, _ = decoder.raw_decode(text[offset:])
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("sample_id") == sample_id and value.get("eval_id") == eval_id:
                result = value
        if result is None:
            raise ValueError(f"Missing judged result for sample {sample_id}")
        results.append(result)
    completed = [r for r in results if isinstance(r.get("equivalent"), bool)]
    failed = [r for r in results if not isinstance(r.get("equivalent"), bool)]
    correct = sum(r["equivalent"] for r in completed)
    metrics = {
        "completed_samples": len(completed),
        "failed_samples": len(failed),
        "requested_samples": len(results),
        "correct_samples": correct,
        "total_samples": len(results),
        "evaluation_failure": json.dumps(failed[0]["failure"]) if failed else "",
    }
    if completed:
        metrics["avg_latency_seconds"] = mean(r["agent_latency_seconds"] for r in completed)
    if not failed:
        metrics["overall_accuracy"] = correct / len(results)
    return metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()
    metrics = summarize(args.eval_id, args.log_dir, args.start, args.stop)
    from agent import sovara_client

    sovara_client.log(eval_run_id=args.eval_id, **metrics)
    print(json.dumps({"eval_id": args.eval_id, **metrics}, indent=2))
