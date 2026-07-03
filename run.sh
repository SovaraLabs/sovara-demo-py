#!/usr/bin/env bash
set -euo pipefail

# Run a range of FinanceBench samples in parallel and queue each run for annotation.
#
# Example:
#   SAMPLE_START=0 SAMPLE_STOP=10 MAX_PARALLEL=3 ./run.sh
#
# SAMPLE_STOP is exclusive, so the example runs sample ids 0 through 9.

SAMPLE_START="${SAMPLE_START:-0}"
SAMPLE_STOP="${SAMPLE_STOP:-}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
LOG_DIR="${LOG_DIR:-logs/queue_samples_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-false}"

if [[ -z "$SAMPLE_STOP" ]]; then
    echo "Error: set SAMPLE_STOP, for example: SAMPLE_START=0 SAMPLE_STOP=10 ./run.sh" >&2
    exit 1
fi

if ! [[ "$SAMPLE_START" =~ ^[0-9]+$ && "$SAMPLE_STOP" =~ ^[0-9]+$ ]]; then
    echo "Error: SAMPLE_START and SAMPLE_STOP must be non-negative integers" >&2
    exit 1
fi

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" -lt 1 ]]; then
    echo "Error: MAX_PARALLEL must be a positive integer" >&2
    exit 1
fi

if [[ "$SAMPLE_START" -ge "$SAMPLE_STOP" ]]; then
    echo "Error: SAMPLE_START must be less than SAMPLE_STOP" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

cleanup() {
    trap - INT TERM
    echo
    echo "Stopping running jobs..."
    jobs -rp | while read -r pid; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait || true
}

trap cleanup INT TERM

run_one() {
    local sample_id="$1"
    local log_file="$LOG_DIR/sample_${sample_id}.log"

    echo "[sample $sample_id] starting"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "uv run python main.py --sample-id $sample_id --queue-for-annotation"
        return 0
    fi

    if uv run python main.py --sample-id "$sample_id" --queue-for-annotation > "$log_file" 2>&1; then
        echo "[sample $sample_id] done -> $log_file"
    else
        local exit_code=$?
        echo "[sample $sample_id] failed with exit code $exit_code -> $log_file"
        return "$exit_code"
    fi
}

echo "Queueing samples [$SAMPLE_START, $SAMPLE_STOP) with MAX_PARALLEL=$MAX_PARALLEL"
echo "Logs: $LOG_DIR"

failures=0
total=$((SAMPLE_STOP - SAMPLE_START))
launched=0
pids=()

for sample_id in $(seq "$SAMPLE_START" $((SAMPLE_STOP - 1))); do
    launched=$((launched + 1))
    echo "[$launched/$total] launching sample $sample_id"

    run_one "$sample_id" &
    pids+=("$!")

    while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_PARALLEL" ]]; do
        sleep 1
    done
done

for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failures=$((failures + 1))
    fi
done

if [[ "$failures" -gt 0 ]]; then
    echo "Finished with $failures failed sample(s)."
    exit 1
fi

echo "Finished successfully."
