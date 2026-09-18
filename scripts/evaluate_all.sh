#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/paper/llama2-7b.yaml}"
OUTPUT_DIR="${2:?Provide a distinct evaluation output directory for this training seed}"
shift 2
cd "$PROJECT_DIR"
for benchmark in asdiv gsm8k math mawps svamp; do
  "$PYTHON_BIN" -m keir validate-benchmark --benchmark "$benchmark" \
    --input "data/benchmarks/${benchmark}_test.jsonl"
done
for benchmark in asdiv gsm8k math mawps svamp; do
  "$PYTHON_BIN" -m keir evaluate --config "$CONFIG" \
    --input "data/benchmarks/${benchmark}_test.jsonl" \
    --output "${OUTPUT_DIR}/${benchmark}_keir.jsonl" "$@"
done
