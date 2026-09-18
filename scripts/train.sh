#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/paper/llama2-7b.yaml}"
if (( $# > 0 )); then shift; fi
cd "$PROJECT_DIR"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" \
  -m keir train --config "$CONFIG" "$@"
