#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/.cache/huggingface"
export UV_CACHE_DIR="$TASK_ROOT/.cache/uv"
export MPLCONFIGDIR="$TASK_ROOT/.cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASK_PYTHON="$TASK_ROOT/.venv/bin/python"
if [[ ! -x "$TASK_PYTHON" ]]; then
  TASK_PYTHON=python3
fi
exec "$TASK_PYTHON" src/pipeline.py "$@"
