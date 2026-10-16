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
ARGS=("$@")
HF_STAGE="all"
HF_ADAPTERS_DIR="$TASK_ROOT/adapters"
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --stage) if ((i+1 < ${#ARGS[@]})); then HF_STAGE="${ARGS[$((i+1))]}"; fi ;;
    --adapters_dir) if ((i+1 < ${#ARGS[@]})); then HF_ADAPTERS_DIR="${ARGS[$((i+1))]}"; fi ;;
  esac
done
case "$HF_STAGE" in
  all|B|sanitize)
    "$TASK_PYTHON" src/hf_adapters.py download \
      --name best_sanitizer \
      --destination "$HF_ADAPTERS_DIR/hf_downloaded/best_sanitizer"
    ;;
  attacks)
    for suffix in raw san; do
      name="victim_${suffix}"
      local_dir="$HF_ADAPTERS_DIR/$name"
      if [[ ! -f "$local_dir/adapter_config.json" || ( ! -f "$local_dir/adapter_model.safetensors" && ! -f "$local_dir/adapter_model.bin" ) ]]; then
        "$TASK_PYTHON" src/hf_adapters.py download \
          --name "$name" \
          --destination "$HF_ADAPTERS_DIR/hf_downloaded/$name"
      fi
    done
    ;;
esac
exec "$TASK_PYTHON" src/pipeline.py "$@"
