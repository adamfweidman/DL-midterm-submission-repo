#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MIDTERM_PYTHON_MODULE="${MIDTERM_PYTHON_MODULE:-anaconda3/2025.06}"
if type module >/dev/null 2>&1; then
  module load "$MIDTERM_PYTHON_MODULE" >/dev/null 2>&1 || true
fi

export MIDTERM_SCRATCH_ROOT="${MIDTERM_SCRATCH_ROOT:-/scratch/$USER/midterm-project}"
export MIDTERM_DATA_DIR="${MIDTERM_DATA_DIR:-$MIDTERM_SCRATCH_ROOT/data/kaggle/svg-generation}"
export MIDTERM_OUTPUT_ROOT="${MIDTERM_OUTPUT_ROOT:-$MIDTERM_SCRATCH_ROOT/outputs}"
export MIDTERM_LOG_ROOT="${MIDTERM_LOG_ROOT:-$MIDTERM_SCRATCH_ROOT/logs}"
export MIDTERM_CACHE_ROOT="${MIDTERM_CACHE_ROOT:-$MIDTERM_SCRATCH_ROOT/cache}"
export MIDTERM_RUNTIME_VENV="${MIDTERM_RUNTIME_VENV:-$MIDTERM_SCRATCH_ROOT/.venv}"
export MIDTERM_SLURM_ACCOUNT="${MIDTERM_SLURM_ACCOUNT:-torch_pr_62_tandon_priority}"
export MIDTERM_SLURM_PARTITION="${MIDTERM_SLURM_PARTITION:-h100_tandon}"
export MIDTERM_PYTHON_BIN="${MIDTERM_PYTHON_BIN:-$(command -v python)}"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT="$MIDTERM_RUNTIME_VENV"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/$USER/.uv-cache}"
export HF_HOME="$MIDTERM_CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export TRANSFORMERS_CACHE="$HUGGINGFACE_HUB_CACHE"
export KAGGLEHUB_CACHE="$MIDTERM_CACHE_ROOT/kagglehub"
export XDG_CACHE_HOME="$MIDTERM_CACHE_ROOT/xdg"
export XDG_CONFIG_HOME="$MIDTERM_CACHE_ROOT/config"
export TORCH_HOME="$MIDTERM_CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$MIDTERM_CACHE_ROOT/triton"
export TMPDIR="$MIDTERM_SCRATCH_ROOT/tmp"

mkdir -p \
  "$MIDTERM_OUTPUT_ROOT" \
  "$MIDTERM_LOG_ROOT" \
  "$MIDTERM_CACHE_ROOT" \
  "$MIDTERM_DATA_DIR" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$KAGGLEHUB_CACHE" \
  "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TMPDIR"

target_python_version="$("$MIDTERM_PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
current_python_version=""
if [ -f "$MIDTERM_RUNTIME_VENV/pyvenv.cfg" ]; then
  current_python_version="$(awk -F' = ' '/^version_info = / {print $2}' "$MIDTERM_RUNTIME_VENV/pyvenv.cfg" || true)"
fi

if [ ! -x "$MIDTERM_RUNTIME_VENV/bin/python" ] || [ "$current_python_version" != "$target_python_version" ]; then
  rm -rf "$MIDTERM_RUNTIME_VENV"
  uv venv --python "$MIDTERM_PYTHON_BIN" "$MIDTERM_RUNTIME_VENV"
fi

ln -sfn "$MIDTERM_RUNTIME_VENV" "$REPO_ROOT/.venv"
