#!/bin/bash
#SBATCH -A torch_pr_62_tandon_priority
#SBATCH --partition=l40s_public
#SBATCH --job-name=midterm-submission
#SBATCH --output=/scratch/%u/midterm-project/logs/midterm-submission-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=02:00:00

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADAPTER_DIR="${1:?Usage: sbatch slurm/generate_submission.slurm <adapter-dir> [output-path] [max-new-tokens]}"
OUTPUT_PATH="${2:-/scratch/$USER/midterm-project/outputs/submission.csv}"
MAX_NEW_TOKENS="${3:-2048}"

cd "$REPO_ROOT"
source "$REPO_ROOT/setup_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================="
echo "Submission Generation"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Adapter: ${ADAPTER_DIR}"
echo "Output: ${OUTPUT_PATH}"
echo "Start time: $(date)"
echo ""

"${MIDTERM_RUNTIME_VENV}/bin/python" scripts/generate_submission.py \
    --adapter-dir "$ADAPTER_DIR" \
    --output "$OUTPUT_PATH" \
    --max-new-tokens "$MAX_NEW_TOKENS"

echo ""
echo "Done: $(date)"
