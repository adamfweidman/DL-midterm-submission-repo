#!/bin/bash
#SBATCH -A torch_pr_62_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --job-name=midterm-loop-validation
#SBATCH --output=/scratch/%u/midterm-project/logs/midterm-loop-validation-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC_PATH="${1:-$REPO_ROOT/state/manual-loop-validation.yaml}"

cd "$REPO_ROOT"

source "$REPO_ROOT/setup_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================="
echo "Midterm Experiment Run"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Account: ${SLURM_JOB_ACCOUNT}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Spec: ${SPEC_PATH}"
echo "Start time: $(date)"
echo "Python: ${MIDTERM_RUNTIME_VENV}/bin/python"
echo ""

"${MIDTERM_RUNTIME_VENV}/bin/python" scripts/confirm_run_spec.py --spec "$SPEC_PATH" --repo-root "$REPO_ROOT"

echo ""
echo "Running experiment iteration..."
"${MIDTERM_RUNTIME_VENV}/bin/python" runner.py
