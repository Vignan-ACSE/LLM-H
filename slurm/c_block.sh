#!/bin/bash
#SBATCH --job-name=c_block
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn4_mangala partition
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Block C on CPU (the partition only hands out GPU slots, so one is reserved).
# 1. The C1-C3 tests, which must pass before calibration touches real data.
# 2. C4-C8: scripts/run_calibration.py -> data/c4 ... data/c8
# 3. results/ refreshed with the Block C tables and figures.
# Needs data/b3/selfcheck_scores.parquet from slurm/b3.sh for SelfCheck-NLI;
# without it C4 runs on the other sampling scores.
# With LIMIT=<n> it runs a short check into data/c_test/ and skips step 3.

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

echo "== tests started at $(date)"
srun python -m pytest src/tests -q

CAL_ARGS=()
if [ -n "${LIMIT:-}" ]; then
    CAL_ARGS=(--trials 5 --size-repeats 10 --output-root data/c_test)
fi

echo "== calibration started at $(date)"
srun python scripts/run_calibration.py "${CAL_ARGS[@]}"

if [ -z "${LIMIT:-}" ]; then
    echo "== results started at $(date)"
    srun python scripts/collect_results.py
fi

job_done
