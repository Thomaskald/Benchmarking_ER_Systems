#!/bin/bash
#SBATCH --job-name=splink_eval
#SBATCH --output=splink_eval_%j.out
#SBATCH --error=splink_eval_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo

set -euo pipefail
mkdir -p results/pairs

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

# Best-config eval, both metric levels (pairwise cross-check + cluster-level
# B-cubed via Splink's own clusterer), all datasets, each in its own subprocess.
srun python3 -u splink_bestconfig_eval.py
# Tip: watch live progress with:  tail -f splink_eval_*.out
