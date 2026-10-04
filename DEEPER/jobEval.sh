#!/bin/bash
#SBATCH --job-name=deeper_eval
#SBATCH --output=deeper_eval_%j.out
#SBATCH --error=deeper_eval_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail
mkdir -p results
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false
# Re-runs the best config of all 10 datasets (~1.6h measured) and writes
# results/deeper_bestconfig_eval.csv. Needs the ft_cache_*.npz files.
srun python3 -u deeper_bestconfig_eval.py
