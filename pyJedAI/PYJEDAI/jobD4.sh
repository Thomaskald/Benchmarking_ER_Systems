#!/bin/bash
#SBATCH --job-name=pyjedai_d4
#SBATCH --output=pyjedai_d4_%j.out
#SBATCH --error=pyjedai_d4_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo

set -euo pipefail
mkdir -p results

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

srun python3 -u pyjedai_ccer_searchD4.py

# Tip: watch live progress with:  tail -f pyjedai_d4_*.out
