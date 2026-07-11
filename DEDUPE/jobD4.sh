#!/bin/bash
#SBATCH --job-name=dedupe_d4
#SBATCH --output=dedupe_d4_%j.out
#SBATCH --error=dedupe_d4_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo

set -euo pipefail
mkdir -p results

# No conda: packages installed directly for python3 (system / user pip).
export PATH="$HOME/.local/bin:$PATH"
# Force user site-packages ahead of system dist-packages so the complete
# zope namespace (with zope.index, needed by dedupe) resolves correctly.
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

srun python3 -u dedupe_ccer_searchD4.py
