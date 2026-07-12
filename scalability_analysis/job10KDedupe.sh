#!/bin/bash
#SBATCH --job-name=dedupe_scale_10k
#SBATCH --output=dedupe_scale_10k_%j.out
#SBATCH --error=dedupe_scale_10k_%j.err
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
python3 -c "import zope; zope.__path__.append('$HOME/.local/lib/python3.10/site-packages/zope'); import dedupe; print('dedupe import OK')"
srun python3 -u dedupe_scalability_search10K.py
