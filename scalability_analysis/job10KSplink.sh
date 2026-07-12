#!/bin/bash
#SBATCH --job-name=splink_scale_10k
#SBATCH --output=splink_scale_10k_%j.out
#SBATCH --error=splink_scale_10k_%j.err
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
# Splink uses DuckDB (no Spark, no Java needed)
python3 -c "import splink; print('splink import OK')"
srun python3 -u splink_scalability_search10K.py
