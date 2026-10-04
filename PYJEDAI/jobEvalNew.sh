#!/bin/bash
#SBATCH --job-name=pyjedai_neweval
#SBATCH --output=pyjedai_neweval_%j.out
#SBATCH --error=pyjedai_neweval_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail
mkdir -p results results/pairs
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$HOME/.cache/huggingface"
export GENSIM_DATA_DIR="$HOME/gensim-data"
# 8 semantic CCER re-runs (~2.9h) + 2 syntactic DER re-runs (~4min)
srun python3 -u pyjedai_new_bestconfig_eval.py
# Tip: tail -f pyjedai_neweval_*.out
