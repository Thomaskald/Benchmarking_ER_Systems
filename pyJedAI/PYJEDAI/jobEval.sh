#!/bin/bash
#SBATCH --job-name=pyjedai_eval
#SBATCH --output=pyjedai_eval_%j.out
#SBATCH --error=pyjedai_eval_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo

set -euo pipefail
mkdir -p results/pairs

# No conda: packages installed for python3 (user pip). Put user bin on PATH.
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

# DER datasets (CORA, CDDB) build embeddings -> give the models a cache dir
# (pre-cache on the login node first if compute nodes have no internet).
export HF_HOME="$HOME/.cache/huggingface"
export GENSIM_DATA_DIR="$HOME/gensim-data"

srun python3 -u pyjedai_bestconfig_eval.py
