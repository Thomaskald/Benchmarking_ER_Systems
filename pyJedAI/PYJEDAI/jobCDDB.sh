#!/bin/bash
#SBATCH --job-name=pyjedai_der_cddb
#SBATCH --output=pyjedai_der_cddb_%j.out
#SBATCH --error=pyjedai_der_cddb_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail
mkdir -p results
# No conda: packages installed for python3 (user pip). Put user bin on PATH.
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false
# DER embeddings workflow downloads language models -> give them a cache dir
# (pre-cache on the login node first if compute nodes have no internet).
export HF_HOME="$HOME/.cache/huggingface"
export GENSIM_DATA_DIR="$HOME/gensim-data"
# B=50, threshold sampled, 1.5h/config cap (set in the .py).
srun python3 -u pyjedai_der_searchCDDB.py
# Tip: watch live progress with:  tail -f pyjedai_der_cddb_*.out
