#!/bin/bash
#SBATCH --job-name=pyjedai_sem_d9
#SBATCH --output=pyjedai_sem_d9_%j.out
#SBATCH --error=pyjedai_sem_d9_%j.err
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
# Semantic workflow downloads language models -> give them a cache dir
# (pre-cache on the login node first if compute nodes have no internet).
export HF_HOME="$HOME/.cache/huggingface"
export GENSIM_DATA_DIR="$HOME/gensim-data"
# B=50, threshold sampled (no sweep), 1.5h/config cap (set in the .py).
srun python3 -u pyjedai_sem_ccer_searchD9.py
# Tip: watch live progress with:  tail -f pyjedai_sem_d9_*.out
