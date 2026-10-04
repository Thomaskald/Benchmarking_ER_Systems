#!/bin/bash
#SBATCH --job-name=deeper_cache
#SBATCH --output=deeper_cache_%j.out
#SBATCH --error=deeper_cache_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# One-off. Builds embeddings/ft_cache_<DS>.npz for all 10 datasets.
# Must finish before any DeepER search job is submitted.
#
# Only THIS job needs the fastText binding; the workers need nothing new. It is
# installed in an isolated venv so it cannot disturb the system packages the
# seven existing frameworks depend on. Falls back to system python3 if the venv
# is absent (e.g. gensim installed instead -- the builder handles both).
PY=python3
if [ -x "$HOME/er_scalability/ftenv/bin/python" ]; then
  PY="$HOME/er_scalability/ftenv/bin/python"
fi
echo "using interpreter: $PY"
srun $PY -u deeper_build_cache.py
