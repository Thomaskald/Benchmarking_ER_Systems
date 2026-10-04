#!/bin/bash
#SBATCH --job-name=deeper_cache_syn
#SBATCH --output=deeper_cache_syn_%j.out
#SBATCH --error=deeper_cache_syn_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# One-off. Builds embeddings/ft_cache_SYN<size>.npz for the scalability ladder.
# Must finish before job10KDeepER.sh / jobDeepER.sh.
# Uses the isolated fastText venv created for the main cache build.
PY=python3
if [ -x "$HOME/er_scalability/ftenv/bin/python" ]; then
  PY="$HOME/er_scalability/ftenv/bin/python"
fi
echo "using interpreter: $PY"
srun $PY -u deeper_build_cache_syn.py
