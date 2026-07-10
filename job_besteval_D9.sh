#!/bin/bash
#SBATCH --job-name=lt_evald9
#SBATCH --output=lt_evald9_%j.out
#SBATCH --error=lt_evald9_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo

set -euo pipefail
mkdir -p results/pairs

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export HF_HOME="$HOME/.cache/huggingface"

echo "Node: $(hostname)"
echo "Using python: $(which python3)"
python3 -c "import linktransformer, torch; print('linktransformer', linktransformer.__version__, '| torch', torch.__version__)"

# Best-config eval for D9: retrains the winning config, then pairwise + B-cubed.
srun python3 -u linktransformer_bestconfig_evalD9.py

# Tip: watch live progress with:  tail -f lt_evald9_*.out
