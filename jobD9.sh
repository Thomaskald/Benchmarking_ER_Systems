#!/bin/bash
#SBATCH --job-name=lt_d9
#SBATCH --output=lt_d9_%j.out
#SBATCH --error=lt_d9_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail
mkdir -p results
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
# Keep HuggingFace model cache in a known place (avoids re-downloading per config)
export HF_HOME="$HOME/.cache/huggingface"
echo "Node: $(hostname)"
echo "Using python: $(which python3)"
python3 -c "import linktransformer, torch; print('linktransformer', linktransformer.__version__, '| torch', torch.__version__)"
srun python3 -u linktransformer_ccer_searchD9.py
# Tip: watch live progress with:  tail -f lt_d9_*.out
