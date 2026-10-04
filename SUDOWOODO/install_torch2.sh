#!/bin/bash
# Second environment: same code, same data, same hardware, modern PyTorch.
#
# Purpose is narrow. FEASIBILITY.md concludes Sudowoodo does not fit the
# frozen protocol, measured on the authors' pinned torch 1.9. Before taking
# that to the supervisor, rule out the obvious objection: "did you try a
# current PyTorch?" torch 2.x CPU is materially faster on transformer
# workloads.
#
# This changes NO part of the protocol -- same 4 cores, same 64G, same 90-min
# cap, same CPU-only, same data. Only the runtime differs.
#
# Uses the cluster's own Python 3.10 in a plain venv. No conda: modern wheels
# exist for 3.10, which was the entire reason the first env needed its own
# interpreter.
#
# RUN ON THE LOGIN NODE. Idempotent.
set -euo pipefail

ROOT="$HOME/er_scalability/sudowoodo"
VENV="$ROOT/venv_torch2"
SRC="$ROOT/src"
SHIMS="$ROOT/shims"
HF2="$ROOT/hf_cache_v2"

if [ ! -d "$SRC/selfsl" ]; then
    echo "error: no checkout at $SRC -- run install_sudowoodo.sh first" >&2
    exit 1
fi

# ------------------------------------------------------------------ 1. venv
if [ ! -x "$VENV/bin/python" ]; then
    echo "==> creating venv at $VENV with $(python3 --version 2>&1)"
    python3 -m venv "$VENV"
else
    echo "==> venv already present, skipping"
fi
PY="$VENV/bin/python"

# -------------------------------------------------------------- 2. packages
# transformers 4.46.3 is the last release that still exports `AdamW`, which
# selfsl/ imports; it is gone by 4.56. Sudowoodo's entire third-party API
# surface is four symbols (AutoModel, AutoTokenizer, AdamW,
# get_linear_schedule_with_warmup) and the code uses no numpy aliases removed
# in 2.x, so this is a low-risk swap.
#
# numpy is held below 2.0 only because it costs nothing to do so and removes
# one variable from a comparison whose whole point is to isolate the torch
# version.
echo "==> installing packages"
"$PY" -m pip install --no-cache-dir --upgrade pip wheel
"$PY" -m pip install --no-cache-dir "numpy<2"
"$PY" -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install --no-cache-dir \
    "transformers==4.46.3" \
    "scikit-learn" \
    "tensorboardX" \
    "tqdm"
# pandas is needed by sudowoodo_bestconfig_eval.py, which reads the split CSVs
# to recover left_id/right_id for the B-Cubed entity universe. The workers and
# search drivers never import it -- they read only the serialised COL/VAL
# files -- which is why a missing pandas stayed invisible until the evaluation
# stage.
"$PY" -m pip install --no-cache-dir "pandas<3"

# -------------------------------------------------- 3. HuggingFace cache
# transformers 4.9.2 and 4.46.3 use different on-disk cache layouts, so the
# existing hf_cache/ is unreadable here. Fetched separately on the login node
# for the same reason as before: compute nodes are assumed offline.
echo "==> pre-downloading distilbert-base-uncased"
mkdir -p "$HF2"
HF_HOME="$HF2" "$PY" - <<'EOF'
from transformers import AutoModel, AutoTokenizer
# roberta is not fetched: it measured 2.2x slower than distilbert and is
# already ruled out.
name = "distilbert-base-uncased"
print("  fetching", name)
AutoTokenizer.from_pretrained(name)
AutoModel.from_pretrained(name)
print("  done")
EOF

# ---------------------------------------------------------- 4. activate
cat > "$ROOT/activate_torch2.sh" <<EOF
# source this INSTEAD OF activate.sh to use the modern-torch environment
export SUDOWOODO_ROOT="$ROOT"
export PATH="$VENV/bin:\$PATH"
export PYTHONPATH="$SHIMS:\${PYTHONPATH:-}"
export HF_HOME="$HF2"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
EOF
echo "==> wrote $ROOT/activate_torch2.sh"

# ------------------------------------------------------------- 5. verify
echo "==> verifying"
# shellcheck disable=SC1090
source "$ROOT/activate_torch2.sh"
cd "$SRC"
python - <<'EOF'
import torch, transformers, sklearn, numpy, pandas
from apex import amp            # the shim
import mlflow                   # the shim
from transformers import AdamW  # gone in 4.56; present here
from selfsl.barlow_twins_simclr import BarlowTwinsSimCLR
print("  python      ", ".".join(map(str, __import__("sys").version_info[:3])))
print("  torch       ", torch.__version__, "cuda:", torch.cuda.is_available())
print("  transformers", transformers.__version__)
print("  numpy       ", numpy.__version__)
print("  pandas      ", pandas.__version__)
print("  imports OK")
EOF

cat <<EOF

==> done. Compare against the torch 1.9 numbers with:

    sbatch jobSmokeTorch2.sh
EOF
