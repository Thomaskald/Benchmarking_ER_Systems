#!/bin/bash
# Sudowoodo CPU install for the cluster login node.
#
# RUN THIS ON THE LOGIN NODE -- it needs internet. Compute nodes on `solo`
# are assumed to have none, so everything that must be downloaded (conda,
# wheels, the HuggingFace checkpoints) is fetched here and read from disk
# later.
#
# Idempotent: each stage is skipped if its output already exists, so a failed
# run can simply be repeated.
#
# Installs into $HOME/er_scalability/sudowoodo:
#     miniconda/   Python 3.7.10 interpreter (the cluster has only 3.10)
#     env/         the conda env with the pinned packages
#     src/         the upstream repo, patched for CPU
#     shims/       import stand-ins for apex and mlflow
#     hf_cache/    pre-downloaded roberta-base / distilbert-base-uncased
#     data/em/     per-dataset COL/VAL files (written by sudowoodo_prepare_data.py)
set -euo pipefail

ROOT="$HOME/er_scalability/sudowoodo"
CONDA_DIR="$ROOT/miniconda"
ENV_DIR="$ROOT/env"
SRC="$ROOT/src"
SHIMS="$ROOT/shims"
HF_CACHE="$ROOT/hf_cache"

mkdir -p "$ROOT"

# ---------------------------------------------------------------- 1. Miniforge
# Sudowoodo is Python 3.7.10. The 2020-era pinned wheels (numpy 1.19.2,
# torch 1.9.0, transformers 4.9.2) have no cp310 builds, so the cluster's
# system 3.10 cannot run this code.
#
# Miniforge rather than Miniconda, deliberately: Miniforge is BSD-3 and
# defaults to the conda-forge channel, so nothing here touches
# repo.anaconda.com or Anaconda's `defaults` channel. Anaconda's terms of
# service require a paid licence for organisations above a headcount
# threshold, and a university generally clears it -- not a risk worth taking
# on a shared cluster for a package manager we only need as a Python
# installer. conda-forge carries python 3.7.10 exactly.
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "==> installing Miniforge into $CONDA_DIR"
    curl -fsSL -o "$ROOT/miniforge.sh" \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$ROOT/miniforge.sh" -b -p "$CONDA_DIR"
    rm -f "$ROOT/miniforge.sh"
else
    echo "==> Miniforge already present, skipping"
fi

# ------------------------------------------------------------------- 2. Env
# --override-channels -c conda-forge makes the channel explicit rather than
# inherited, so this cannot silently fall back to `defaults` if the user has
# a ~/.condarc from somewhere else.
if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "==> creating Python 3.7.10 env at $ENV_DIR"
    "$CONDA_DIR/bin/conda" create -y -p "$ENV_DIR" \
        --override-channels -c conda-forge python=3.7.10
else
    echo "==> env already present, skipping"
fi
PY="$ENV_DIR/bin/python"
echo "==> $("$PY" --version 2>&1)"

# ---------------------------------------------------------------- 3. Packages
# requirements.txt as published does not install and is wider than the code
# needs. Deviations, all verified against the imports in the entity-matching
# path (train_bt.py, blocking.py, selfsl/*):
#
#   sklearn==0.0        -> scikit-learn==0.24.2   (`sklearn` is a stub package
#                          that just pulls scikit-learn; pinning 0.24.2 keeps a
#                          version contemporary with the paper that has cp37
#                          wheels)
#   torch==1.9.0+cu111  -> torch==1.9.0+cpu       (no GPU, per the protocol)
#   spacy, nltk, jsonlines, sentencepiece, regex, scipy
#                       -> DROPPED. Nothing in the EM path imports them; spacy
#                          2.2.3 in particular needs a source build on a modern
#                          toolchain and is pure risk for no gain. (They are
#                          used, if at all, by column_type_detection/, which
#                          this study does not run.)
#   protobuf==3.20.3    -> ADDED. tensorboardX 2.0 breaks against protobuf 4.x,
#                          which pip would otherwise resolve to.
#   apex, mlflow        -> shimmed, see stage 5.
echo "==> installing packages"
"$PY" -m pip install --no-cache-dir --upgrade "pip<24" "setuptools<60" wheel
"$PY" -m pip install --no-cache-dir "numpy==1.19.2"
"$PY" -m pip install --no-cache-dir "torch==1.9.0+cpu" \
    -f https://download.pytorch.org/whl/torch_stable.html
"$PY" -m pip install --no-cache-dir \
    "transformers==4.9.2" \
    "scikit-learn==0.24.2" \
    "tensorboardX==2.0" \
    "protobuf==3.20.3" \
    "tqdm==4.41.0"
# pandas is not a Sudowoodo dependency -- sudowoodo_prepare_data.py needs it to
# read the study's tables and split CSVs. 1.3.5 is the last release with cp37
# wheels; it requires numpy>=1.17.3, so the pinned 1.19.2 stands and is not
# silently upgraded out from under torch.
"$PY" -m pip install --no-cache-dir "pandas==1.3.5"

# -------------------------------------------------------------- 4. Source
if [ ! -d "$SRC/.git" ]; then
    echo "==> cloning megagonlabs/sudowoodo"
    git clone --depth 1 https://github.com/megagonlabs/sudowoodo.git "$SRC"
else
    echo "==> source already cloned, skipping"
fi

# --------------------------------------------------------------- 5. Shims
# Two imports are unconditional at module scope but only *used* behind flags
# this study never sets. Rather than editing three upstream files, they are
# satisfied by stand-in modules on PYTHONPATH -- the upstream source stays
# byte-identical except for the two device lines in stage 6, which keeps the
# "what did you change from the authors' code" list in the paper short.
echo "==> writing shims"
mkdir -p "$SHIMS/apex"
cat > "$SHIMS/apex/__init__.py" <<'EOF'
"""CPU stand-in for NVIDIA Apex.

Sudowoodo does `from apex import amp` at module scope in blocking.py,
selfsl/dm.py and selfsl/barlow_twins_simclr.py, so the import must succeed
even though every call site is guarded by `--fp16`. This study runs CPU-only
and never passes --fp16, so nothing here is ever reached. Each entry point
raises instead of degrading quietly: a silent no-op would let a stray --fp16
change the numerics without anyone noticing.
"""


class _Amp:
    def initialize(self, *args, **kwargs):
        raise RuntimeError(
            "apex.amp.initialize called, but this is a CPU-only install with "
            "no Apex. Drop the --fp16 flag.")

    def scale_loss(self, *args, **kwargs):
        raise RuntimeError(
            "apex.amp.scale_loss called, but this is a CPU-only install with "
            "no Apex. Drop the --fp16 flag.")


amp = _Amp()
EOF

cat > "$SHIMS/mlflow.py" <<'EOF'
"""No-op stand-in for mlflow.

train_bt.py, run_all_em.py and three modules under selfsl/ log to mlflow, but
mlflow is absent from requirements.txt. Installing it would pull sqlalchemy,
alembic, flask and a protobuf range that conflicts with the pinned
numpy 1.19.2 / tensorboardX 2.0, for output this study does not read -- the
benchmark takes its numbers from the worker's RESULT_JSON line. So the calls
are absorbed here.
"""


class _NullRun:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def log_param(*args, **kwargs):
    pass


def log_params(*args, **kwargs):
    pass


def log_metric(*args, **kwargs):
    pass


def log_metrics(*args, **kwargs):
    pass


def log_artifact(*args, **kwargs):
    pass


def set_tag(*args, **kwargs):
    pass


def set_tracking_uri(*args, **kwargs):
    pass


def start_run(*args, **kwargs):
    return _NullRun()


def end_run(*args, **kwargs):
    pass


def active_run():
    return None
EOF

# --------------------------------------------------------- 6. CPU device patch
# `model = model.cuda()` is unconditional in both trainers, two lines above a
# `device` variable that was already computed correctly. These are the only
# two edits to upstream source.
echo "==> patching hard-coded .cuda() calls"
for f in "$SRC/selfsl/barlow_twins_simclr.py" "$SRC/selfsl/dm.py"; do
    if grep -q "^    model = model.cuda()" "$f"; then
        sed -i 's/^    model = model\.cuda()$/    model = model.to(device)  # CPU-only run: was .cuda()/' "$f"
        echo "    patched $f"
    else
        echo "    $f already patched"
    fi
done

# ------------------------------------------------- 7. Pre-fetch the LM weights
# Compute nodes are assumed offline. transformers 4.9.2 will read these from
# HF_HOME; TRANSFORMERS_OFFLINE=1 in the job scripts makes a cache miss a loud
# error instead of a hang on a blocked socket.
echo "==> pre-downloading roberta-base and distilbert-base-uncased"
mkdir -p "$HF_CACHE"
HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE" "$PY" - <<'EOF'
from transformers import AutoModel, AutoTokenizer
for name in ("distilbert-base-uncased", "roberta-base"):
    print("  fetching", name)
    AutoTokenizer.from_pretrained(name)
    AutoModel.from_pretrained(name)
print("  done")
EOF

# ------------------------------------------------------------- 8. activate.sh
cat > "$ROOT/activate.sh" <<EOF
# source this before running anything Sudowoodo
export SUDOWOODO_ROOT="$ROOT"
export PATH="$ENV_DIR/bin:\$PATH"
export PYTHONPATH="$SHIMS:\${PYTHONPATH:-}"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
EOF
echo "==> wrote $ROOT/activate.sh"

# ------------------------------------------------------------- 9. Verify
echo "==> verifying"
# shellcheck disable=SC1090
source "$ROOT/activate.sh"
cd "$SRC"
python - <<'EOF'
import torch, transformers, sklearn, numpy, pandas, tensorboardX
from apex import amp            # the shim
import mlflow                   # the shim
from selfsl.barlow_twins_simclr import BarlowTwinsSimCLR
from selfsl.dataset import DMDataset
from selfsl.bt_dataset import BTDataset
print("  torch       ", torch.__version__, "cuda:", torch.cuda.is_available())
print("  transformers", transformers.__version__)
print("  sklearn     ", sklearn.__version__)
print("  numpy       ", numpy.__version__)
print("  pandas      ", pandas.__version__)
assert numpy.__version__ == "1.19.2", "numpy was upgraded -- check the pip order"
print("  imports OK")
EOF

cat <<EOF

==> install complete.

    source $ROOT/activate.sh
    cd $SRC

Next: build the COL/VAL data with sudowoodo_prepare_data.py, then run the
feasibility check with jobSmoke.sh BEFORE committing to any search.
EOF
