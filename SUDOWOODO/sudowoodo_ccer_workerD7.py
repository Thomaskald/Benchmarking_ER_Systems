"""Sudowoodo on CCER dataset D7: one configuration in, one
RESULT_JSON line out.

Evaluated as a pair classifier over the fixed train/valid/test split files --
identical treatment to MAGELLAN, RECORDLINKAGE, LINKTRANSFORMER and DEEPER, so
Sudowoodo's own blocking (`--blocking` / `--k`) is not exercised. Running it
would put Sudowoodo's B-Cubed on a candidate universe no other framework's is
computed on.

One worker per dataset, matching the layout of every other framework in
this study. The raw per-dataset detail (paths, delimiters, attribute
alignment, the SCADS rename map) lives in sudowoodo_prepare_data.py, which
normalises all ten datasets into data/em/<TASK>/ -- so unlike the DeepER
workers this file carries only the dataset name.

DIFFERENCES FROM UPSTREAM, all forced and all reported:

  * threshold -- upstream's selfsl/utils.evaluate sweeps np.arange(0, 1.0,
    0.05), a 20-point grid. The frozen protocol is the 101-point
    np.arange(0, 1.001, 0.01). This worker collects probabilities and sweeps
    the protocol grid itself rather than calling upstream's evaluate.
  * seed -- Sudowoodo seeds from --run_id (train_bt.py:88). Set to 42.
  * --size -- NOT used. It truncates the training set to 500 pairs and
    redirects the validation set to train.txt (train_bt.py:98-101), which is
    incompatible with choosing a threshold on validation. Note that 500 is
    the authors' own semi-supervised setting, so this worker runs Sudowoodo
    with far more supervision than the paper does.
  * fp16 / Apex -- not used, CPU-only run.
  * --bootstrap -- NOT AVAILABLE under this protocol, and not merely
    disabled. selfsl/bootstrap.py reads `hp.size` as the number of LABELLED
    training pairs and treats `all_sims[hp.size:]` as an unlabelled pool to
    pseudo-label. With the full training split -- which the frozen protocol
    requires, and which every other framework in this study gets -- there is
    no unlabelled pool: `hp.size=None` raises TypeError at bootstrap.py:119,
    and setting it to len(trainset) makes the pool empty and raises
    IndexError two lines later. Sudowoodo's pseudo-labelling is defined only
    in the low-label regime the paper targets. Confirmed empirically: jobs
    173391/173392 returned this TypeError on 8 of 10 configs.
  * --clustering -- not used; it adds a KMeans pass over the corpus on top
    of an already over-budget pre-training phase.

The epoch loop, the model, the SSL step and the fine-tune step are upstream's
own (selfsl.barlow_twins_simclr), not reimplementations. Model selection is
upstream's: the epoch with the best validation F1 wins.
"""

import sys
import os
import json
import time
import random
import resource
import warnings
import logging
import contextlib

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np
import torch
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score

SRC = os.path.join(os.environ.get("SUDOWOODO_ROOT",
                                  os.path.expanduser("~/er_scalability/sudowoodo")),
                   "src")
sys.path.insert(0, SRC)
os.chdir(SRC)

from selfsl.barlow_twins_simclr import BarlowTwinsSimCLR, selfsl_step, fine_tune_step
from selfsl.bt_dataset import BTDataset
from selfsl.dataset import DMDataset

TASK = "D7"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)
SEED = 42


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def make_hp(cfg):
    """train_bt.py's argparse defaults, with the searched knobs applied."""
    import argparse
    return argparse.Namespace(
        task=cfg["task"], task_type="em", logdir="search_logs/",
        run_id=SEED,
        batch_size=int(cfg["batch_size"]), max_len=int(cfg["max_len"]),
        size=None, lr=float(cfg["lr"]),
        n_epochs=int(cfg["n_epochs"]), lm=cfg["lm"], fp16=False,
        ssl_method=cfg["ssl_method"], n_ssl_epochs=int(cfg["n_ssl_epochs"]),
        da=cfg["da"], cutoff_ratio=0.05,
        clustering=False, num_clusters=90,
        # bootstrap is hard-False, not sampled: see the docstring above.
        zero=False, bootstrap=False, multiplier=8,
        projector=str(cfg["projector"]), scale_loss=1.0 / 256, lambd=3.9e-3,
        alpha_bt=0.001,
        save_ckpt=False, use_saved_ckpt=False,
        blocking=False, threshold=None, k=10, tfidf=False, lm_only=False,
        mlflow_tag=None,
    )


def collect_probs(model, iterator):
    """Match probability and gold label for every pair, in order."""
    probs, ys = [], []
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            x1, x2, x12, y = batch
            logits = model(2, x1, x2, x12)
            probs += logits.softmax(dim=1)[:, 1].cpu().numpy().tolist()
            ys += y.cpu().numpy().tolist()
    model.train()
    return np.asarray(probs), np.asarray(ys, dtype=int)


def sweep(probs, y):
    """Best F1 over the protocol's 101-point grid."""
    best_f1, best_t = -1.0, 0.5
    for t in THRESHOLD_GRID:
        f1 = f1_score(y, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_f1, best_t


def main():
    cfg = json.loads(sys.argv[1])
    cfg["task"] = TASK
    t_start = time.time()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    hp = make_hp(cfg)
    path = os.path.join("data", "em", hp.task)

    trainset = DMDataset(os.path.join(path, "train.txt"), lm=hp.lm,
                         size=None, max_len=hp.max_len, da=None)
    validset = DMDataset(os.path.join(path, "valid.txt"), lm=hp.lm,
                         size=None, max_len=hp.max_len)
    testset = DMDataset(os.path.join(path, "test.txt"), lm=hp.lm,
                        size=None, max_len=hp.max_len)
    # size=10000 is hard-coded upstream (train_bt.py:144) and is NOT a
    # per-dataset choice. The unlabeled corpus is derived from train.txt
    # alone, so no framework-specific data reaches the model.
    nolabel = BTDataset(os.path.join(path, "train_no_label.txt"), lm=hp.lm,
                        size=10000, max_len=hp.max_len, da=hp.da)

    ft_batch = max(hp.batch_size // 2, 1)      # upstream: half of barlow twins'
    ssl_iter = data.DataLoader(dataset=nolabel, batch_size=hp.batch_size,
                               shuffle=True, num_workers=0,
                               collate_fn=nolabel.pad)
    train_iter = data.DataLoader(dataset=trainset, batch_size=ft_batch,
                                 shuffle=True, num_workers=0,
                                 collate_fn=trainset.pad)
    valid_iter = data.DataLoader(dataset=validset, batch_size=hp.batch_size,
                                 shuffle=False, num_workers=0,
                                 collate_fn=validset.pad)
    test_iter = data.DataLoader(dataset=testset, batch_size=hp.batch_size,
                                shuffle=False, num_workers=0,
                                collate_fn=testset.pad)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm).to(device)
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    n_ssl = hp.n_ssl_epochs
    n_ssl_steps = len(nolabel) // hp.batch_size * n_ssl
    n_ft_steps = len(trainset) // ft_batch * max(hp.n_epochs - n_ssl, 0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=max(n_ssl_steps + n_ft_steps, 1))

    best_valid_f1, best_t = -1.0, 0.5
    best_probs_test, y_te = None, None
    best_epoch = None

    # Upstream's loop (selfsl/barlow_twins_simclr.py:565-...), with tensorboard
    # and mlflow logging dropped and the threshold sweep widened.
    for epoch in range(1, hp.n_epochs + 1):
        model.train()
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            if epoch <= n_ssl:
                selfsl_step(ssl_iter, train_iter, model, optimizer, scheduler, hp)
            else:
                fine_tune_step(train_iter, model, optimizer, scheduler, hp)

        if epoch > n_ssl:
            probs_va, y_va = collect_probs(model, valid_iter)
            f1_va, t_va = sweep(probs_va, y_va)
            if f1_va > best_valid_f1:
                best_valid_f1, best_t, best_epoch = f1_va, t_va, epoch
                best_probs_test, y_te = collect_probs(model, test_iter)

    if best_probs_test is None:
        raise RuntimeError("no fine-tuning epoch ran: n_epochs (%d) must exceed "
                           "n_ssl_epochs (%d)" % (hp.n_epochs, n_ssl))

    preds = (best_probs_test >= best_t).astype(int)
    test_p = precision_score(y_te, preds, zero_division=0)
    test_r = recall_score(y_te, preds, zero_division=0)
    test_f1 = f1_score(y_te, preds, zero_division=0)

    curve = []
    for t in THRESHOLD_GRID:
        pr = (best_probs_test >= t).astype(int)
        curve.append({"t": round(float(t), 3),
                      "precision": round(precision_score(y_te, pr, zero_division=0), 6),
                      "recall": round(recall_score(y_te, pr, zero_division=0), 6),
                      "f1": round(f1_score(y_te, pr, zero_division=0), 6)})

    pkeys = ["task", "lm", "ssl_method", "n_ssl_epochs", "n_epochs", "max_len",
             "batch_size", "lr", "da", "projector"]
    result = {
        "config_id": cfg.get("config_id"),
        "params": {k: cfg.get(k) for k in pkeys if k in cfg},
        "status": "OK",
        "chosen_threshold": round(best_t, 3),
        "valid_f1_at_threshold": round(best_valid_f1, 6),
        "test_point": {"t": round(best_t, 3), "precision": round(test_p, 6),
                       "recall": round(test_r, 6), "f1": round(test_f1, 6)},
        "pr_curve": curve,
        "best_epoch": best_epoch,
        "n_train_pairs": len(trainset),
        "time_sec": round(time.time() - t_start, 2),
        "peak_mem_mb": round(peak_mem_mb(), 1),
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
