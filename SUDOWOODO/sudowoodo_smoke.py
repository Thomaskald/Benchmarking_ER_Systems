"""Feasibility check: can Sudowoodo finish a config inside the 90-minute cap
on 4 CPU cores?

This was the main open question for Sudowoodo. Answering it by launching a
real search would burn a 5-hour job slot to learn one number. Instead this
builds the real model on the real data, times a handful of optimizer steps per
dataset, and extrapolates.

Why it measures EVERY dataset rather than one and scaling:

  * Step COUNT varies -- the fine-tune phase walks every training pair each
    epoch, and D6 has 193460 of them against D4's 1376.
  * Step COST varies too. Both `pad` functions pad to the longest sequence in
    the batch, not to max_len, so a dataset of short records (D4: title,
    authors, venue, year) gets cheaper steps than one carrying `description`
    or `abstract` (D3, D7) or 12 attributes (CORA). Transformer cost grows
    with sequence length, so this is not a small effect.

The language model is loaded once and reused across datasets -- it does not
depend on the data -- so covering all ten costs little more than covering one.

Nothing here writes results. It exists to decide whether Sudowoodo can be run
under the frozen protocol at all.

Usage:  python sudowoodo_smoke.py [--lm distilbert|roberta] [--tasks D4,D6]
"""

import argparse
import os
import sys
import time

import torch
from torch.utils import data
from transformers import AdamW

# This script lives next to the job scripts, not inside the checkout, so put
# the upstream source on the path and run from there -- Sudowoodo resolves
# data/em/<task> relative to the working directory.
SRC = os.path.join(os.environ.get("SUDOWOODO_ROOT",
                                  os.path.expanduser("~/er_scalability/sudowoodo")),
                   "src")
if not os.path.isdir(os.path.join(SRC, "selfsl")):
    raise SystemExit("no Sudowoodo checkout at %s -- run install_sudowoodo.sh "
                     "and source activate.sh first" % SRC)
sys.path.insert(0, SRC)
os.chdir(SRC)

from selfsl.barlow_twins_simclr import BarlowTwinsSimCLR
from selfsl.bt_dataset import BTDataset
from selfsl.dataset import DMDataset

TIME_CAP_SEC = 90 * 60      # the study's per-config cap -- frozen
N_WARMUP = 2
N_TIMED = 6
EVAL_BATCHES = 12           # eval is measured on a slice and scaled per pair;
                            # D6's valid split alone is 64487 pairs
SETTINGS = [(3, 50), (3, 10), (2, 5), (1, 3), (1, 1)]


def hparams(lm, batch_size, max_len, ssl_method):
    """train_bt.py's argparse defaults, with the knobs we vary set."""
    import argparse as _a
    return _a.Namespace(
        task="", task_type="em", logdir="smoke/", run_id=0,
        batch_size=batch_size, max_len=max_len, size=None, lr=5e-5,
        n_epochs=20, lm=lm, fp16=False,
        ssl_method=ssl_method, n_ssl_epochs=3,
        da="cutoff", cutoff_ratio=0.05,
        clustering=False, num_clusters=90,
        zero=False, bootstrap=False, multiplier=8,
        projector="4096", scale_loss=1.0 / 256, lambd=3.9e-3, alpha_bt=0.001,
        save_ckpt=False, use_saved_ckpt=False,
        blocking=False, threshold=None, k=10, tfidf=False, lm_only=False,
        mlflow_tag=None,
    )


def fmt(sec):
    if sec < 90:
        return "%.0fs" % sec
    if sec < 5400:
        return "%.0fm" % (sec / 60)
    return "%.1fh" % (sec / 3600)


def time_steps(fn):
    """Seconds per step, warm-up excluded. Also returns mean padded length."""
    lens = []
    for _ in range(N_WARMUP):
        lens.append(fn())
    t0 = time.time()
    for _ in range(N_TIMED):
        lens.append(fn())
    return (time.time() - t0) / N_TIMED, sum(lens) / float(len(lens))


def measure(task, model, optimizer, hp):
    """Per-step costs and sizes for one dataset."""
    path = os.path.join("data", "em", task)
    trainset = DMDataset(os.path.join(path, "train.txt"), lm=hp.lm,
                         size=None, max_len=hp.max_len, da=None)
    validset = DMDataset(os.path.join(path, "valid.txt"), lm=hp.lm,
                         size=None, max_len=hp.max_len)
    testset = DMDataset(os.path.join(path, "test.txt"), lm=hp.lm,
                        size=None, max_len=hp.max_len)
    # size=10000 is hard-coded in train_bt.py, so the pre-training corpus is
    # the same SIZE for every dataset -- but not the same cost, because the
    # records differ in length.
    nolabel = BTDataset(os.path.join(path, "train_no_label.txt"), lm=hp.lm,
                        size=10000, max_len=hp.max_len, da=hp.da)

    # -------------------------------------------------------- SSL step
    ssl_iter = data.DataLoader(dataset=nolabel, batch_size=hp.batch_size,
                               shuffle=True, num_workers=0,
                               collate_fn=nolabel.pad)
    ssl_batches = [iter(ssl_iter)]

    def ssl_step():
        try:
            yA, yB = next(ssl_batches[0])
        except StopIteration:
            ssl_batches[0] = iter(ssl_iter)
            yA, yB = next(ssl_batches[0])
        optimizer.zero_grad()
        if hp.ssl_method == "simclr":
            loss = model(0, yA, yB, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
        elif hp.ssl_method == "barlow_twins":
            loss = model(1, yA, yB, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
        else:
            alpha = 1 - hp.alpha_bt
            loss = (alpha * model(0, yA, yB, [], da=hp.da,
                                  cutoff_ratio=hp.cutoff_ratio)
                    + hp.alpha_bt * model(1, yA, yB, [], da=hp.da,
                                          cutoff_ratio=hp.cutoff_ratio))
        loss.backward()
        optimizer.step()
        return yA.shape[1]

    model.train()
    ssl_sec, ssl_len = time_steps(ssl_step)

    # -------------------------------------------------- fine-tune step
    ft_batch = max(hp.batch_size // 2, 1)   # what the trainer actually uses
    ft_iter = data.DataLoader(dataset=trainset, batch_size=ft_batch,
                              shuffle=True, num_workers=0,
                              collate_fn=trainset.pad)
    ft_batches = [iter(ft_iter)]
    criterion = torch.nn.CrossEntropyLoss()

    def ft_step():
        try:
            batch = next(ft_batches[0])
        except StopIteration:
            ft_batches[0] = iter(ft_iter)
            batch = next(ft_batches[0])
        optimizer.zero_grad()
        x1, x2, x12, y = batch
        loss = criterion(model(2, x1, x2, x12), y.to(model.device))
        loss.backward()
        optimizer.step()
        return x12.shape[1]

    ft_sec, ft_len = time_steps(ft_step)

    # -------------------------------------------------------- eval rate
    eval_iter = data.DataLoader(dataset=validset, batch_size=hp.batch_size,
                                shuffle=False, num_workers=0,
                                collate_fn=validset.pad)
    model.eval()
    seen, t0 = 0, time.time()
    with torch.no_grad():
        for i, batch in enumerate(eval_iter):
            if i >= EVAL_BATCHES:
                break
            x1, x2, x12, y = batch
            model(2, x1, x2, x12)
            seen += len(y)
    eval_per_pair = (time.time() - t0) / max(seen, 1)
    model.train()

    return dict(task=task, n_train=len(trainset),
                n_eval=len(validset) + len(testset),
                ssl_sec=ssl_sec, ssl_len=ssl_len,
                ft_sec=ft_sec, ft_len=ft_len,
                ft_steps=max(len(trainset) // ft_batch, 1),
                ssl_steps=len(nolabel) // hp.batch_size,
                eval_per_pair=eval_per_pair)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="distilbert", choices=["distilbert", "roberta"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--ssl_method", default="combined",
                    choices=["simclr", "barlow_twins", "combined"])
    ap.add_argument("--tasks", default=None,
                    help="comma-separated; default = every dataset found")
    args = ap.parse_args()

    n_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    torch.set_num_threads(n_threads)
    torch.manual_seed(42)

    hp = hparams(args.lm, args.batch_size, args.max_len, args.ssl_method)
    data_root = os.path.join(SRC, "data", "em")
    if args.tasks:
        tasks = args.tasks.split(",")
    else:
        tasks = sorted(t for t in os.listdir(data_root)
                       if os.path.isfile(os.path.join(data_root, t, "train.txt")))
    if not tasks:
        raise SystemExit("no data under %s -- run sudowoodo_prepare_data.py first"
                         % data_root)

    print("Sudowoodo CPU feasibility check")
    print("  lm=%s ssl_method=%s batch_size=%d max_len=%d threads=%d"
          % (args.lm, args.ssl_method, args.batch_size, args.max_len, n_threads))
    print("  cuda available: %s (expected False)" % torch.cuda.is_available())
    print()

    # The LM does not depend on the data, so it is built once and reused.
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm).to(device)
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    startup = time.time() - t0
    print("  model load: %s (counted as per-config startup)" % fmt(startup))
    print()

    print("  Measured per-step cost, per dataset "
          "(seq = mean padded sequence length):")
    print()
    print("    %-6s %-9s %-11s %-11s" % ("task", "train", "ssl step", "ft step"))
    print("    " + "-" * 44)
    rows = []
    for task in tasks:
        r = measure(task, model, optimizer, hp)
        rows.append(r)
        print("    %-6s %-9d %-11s %-11s"
              % (task, r["n_train"],
                 "%.2fs seq%d" % (r["ssl_sec"], r["ssl_len"]),
                 "%.2fs seq%d" % (r["ft_sec"], r["ft_len"])))
    print()

    print("  Projected wall time per config. Lower bound -- excludes")
    print("  --bootstrap and --clustering, which add an encode pass each.")
    print()
    header = "    %-6s" % "task"
    for n_ssl, n_total in SETTINGS:
        header += " %-11s" % ("(%d,%d)" % (n_ssl, n_total))
    print(header)
    print("    " + "-" * (len(header) - 4))
    for r in rows:
        line = "    %-6s" % r["task"]
        for n_ssl, n_total in SETTINGS:
            n_ft = max(n_total - n_ssl, 0)
            total = (startup
                     + n_ssl * r["ssl_sec"] * r["ssl_steps"]
                     + n_ft * r["ft_sec"] * r["ft_steps"]
                     + n_total * r["eval_per_pair"] * r["n_eval"])
            line += " %-10s%s" % (fmt(total), " " if total < TIME_CAP_SEC else "!")
        print(line)
    print()
    print("    '!' = over the %s cap. Those become TIMEOUT rows, which is a"
          % fmt(TIME_CAP_SEC))
    print("    reportable result -- but a dataset that is '!' in EVERY column")
    print("    cannot be searched at all under the frozen protocol, and that")
    print("    needs raising with the supervisor.")
    print()
    print("    (3,50) is the authors' published setting.")


if __name__ == "__main__":
    main()
