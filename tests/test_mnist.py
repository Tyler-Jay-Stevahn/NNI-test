#!/usr/bin/env python3
"""test_mnist.py — real-data MNIST validation for proposals.

Unlike the synthetic compile gate (pipeline/smoke_test.py), this runs a small
amount of REAL labeled training on MNIST to confirm a proposal not only builds
but learns above chance-level.

By default it tests only proposals whose spec declares dataset == 'mnist'.
With --all it attempts EVERY proposal on MNIST input (1x28x28, 10 classes) to
answer "which of these architectures run on MNIST?".

Results are written to tests/mnist_results.jsonl.

Usage:
    .venv/bin/python tests/test_mnist.py            # only dataset=='mnist'
    .venv/bin/python tests/test_mnist.py --all      # every proposal on MNIST
    .venv/bin/python tests/test_mnist.py --id Thpo-mnist-M01
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from build_model import build_model  # noqa: E402

DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

N_TRAIN = 2000
N_VAL = 1000
BATCH = 64
EPOCHS = 3
CHANCE = 0.10          # 10-class random baseline
THRESHOLD = 0.15       # "learns" if it beats chance by a margin


def _loaders():
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(DATA, train=True, download=True, transform=tfm)
    val = datasets.MNIST(DATA, train=False, download=True, transform=tfm)
    tr = DataLoader(Subset(train, list(range(N_TRAIN))),
                    batch_size=BATCH, shuffle=True)
    va = DataLoader(Subset(val, list(range(N_VAL))),
                    batch_size=BATCH, shuffle=False)
    return tr, va


def test_one(spec):
    """Build for MNIST (1x28x28, 10-class) and train a few real batches.

    Returns a dict of per-variant measurements so the dashboard can plot more
    than just accuracy: final train/val loss, per-sample inference time, and
    the total parameter count.
    """
    spec_mnist = {**spec, "dataset": "mnist"}   # we are testing ON mnist
    model = build_model(spec_mnist)
    param_count = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)  # test-owned; not in proposal
    crit = torch.nn.CrossEntropyLoss()
    tr, va = _loaders()
    model.train()
    train_loss = 0.0
    n_train = 0
    for _ in range(EPOCHS):
        for x, y in tr:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
            n_train += x.size(0)
    train_loss /= max(1, n_train)
    model.eval()
    correct = tot = n_val = 0
    val_loss = 0.0
    start = time.time()
    with torch.no_grad():
        for x, y in va:
            out = model(x)
            val_loss += crit(out, y).item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            tot += y.numel()
            n_val += x.size(0)
    val_loss /= max(1, n_val)
    inference_ms = (time.time() - start) / max(1, n_val) * 1000.0
    acc = correct / tot
    return {
        "val_acc": round(acc, 4),
        "train_loss": round(float(train_loss), 4),
        "val_loss": round(float(val_loss), 4),
        "inference_ms": round(float(inference_ms), 4),
        "param_count": param_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="test one proposal id on MNIST")
    ap.add_argument("--all", action="store_true",
                    help="attempt every proposal on MNIST")
    args = ap.parse_args()

    props = [json.loads(l) for l in open(os.path.join(ROOT, "proposals.jsonl"))
             if l.strip()]
    mnist_props = [p for p in props if p["spec"].get("dataset") == "mnist"]

    if args.id:
        targets = [p for p in props if p["id"] == args.id]
    elif args.all:
        targets = props
    else:
        targets = mnist_props

    print(f"MNIST test: {len(targets)} proposal(s) "
          f"({len(mnist_props)} declared for MNIST out of {len(props)} total)\n")

    results = []
    for p in targets:
        r = {"id": p["id"], "declared_dataset": p["spec"].get("dataset")}
        try:
            res = test_one(p["spec"])
            r["status"] = "ok"
            r["val_acc"] = res["val_acc"]
            r["train_loss"] = res["train_loss"]
            r["val_loss"] = res["val_loss"]
            r["inference_ms"] = res["inference_ms"]
            r["param_count"] = res["param_count"]
            r["above_chance"] = res["val_acc"] > THRESHOLD
            mark = "OK " if res["val_acc"] > THRESHOLD else "LOW"
        except Exception as e:  # noqa
            r["status"] = "fail"
            r["error"] = f"{type(e).__name__}: {e}"
            mark = "FAIL"
        results.append(r)
        acc_s = f"{r.get('val_acc'):.4f}" if "val_acc" in r else "-"
        print(f"[{mark}] {r['id']:24s} val_acc={acc_s}  "
              f"(declared: {r['declared_dataset']})")

    with open(os.path.join(ROOT, "tests", "mnist_results.jsonl"), "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    learned = sum(1 for r in results if r.get("above_chance"))
    print(f"\n{learned}/{len(results)} ran on MNIST and learned above chance "
          f"(>{THRESHOLD}); {sum(1 for r in results if r['status']=='fail')} failed to build/run.")


if __name__ == "__main__":
    main()
