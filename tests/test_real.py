#!/usr/bin/env python3
"""test_real.py — real-data training for NNI-Test proposals.

Generalizes tests/test_mnist.py: for each targeted proposal it builds the model
from its spec, trains a few real epochs on the proposal's declared dataset
(via pipeline/datasets_real.py), and records accuracy / losses / inference time.

Results go to tests/real_results.jsonl with the same schema as
mnist_results.jsonl so the dashboard can render them uniformly:

    id, declared_dataset, status, val_acc, train_loss, val_loss,
    inference_ms, param_count, above_chance

Usage:
    .venv/bin/python tests/test_real.py            # only status=='tested'
    .venv/bin/python tests/test_real.py --all      # every proposal
    .venv/bin/python tests/test_real.py --id Tnas-cifar-M05
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from build_model import build_model, output_size  # noqa: E402
import datasets_real  # noqa: E402

N_TRAIN = 600
N_VAL = 200
BATCH = 64
EPOCHS = 3


def test_one(spec):
    """Build + train + validate on the proposal's real dataset.

    Returns the measurement dict (or raises)."""
    dataset = spec.get("dataset")
    clustering = spec.get("task_type") == "clustering"
    n_classes = output_size(spec)
    tr, va, nc = datasets_real.get_loaders(
        dataset, task_type=spec.get("task_type"), n_classes=n_classes,
        n_train=N_TRAIN, n_val=N_VAL, batch=BATCH)

    model = build_model(spec)
    param_count = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    crit = torch.nn.CrossEntropyLoss()

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

    # validation / inference timing
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
    acc = correct / tot if tot else 0.0
    # chance baseline: 1/n_classes (random), margin 1.5x
    chance = 1.0 / max(1, n_classes)
    above = acc > chance * 1.5
    return {
        "val_acc": round(acc, 4),
        "train_loss": round(float(train_loss), 4),
        "val_loss": round(float(val_loss), 4),
        "inference_ms": round(float(inference_ms), 4),
        "param_count": param_count,
        "above_chance": above,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="test one proposal id")
    ap.add_argument("--all", action="store_true", help="test every proposal")
    args = ap.parse_args()

    props = [json.loads(l) for l in open(os.path.join(ROOT, "proposals.jsonl")) if l.strip()]
    if args.id:
        targets = [p for p in props if p["id"] == args.id]
    elif args.all:
        targets = props
    else:
        targets = [p for p in props if p.get("status") == "tested"]

    print(f"real-data test: {len(targets)} proposal(s)\n")
    results = []
    for p in targets:
        pid = p["id"]
        spec = p.get("spec", {})
        r = {"id": pid, "declared_dataset": spec.get("dataset")}
        try:
            res = test_one(spec)
            r["status"] = "ok"
            r.update(res)
            mark = "OK " if res["above_chance"] else "LOW"
        except Exception as e:  # noqa
            r["status"] = "fail"
            r["error"] = f"{type(e).__name__}: {e}"
            mark = "FAIL"
        results.append(r)
        acc_s = f"{r.get('val_acc'):.4f}" if "val_acc" in r else "-"
        print(f"[{mark}] {pid:24s} val_acc={acc_s}  (dataset={r['declared_dataset']})")

    out = os.path.join(ROOT, "tests", "real_results.jsonl")
    with open(out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    learned = sum(1 for r in results if r.get("above_chance"))
    failed = sum(1 for r in results if r["status"] == "fail")
    print(f"\n{learned}/{len(results)} learned above chance (x1.5 random); {failed} failed to build/run.")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
