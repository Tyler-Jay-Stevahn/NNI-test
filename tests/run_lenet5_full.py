#!/usr/bin/env python3
"""Proper full-MNIST training of the LeNet-5 proposal (Tlenet-mnist-M01).

This is a REAL training run (not the 3-epoch smoke harness): full 60k MNIST
train, evaluated on the full 10k test set. Reports final test accuracy,
train/val loss, and per-epoch progress so the number is meaningful.
"""
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

EPOCHS = 12
BATCH = 128
LR = 0.05
DECAY = 0.5  # lr step decay every 4 epochs
DEVICE = "cpu"


def main():
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    data_dir = os.path.join(ROOT, "data")
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    tr = DataLoader(train, batch_size=BATCH, shuffle=True, drop_last=True)
    te = DataLoader(test, batch_size=256, shuffle=False, drop_last=False)

    # build from the proposal spec (the exact LeNet-5 definition)
    spec = None
    for line in open(os.path.join(ROOT, "proposals.jsonl")):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d["id"] == "Tlenet-mnist-M01":
            spec = d["spec"]
            break
    assert spec is not None, "proposal Tlenet-mnist-M01 not found"

    model = build_model(spec).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=DECAY)
    crit = torch.nn.CrossEntropyLoss()

    print(f"LeNet-5 params: {n_params:,}  device: {DEVICE}")
    print(f"EPOCHS={EPOCHS} BATCH={BATCH} LR={LR} (x{DECAY} every 4 ep)\n")

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        n = 0
        for x, y in tr:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = running / max(1, n)
        sched.step()

        model.eval()
        correct = tot = 0
        with torch.no_grad():
            for x, y in te:
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                tot += y.numel()
        acc = correct / tot
        print(f"ep{ep:02d}  train_loss={train_loss:.4f}  "
              f"test_acc={acc*100:.2f}%  ({time.time()-t0:.1f}s)")

    # final full-test loss + inference timing (schema fields)
    model.eval()
    tloss = 0.0
    tcount = 0
    t0 = time.time()
    with torch.no_grad():
        for x, y in te:
            out = model(x)
            tloss += crit(out, y).item() * x.size(0)
            tcount += x.size(0)
    val_loss = tloss / max(1, tcount)
    inference_ms = (time.time() - t0) / max(1, tcount) * 1000.0

    print(f"\nFINAL test_acc = {acc*100:.2f}%  (n_params={n_params:,})")
    print(f"test_loss = {val_loss:.4f}  inference = {inference_ms:.3f} ms/sample")

    row = {
        "id": "Tlenet-mnist-M01",
        "declared_dataset": "mnist",
        "status": "ok",
        "val_acc": round(float(acc), 4),
        "train_loss": round(float(train_loss), 4),
        "val_loss": round(float(val_loss), 4),
        "inference_ms": round(float(inference_ms), 4),
        "param_count": n_params,
        "above_chance": bool(acc > 1.0 / 10 * 1.5),
    }
    # merge-write into the single results file (replace by id, keep others)
    out_path = os.path.join(ROOT, "tests", "real_results.jsonl")
    rows = []
    if os.path.exists(out_path):
        with open(out_path) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    rows = [r for r in rows if r.get("id") != row["id"]]
    rows.append(row)
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
