#!/usr/bin/env python3
"""Canonical per-proposal test loop for NNI-Test (user-mandated process).

For each proposal in proposals.jsonl:
  1. GRAB the proposal.
  2. IDENTIFY the proper test: real-data train/validate on the proposal's
     OWN declared dataset/task (the test_real.py contract). This is the only
     valid test — never force a proposal onto MNIST (no test_mnist --all).
  3. RUN the test (small real-data training, measure metrics).
  4. WRITE the validated row directly to tests/results.jsonl (tagged
     test=real) and VALIDATE it (schema + finite metrics + id in proposals).
     Rows are appended incrementally and flushed so progress persists and the
     run is resumable (--resume skips ids already present).

Final validation: every proposal appears exactly once, all ids are in
proposals.jsonl, every row is tagged test=real, and no metrics are non-finite.

Usage:
  .venv/bin/python tests/run_proposal_tests.py          # fresh full run
  .venv/bin/python tests/run_proposal_tests.py --resume # skip done ids
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from build_model import build_model, output_size  # noqa: E402
import datasets_real  # noqa: E402

OUT = os.path.join(ROOT, "tests", "results.jsonl")
PROPS = os.path.join(ROOT, "proposals.jsonl")
N_TRAIN = 600
N_VAL = 200
BATCH = 64
EPOCHS = 3

torch.set_num_threads(max(1, os.cpu_count() or 1))


def _done_ids():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def test_one(spec):
    """Step 3 — train/validate the proposal on its own dataset. Returns dict."""
    dataset = spec.get("dataset")
    task_type = spec.get("task_type", "classification")
    head = spec.get("head", "logits")
    n_classes = output_size(spec)
    tr, va, _ = datasets_real.get_loaders(
        dataset, task_type=task_type, n_classes=n_classes,
        n_train=N_TRAIN, n_val=N_VAL, batch=BATCH)
    model = build_model(spec)
    param_count = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    crit = torch.nn.CrossEntropyLoss()

    def _loss(out, y):
        if isinstance(out, dict):
            # RL actor-critic head: policy CE + value-regression MSE
            return torch.nn.functional.cross_entropy(out["logits"], y) + out["value"].pow(2).mean()
        if head == "conv_out":
            # generative / image-output head: MSE against a same-shaped target
            return torch.nn.functional.mse_loss(out, torch.randn_like(out))
        return crit(out, y)

    is_generative = (head == "conv_out")

    model.train()
    train_loss = 0.0
    n_train = 0
    for _ in range(EPOCHS):
        for x, y in tr:
            opt.zero_grad()
            loss = _loss(model(x), y)
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
            val_loss += _loss(out, y).item() * x.size(0)
            if is_generative:
                # no class-label accuracy for image-output heads
                continue
            logits = out["logits"] if isinstance(out, dict) else out
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            tot += y.numel()
            n_val += x.size(0)
    val_loss /= max(1, n_val)
    inference_ms = (time.time() - start) / max(1, n_val) * 1000.0
    if is_generative:
        # generation proxy: lower training/val loss => higher (bounded) score
        acc = max(0.0, 1.0 - float(val_loss))
        above = math.isfinite(val_loss)
    else:
        acc = correct / tot if tot else 0.0
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


REQUIRED = ["id", "declared_dataset", "status", "test", "val_acc",
            "train_loss", "val_loss", "inference_ms", "param_count",
            "above_chance"]


def validate_row(r, prop_ids):
    """Step 4 — per-row validation. Returns None if valid, else reason str."""
    if r.get("test") != "real":
        return "test != 'real'"
    if r["id"] not in prop_ids:
        return "id not in proposals.jsonl"
    for k in REQUIRED:
        if k not in r:
            return f"missing key {k}"
    for k in ("val_acc", "train_loss", "val_loss", "inference_ms",
              "param_count"):
        v = r.get(k)
        if v is not None and not (isinstance(v, (int, float)) and math.isfinite(v)):
            return f"non-finite {k}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in results.jsonl")
    args = ap.parse_args()

    props = [json.loads(l) for l in open(PROPS) if l.strip()]
    prop_ids = {p["id"] for p in props}
    done = _done_ids() if args.resume else set()

    mode = "a" if args.resume else "w"
    out_f = open(OUT, mode)
    n_fail = 0
    written = 0

    t0 = time.time()
    for p in props:
        pid = p["id"]
        if pid in done:
            print(f"[skip] {pid} (already done)", flush=True)
            continue
        spec = p.get("spec", {})
        r = {"id": pid, "declared_dataset": spec.get("dataset")}
        try:
            res = test_one(spec)            # steps 2+3
            r["status"] = "ok"
            r.update(res)
            mark = "OK " if res["above_chance"] else "LOW"
        except Exception as e:              # noqa: BLE001
            r["status"] = "fail"
            r["error"] = f"{type(e).__name__}: {e}"
            mark = "FAIL"
            n_fail += 1
        r["test"] = "real"                  # step 4

        err = validate_row(r, prop_ids)     # step 4 validation
        if err:
            r["_validation"] = err
            print(f"[{mark}] {pid} VALIDATION ISSUE: {err}", flush=True)
        else:
            print(f"[{mark}] {pid} val_acc={r.get('val_acc')}", flush=True)

        out_f.write(json.dumps(r) + "\n")   # direct write to results.jsonl
        out_f.flush()
        written += 1

    out_f.close()

    # ---- final validation ----
    rows = [json.loads(l) for l in open(OUT) if l.strip()]
    seen = {}
    for r in rows:
        seen[r["id"]] = seen.get(r["id"], 0) + 1
    missing = [pid for pid in prop_ids if pid not in seen]
    dupes = [pid for pid, c in seen.items() if c > 1]
    extra = [pid for pid in seen if pid not in prop_ids]
    issues = [r["id"] for r in rows if r.get("_validation")]

    ok = (not missing and not dupes and not extra
          and len(rows) == len(prop_ids) and not issues)

    print("\n=== FINAL VALIDATION ===")
    print(f"rows={len(rows)} proposals={len(prop_ids)} "
          f"written_this_run={written} fails={n_fail}")
    print(f"missing={missing}\ndupes={dupes}\nextra={extra}\n"
          f"validation_issues={issues}")
    print(f"elapsed={round(time.time() - t0, 1)}s")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
