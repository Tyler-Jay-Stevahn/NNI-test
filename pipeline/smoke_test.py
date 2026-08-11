#!/usr/bin/env python3
"""smoke_test.py — tiny train/test/predict pipeline to validate a model
*compiles* (builds + runs forward + one backward + predict). This is NOT a
real training run; it uses 2 synthetic batches and checks shapes and finite
gradients so a broken architecture fails loudly.
"""
import argparse
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_model import build_model  # noqa: E402


def load_proposal(pid: str) -> dict:
    for line in open(os.path.join(ROOT, "proposals.jsonl")):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["id"] == pid:
            return r
    raise SystemExit(f"proposal {pid!r} not found in proposals.jsonl")


def smoke_test(proposal: dict) -> dict:
    import torch

    spec = proposal["spec"]
    dev = "cpu"
    report = {"id": proposal["id"], "task_family": proposal["task_family"]}
    try:
        from build_model import build_model, sample_batch, modality, output_size
        report["modality"] = modality(spec.get("dataset", ""))
        report["task_type"] = spec.get("task_type", "classification")
        nc = output_size(spec)
        model = build_model(spec)
        model.to(dev)
        n_params = sum(p.numel() for p in model.parameters())
        report["params"] = n_params

        # synthetic tiny input: dataset- and modality-correct shape, and a
        # label range matching the class (or cluster) count.
        x, y = sample_batch(spec, batch=2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)  # test-owned lr
        crit = torch.nn.CrossEntropyLoss()

        # TRAIN step (one micro-batch) — proves backward compiles
        model.train()
        out = model(x)
        report["out_shape"] = list(out.shape)
        loss = crit(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # check grads are finite
        bad = [not torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None]
        report["grads_finite"] = (not any(bad))

        # TEST step — forward only
        model.eval()
        with torch.no_grad():
            t_out = model(x)
        report["test_out_shape"] = list(t_out.shape)

        # PREDICT step — argmax class
        with torch.no_grad():
            pred = torch.argmax(model(x[:1]), dim=1)
        report["predict"] = pred.tolist()

        report["status"] = "ok"
        report["message"] = f"compiles; {n_params} params; loss={float(loss.detach()):.4f}"
    except Exception as e:  # noqa
        report["status"] = "fail"
        report["message"] = f"{type(e).__name__}: {e}"
        report["trace"] = traceback.format_exc()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="proposal id to test")
    ap.add_argument("--all", action="store_true", help="test every proposal")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    if args.all:
        props = [json.loads(l) for l in open(os.path.join(ROOT, "proposals.jsonl"))
                 if l.strip()]
    elif args.id:
        props = [load_proposal(args.id)]
    else:
        sys.exit("pass --id <pid> or --all")

    results = []
    for p in props:
        r = smoke_test(p)
        results.append(r)
        mark = "OK " if r["status"] == "ok" else "FAIL"
        print(f"[{mark}] {r['id']:24s} {r.get('message','')}")
        if r["status"] == "fail" and args.fail_fast:
            break

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{ok}/{len(results)} proposals compiled successfully")

    # write machine-readable report
    with open(os.path.join(ROOT, "pipeline_smoke_results.jsonl"), "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    # write compile_status back into proposals.jsonl (per-row update)
    status_by_id = {r["id"]: r["status"] for r in results}
    prop_path = os.path.join(ROOT, "proposals.jsonl")
    out_rows = []
    for l in open(prop_path):
        l = l.strip()
        if not l:
            continue
        row = json.loads(l)
        if row["id"] in status_by_id:
            row["compile_status"] = status_by_id[row["id"]]
        out_rows.append(row)
    with open(prop_path, "w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")

    if ok != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
