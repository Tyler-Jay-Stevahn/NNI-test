# MISSION — Learning Microsoft NNI (Neural Network Intelligence)

## Why this matters to me (Tyler)
I am an undergrad (Psychology + Philosophy minor, moving toward Industrial
Engineering) with a hands-on local-AI/ML hobby. I run a headless PowerEdge R530
with an NVIDIA Tesla M40 and use local models. I got curious about NNI as an
AutoML toolkit and had been experimentally building a proposal sweep over its
components (the `proposals.jsonl` in this folder is that artifact — a set of
model architecture proposals across HPO, NAS, compression, quantization,
pruning, feature engineering, adversarial robustness, plus audio / text /
time-series / clustering modalities).

The count is **not fixed**: `gen_proposals.py` emits `MODELS_PER_FAMILY`
variants (widths 32/64/128/256/512) × `len(TASK_FAMILIES)`. New families are
added over time (audio, text, time-series, and clustering families were added
2026-08-11), so the proposal count grows — check `wc -l proposals.jsonl` or the
dashboard Overview for the live number. As of the last run it was
`len(TASK_FAMILIES) × 5` proposals.

I want to *understand* NNI properly — what it is, what it does, what's worth
using today, and what its limits are — rather than just run scripts at it.

## Critical context (verified 2026-08-10)
- NNI is **ARCHIVED** on GitHub (Microsoft stopped active maintenance). Stars
  ~14.4k. It still works, but no new features/fixes. Any learning must account
  for this: it is a reference/legacy toolkit, not a live one.
- NNI covers four pillars: **Hyperparameter Tuning (HPO)**, **Neural
  Architecture Search (NAS)**, **Model Compression** (pruning + quantization),
  and **Feature Engineering** (for tabular). It also has **Assessors** (early
  stopping) and **Advisors**.

## Success looks like
- I can explain NNI's four pillars and name the main algorithms in each.
- I can decide *when* NNI is the right tool vs. modern alternatives (Optuna,
  Ray Tune, AutoKeras, or just writing the loop myself).
- I understand the archived status and its practical implications for my
  local-R530 hobby use.

## Non-goals (for now)
- Not re-running a giant sweep — the earlier "too wide" idea was scrapped.
  Learning > brute-force experimentation.
