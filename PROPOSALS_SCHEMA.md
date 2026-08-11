# proposals.jsonl — schema

One JSON object per line. Each line is one proposed model. The format is
intentionally OPEN: it can describe any layer, including layers and whole
models that do not exist yet.

## Top-level fields

- `id` — unique id, format `T<task_family>-M<nn>` (example: `Tadv-robust-M01`).
- `task` — task tag (same value as `task_family`).
- `task_family` — which domain the proposal belongs to.
- `status` — lifecycle: `proposed`, `approved`, or `tested`.
- `created` — ISO date the proposal was authored.
- `parent` — id of the proposal this variant came from (`null` for the root);
  used for NAS / lineage tracking.
- `rationale` — why this model and this layer-width variant was proposed.
- `citations` — list of >=3 real references (title, url, why). Every URL is
  live-checked by `verify_proposals.py`.
- `compile_status` — `untested` when generated; set to `ok` or `fail` by
  `pipeline/smoke_test.py` after the train/test/predict compile gate runs.
- `spec` — nested architecture only (see below). **No training hyperparameters**
  (no lr / weight_decay / batch_size / epochs / optimizer / augmentation) — those
  belong to the training run, not the model definition.
- `expected` — open slot for results: `held_out_acc` (null until tested) and
  `note`.

## Invariants enforced by gen_proposals.py

- **Uniqueness (by layer dimensions)**: a proposal is a DUPLICATE only when it
  repeats the SAME task AND the SAME layer dimensions (architecture + widths).
  The same architecture is explicitly allowed for a different task OR with
  different layer sizes (e.g. a dense layer of 32 vs 64). The check keys on
  `(task_family, arch_fingerprint)`.
- **Distinct variants by width**: within a family, M01..M05 use layer widths
  32 / 64 / 128 / 256 / 512, so they are separate models distinguished by their
  LAYER DIMENSIONS, not by any training setting.
- **Citations**: every task family carries >=3 curated, real references.
- **No training HPs in the list**: proposals describe models (architecture +
  layer sizes), never training configurations.
- **Citations**: every task family carries >=3 curated, real references.

## Folder layout

- `gen_proposals.py` — generator (emits `proposals.jsonl`, enforces uniqueness).
- `verify_proposals.py` — math/physics/thermo + live-citation checks.
- `pipeline/` — synthetic compile gate (`smoke_test.py`) + model builder.
- `tests/` — real-data tests, e.g. `test_mnist.py` (downloads MNIST, writes
  `tests/mnist_results.jsonl`).
- `data/` — downloaded datasets (git-ignored; `.gitignore` excludes `.data/`).
- `PROPOSALS_SCHEMA.md`, `VERIFICATION.md` — docs.

## `spec` fields

- `model` — architecture family name (free string).
- `dataset` — dataset name (free string).
- `blocks` — LIST of layer dicts. **Each layer dict is free-form.** The only
  required key is `type` (any string, known or invented). Any other key is
  allowed. Optional per-layer keys used by this repo:
  - `novel` — true if the layer does not exist in published form.
  - `definition` — literal (hypothetical) code for the layer.
  - `refs` — list of paper/URL references the layer builds on.
  - `note` — free text.
- `head` — output head (logits, gap+mlp, regression, …).
- Optional family knobs: `quant`, `prune`, `feature_search`.
- Optional OPEN-MODEL keys:
  - `novel` — true if the whole model does not exist yet.
  - `hypothesis` — what the novel model is meant to achieve.
  - `implementation_status` — `designed` / `prototyped` / `implemented`.

> **Training hyperparameters are NOT part of a `spec`.** `lr`, `optimizer`,
> `weight_decay`, `batch_size`, `epochs`, and `augmentation` belong to a
> *training run*, not to the model definition, and `gen_proposals.py` does not
> emit them. They are intentionally absent from `proposals.jsonl`.

## Open contract (important)

Nothing in this format restricts `type` to a fixed vocabulary. A training
script / model builder downstream must map each `type` string to real code.
For known types that mapping is ordinary. For a `novel` layer the mapping must
be written (the `definition` field can carry a starting implementation). NNI's
Retiarii NAS and mutator API support declaring and searching over custom
operators, so custom and not-yet-existing layers are in scope.
