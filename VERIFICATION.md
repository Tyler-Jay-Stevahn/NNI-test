# Proposal verification workflow

Generate, then verify. Every proposal is checked against mathematics,
physics, and thermodynamics field research before it is trusted.

## Steps

    python3 gen_proposals.py        # write proposals.jsonl (one per variant)
    python3 verify_proposals.py     # write proposals_verification.jsonl

## What verify_proposals.py checks

It is a LITERATURE-GROUNDED LOGICAL verification, not a formal proof or a
physical simulation. For each proposal it runs checks in four fields:

- `internal`      layer-dimension validation (units/channels/estimators/modes are
                  positive ints); training hyperparameters are intentionally NOT
                  part of a proposal's model definition.
                  novel layer must carry a `definition` and `refs`.
- `mathematics`   Parseval/Plancherel energy preservation across orthonormal
                  FFT round-trips; FNO-style low-frequency mode truncation;
                  convex-gate shape consistency.
- `physics`       translation equivariance/invariance vs Noether symmetry;
                  energy bounds of combined paths; cross-domain shape
                  consistency (e.g. FFT + wavelet).
- `thermodynamics` Landauer bound on forward pass (no bit erasure -> no heat
                  floor); second law (no free-energy creation); Shannon
                  rate-distortion for quant/prune; PAC-Bayes bounds on
                  inductive-bias claims.

Each check records: field, name, result (pass/warn/fail), reasoning, and
real citations (Parseval, Nyquist, Noether, Landauer, FNO, GFNet, scattering,
PAC-Bayes, Shannon rate-distortion, …).

## Reading the result

- `pass` — consistent with the relevant principle.
- `warn` — cannot be proven from the proposal alone (e.g. a genuinely novel
  op that is not yet implemented, or a deliberate design break like CoordConv).
- `fail` — conflicts with a known bound (e.g. a "perfect / zero error" claim).

A `warn` on a `novel:true` layer is expected: the layer does not exist yet, so
its shape/domain consistency is provable only once `definition` is completed.

## Recent finding (loop closed)

Verification caught a real logic error in the generated novel layer:
`resonant_spectral_mix` declared `modes` but the definition never truncated
high frequencies — `modes` was a dead parameter (not FNO-consistent). Fixed by
adding an explicit low-frequency zeroing (`f[..., m:, :] = 0.0`), after which
the check passes.

## Compile smoke test (separate, real execution)

`pipeline/` holds a tiny train/test/predict harness that proves each proposed
model actually *builds and runs*, not just that the JSON is well-formed:

- `pipeline/build_model.py` — turns a `spec` into a torch model. Known layer
  `type`s are hand-built; layers marked `novel` (with a `definition`) are
  exec'd in a controlled namespace so a genuinely new layer becomes runnable.
  A shape-preserving `wavelet_transform` stub is injected so novel layers that
  name a not-yet-written op still compile.
- `pipeline/smoke_test.py` — for each proposal: builds the model, runs ONE
  forward + ONE backward + a predict on 2 synthetic 3x16x16 batches, and checks
  output shape and finite gradients. Writes `pipeline_smoke_results.jsonl`.
- `pipeline/run.sh` — gen + verify + smoke-test in one pass.

Output is `N/N proposals compiled successfully`. This caught real bugs that the
JSON/principles checks could not: chained conv blocks with a hard-coded input
channel of 3 (crashed on the 2nd block), and novel layers receiving
descriptive kwargs their constructor didn't accept.

Run it:
    bash pipeline/run.sh
or, for one model:
    .venv/bin/python pipeline/smoke_test.py --id Tadv-robust-M01

Note: this is a COMPILE gate, not training. It uses 2 synthetic batches and
asserts the graph + backward + predict run and gradients are finite. It does
not measure accuracy or convergence.

## Real-data MNIST test (tests/test_mnist.py)

Separate from the synthetic compile gate, `tests/test_mnist.py` runs a small
amount of REAL labeled training on MNIST to confirm a proposal not only builds
but learns above chance. It downloads MNIST into `data/` (git-ignored) and
writes `tests/mnist_results.jsonl`.

Two modes:
- Default: tests only proposals whose `spec.dataset == "mnist"` (the
  `hpo-mnist` family).
- `--all`: attempts EVERY proposal on MNIST input (1x28x28, 10 classes) to
  answer "which of these architectures actually run on MNIST?" Models declared
  for 3-channel datasets score at/below chance on MNIST's single channel — the
  test makes that explicit instead of hiding it.

A proposal "works for MNIST" when it both builds for 1x28x28 input AND reaches
val accuracy above the 0.15 threshold (chance = 0.10) after 3 short epochs on
2k samples.

Run it:
    .venv/bin/python tests/test_mnist.py            # declared-for-MNIST only
    .venv/bin/python tests/test_mnist.py --all      # every proposal on MNIST
    .venv/bin/python tests/test_mnist.py --id Thpo-mnist-M01
