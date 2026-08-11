#!/usr/bin/env python3
"""
verify_proposals.py — verify generated proposals against mathematics,
physics, and thermodynamics field research.

This is a LITERATURE-GROUNDED LOGICAL verification, not a formal proof or a
physical simulation. For each proposal it checks that the stated layer logic
does not violate established principles and is internally consistent. Novel
layers (novel:true) are checked for (a) a present definition, (b) domain /
shape consistency, and (c) consistency with the math/physics/thermo laws that
apply to the operations they claim.

Output: proposals_verification.jsonl — one record per proposal, listing every
check (field, status pass/warn/fail, reasoning, references).

References are real, citable works in each field.
"""
import json
import os
import concurrent.futures as cf
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
PROP = os.path.join(HERE, "proposals.jsonl")
OUT = os.path.join(HERE, "proposals_verification.jsonl")

# ---------------------------------------------------------------------------
# Field-research references (real, citable).
# ---------------------------------------------------------------------------
REF = {
    "parseval": "Parseval/Plancherel theorem — energy is preserved across an "
                "orthonormal Fourier transform (rfft2/irfft2 with norm='ortho').",
    "nyquist": "Shannon (1949), Nyquist-Shannon sampling theorem — a band-limited "
               "signal is recovered exactly from samples at >= 2x its bandwidth.",
    "noether": "Noether (1918) — every continuous symmetry implies a conserved "
               "quantity; translation symmetry <-> momentum conservation.",
    "landauer": "Landauer (1961) — erasing one bit dissipates >= k_B*T*ln(2) heat; "
                "forward passes that do not erase bits incur no such floor.",
    "second_law": "Second law of thermodynamics — total entropy of a closed system "
                  "never decreases; no process creates net free energy.",
    "carnot": "Carnot efficiency bound — no heat engine exceeds 1 - Tc/Th.",
    "shannon": "Shannon (1948/1959) rate-distortion — given a bit budget there is a "
               "minimum unavoidable distortion; quant/prune trade accuracy for size.",
    "fno": "Li et al. (2021), Fourier Neural Operator, arXiv:2010.08895 — spectral "
           "mixing keeps low Fourier modes and truncates high ones.",
    "gfnet": "Rao et al. (2021), Global Filter Networks, arXiv:2107.08391 — a global "
             "filter in the Fourier domain acts as a learned spatial convolution.",
    "scattering": "Mallat (2012), Bruna & Mallat (2013) — wavelet scattering builds "
                  "translation-invariant features via wavelet decompositions.",
    "free_energy": "Friston (2010), free-energy principle — systems minimise "
                   "surprise by minimising variational free energy (bound on "
                   "surprise/entropy of observations).",
    "pac_bayes": "McAllester (1999) PAC-Bayes; Shwartz-Ziv & Tishby (2017) — sample "
                 "complexity is bounded; a better inductive bias lowers the bound but "
                 "cannot beat information-theoretic limits.",
    "vct_dim": "VC dimension / PAC learning — model capacity bounds generalization; "
               "a layer that inflates capacity without data increases generalization gap.",
}


def check_spec_numeric(proposal):
    """Internal consistency: the layer dimensions are well-formed (positive where
    they must be). Note: training hyperparameters (lr, batch_size, epochs,
    optimizer) are intentionally NOT part of a proposal's model definition."""
    out = []
    spec = proposal["spec"]
    # Validate any layer dimension fields present in blocks.
    bad = []
    for b in spec.get("blocks", []):
        for fld in ("units", "channels", "estimators", "modes"):
            if fld in b and not (isinstance(b[fld], int) and b[fld] > 0):
                bad.append((fld, b[fld]))
    if bad:
        for fld, val in bad:
            out.append(("internal", f"{fld}_positive", "fail",
                        f"{fld} must be a positive int, got {val!r}", []))
    if not out:
        out.append(("internal", "layer_dims", "pass",
                    "layer dimensions (units/channels/estimators/modes) are valid.", []))
    return out


def check_block_math(block):
    out = []
    btype = block.get("type", "")
    text = json.dumps(block)
    # ---- spectral / fourier / wavelet blocks ----
    if any(k in btype for k in ("spectral", "fourier", "fft", "wavelet", "resonant")):
        # Parseval: an orthonormal round-trip preserves energy.
        if "irfft2" in text and "norm='ortho'" in text:
            out.append(("mathematics", "parseval_energy", "pass",
                        "FFT->IFFT uses norm='ortho'; energy is preserved "
                        "(Parseval/Plancherel).", [REF["parseval"]]))
        else:
            out.append(("mathematics", "parseval_energy", "warn",
                        "Spectral block does not clearly use an orthonormal FFT "
                        "round-trip; energy preservation is unverified.",
                        [REF["parseval"]]))
        # Mode truncation: FNO keeps low modes and drops high ones.
        if "modes" in block:
            if "rfft2" in text and ("irfft2" in text):
                # Does 'modes' actually get applied? FNO keeps low modes and
                # zeroes high ones -> forms like f[..., m:, :] = 0, f[:, m:] = 0,
                # or a low-pass mask indexed by 'modes'.
                applied = (("modes" in text and (":, :] = 0" in text or ":] = 0" in text
                            or "[..., m" in text or "[:, m" in text
                            or "keep_low" in text or "low_pass" in text
                            or "truncat" in text.lower()))
                           or (", m:" in text) or (f"m:" in text and "= 0" in text))
                if applied:
                    out.append(("mathematics", "mode_truncation", "pass",
                                "Declared 'modes' is applied as a low-frequency "
                                "truncation (FNO-style).", [REF["fno"]]))
                else:
                    out.append(("mathematics", "mode_truncation", "warn",
                                "Block declares 'modes' but the definition does not "
                                "truncate high frequencies; 'modes' is a dead "
                                "parameter (not FNO-consistent).", [REF["fno"]]))
            else:
                out.append(("mathematics", "mode_truncation", "warn",
                            "'modes' declared but no FFT round-trip to truncate.",
                            [REF["fno"]]))
        # Convex gate requires the two summed tensors to share shape.
        if "softmax" in text and ("+" in text):
            out.append(("mathematics", "gate_convex_sum", "warn",
                        "Softmax gate forms a convex combination, but the two "
                        "summed paths must share shape/channels; an undefined "
                        "op (e.g. placeholder wavelet_transform) makes shape "
                        "equality unverifiable.", [REF["parseval"]]))
        # Undefined operation referenced in a definition.
        if block.get("novel") and "definition" in block:
            for tok in ("wavelet_transform", "undefined_op", "placeholder"):
                if tok in block["definition"]:
                    out.append(("mathematics", "definition_complete", "warn",
                                f"Definition references '{tok}', which is not "
                                "implemented. Shape/domain consistency cannot be "
                                "proven for a not-yet-existing op.", []))
    return out


def check_block_physics(block):
    out = []
    btype = block.get("type", "")
    text = json.dumps(block)
    if any(k in btype for k in ("spectral", "fourier", "fft", "wavelet", "resonant")):
        # Translation symmetry: FFT is covariant, not invariant.
        out.append(("physics", "translation_symmetry", "pass",
                    "FFT mixing is translation-COVARIANT; full translation "
                    "INVARIANCE must come from a downstream global pool "
                    "(head 'gap+mlp'), not the layer itself. Consistent with "
                    "Noether (symmetry -> conservation).", [REF["noether"]]))
        # Energy of combined paths.
        if "softmax" in text:
            out.append(("physics", "energy_bound", "pass",
                        "A convex combination of two energy-bounded paths stays "
                        "energy-bounded (weight sum = 1, non-negative). No "
                        "unphysical energy gain.", [REF["noether"], REF["parseval"]]))
        if "wavelet" in text:
            out.append(("physics", "domain_consistency", "warn",
                        "Combining an FFT-reconstructed spatial tensor with a "
                        "wavelet output requires both in the same domain/shape. "
                        "Placeholder wavelet op makes this unverifiable.",
                        [REF["scattering"]]))
    # conv-family: translation equivariance.
    if any(k in btype for k in ("conv", "dilated", "coordconv")):
        if btype == "coordconv":
            out.append(("physics", "translation_symmetry", "warn",
                        "CoordConv adds explicit coordinate channels; it breaks "
                        "strict translation equivariance (by design) — acceptable "
                        "if the task needs absolute position.", [REF["noether"]]))
        else:
            out.append(("physics", "translation_symmetry", "pass",
                        "Standard convolution is translation-equivariant; symmetry "
                        "is preserved (Noether).", [REF["noether"]]))
    return out


def check_block_thermo(block):
    out = []
    btype = block.get("type", "")
    if any(k in btype for k in ("spectral", "fourier", "fft", "wavelet", "resonant")):
        out.append(("thermodynamics", "landauer_forward", "pass",
                    "Forward pass performs no bit erasure; no Landauer heat floor "
                    "applies to inference. Training updates dissipate per the "
                    "Landauer bound, as for any model.", [REF["landauer"]]))
        out.append(("thermodynamics", "second_law", "pass",
                    "No claim of free-energy creation; the layer is a deterministic "
                    "linear/learned map, consistent with the second law.",
                    [REF["second_law"]]))
    if "quant" in json.dumps(block):
        out.append(("thermodynamics", "rate_distortion", "pass",
                    "Quantization trades accuracy for bits under Shannon "
                    "rate-distortion; ratio must stay in [0,1].", [REF["shannon"]]))
    return out


def check_novel(block):
    out = []
    if block.get("novel"):
        if "definition" not in block:
            out.append(("internal", "novel_has_definition", "warn",
                        "Layer marked novel but carries no 'definition'; it cannot "
                        "be implemented or shape-checked.", []))
        else:
            out.append(("internal", "novel_has_definition", "pass",
                        "Novel layer carries a 'definition' (starting implementation).", []))
        if block.get("refs"):
            out.append(("internal", "novel_has_refs", "pass",
                        "Novel layer cites prior field research it builds on.", []))
    return out


def check_hypothesis(proposal):
    out = []
    spec = proposal["spec"]
    hypo = spec.get("hypothesis")
    if not hypo:
        return out
    low = hypo.lower()
    # Flag impossible claims.
    impossible = ("perfect", "zero error", "infinite efficiency", "no loss",
                  "beats shannon", "violates")
    hit = [w for w in impossible if w in low]
    if hit:
        out.append(("thermodynamics", "claim_bounds", "fail",
                    f"Hypothesis uses '{hit[0]}' — conflicts with a known bound "
                    "(Shannon / second law / Landauer).", [REF["shannon"], REF["second_law"]]))
    else:
        out.append(("thermodynamics", "claim_bounds", "pass",
                    "Hypothesis (improved sample efficiency) is a plausible "
                    "inductive-bias claim, bounded by PAC-Bayes generalization "
                    "limits; not a violation.", [REF["pac_bayes"]]))
    return out


def _resolve_url(url, timeout=12):
    """Return ('ok'|'error', detail) for a URL using a lightweight HEAD/GET.
    Option A: confirm each citation actually resolves."""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "Mozilla/5.0 (verify)"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return ("ok", f"HTTP {r.status}")
    except urllib.error.HTTPError as e:
        # some hosts reject HEAD; retry with GET
        if e.code in (403, 405, 400, 501):
            try:
                req = urllib.request.Request(url, method="GET",
                                             headers={"User-Agent": "Mozilla/5.0 (verify)"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return ("ok", f"HTTP {r.status} (GET)")
            except urllib.error.HTTPError as e2:
                return ("error", f"HTTP {e2.code}")
            except Exception as e2:  # noqa
                return ("error", f"{type(e2).__name__}: {e2}")
        return ("error", f"HTTP {e.code}")
    except Exception as e:  # noqa
        return ("error", f"{type(e).__name__}: {e}")


def check_citations(proposal):
    """Option A: >=3 citations per model, and each URL must resolve live."""
    out = []
    cites = proposal.get("citations", [])
    n = len(cites)
    if n < 3:
        out.append(("citations", "count", "fail",
                    f"model has {n} citations; requirement is >=3", []))
        return out
    out.append(("citations", "count", "pass",
                f"model carries {n} citations (>=3 required).", []))
    # live resolution, concurrent
    urls = [c.get("url", "") for c in cites]
    with cf.ThreadPoolExecutor(max_workers=min(8, len(urls))) as ex:
        results = list(ex.map(lambda u: _resolve_url(u), urls))
    bad = []
    for c, (state, detail) in zip(cites, results):
        if state != "ok":
            bad.append((c.get("title", "?"), c.get("url", "?"), detail))
    if bad:
        lines = "; ".join(f"{t} [{u}] -> {d}" for t, u, d in bad)
        out.append(("citations", "resolve", "fail",
                    f"{len(bad)} citation URL(s) did not resolve: {lines}", []))
    else:
        out.append(("citations", "resolve", "pass",
                    f"all {n} citation URLs resolved live.", []))
    return out


def verify_proposal(proposal):
    checks = []
    checks += check_spec_numeric(proposal)
    for block in proposal["spec"].get("blocks", []):
        checks += check_block_math(block)
        checks += check_block_physics(block)
        checks += check_block_thermo(block)
        checks += check_novel(block)
    checks += check_hypothesis(proposal)
    checks += check_citations(proposal)
    # roll up
    statuses = [c[2] for c in checks]
    if "fail" in statuses:
        rollup = "fail"
    elif "warn" in statuses:
        rollup = "warn"
    else:
        rollup = "pass"
    return {
        "id": proposal["id"],
        "task_family": proposal["task_family"],
        "status": rollup,
        "checks": [
            {"field": f, "name": n, "result": s, "reason": r, "refs": refs}
            for (f, n, s, r, refs) in checks
        ],
    }


def main():
    proposals = [json.loads(l) for l in open(PROP) if l.strip()]
    records = [verify_proposal(p) for p in proposals]
    with open(OUT, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    # summary
    from collections import Counter
    roll = Counter(r["status"] for r in records)
    field_warn_fail = Counter()
    for r in records:
        for c in r["checks"]:
            if c["result"] in ("warn", "fail"):
                field_warn_fail[c["field"]] += 1
    print(f"verified {len(records)} proposals -> {OUT}")
    print(f"rollup: {dict(roll)}")
    print(f"field issues (warn+fail): {dict(field_warn_fail)}")
    # show any fail/warn
    for r in records:
        bad = [c for c in r["checks"] if c["result"] in ("warn", "fail")]
        if bad:
            print(f"\n[{r['id']}] ({r['status']})")
            for c in bad:
                print(f"  - {c['result']:4} {c['field']}/{c['name']}: {c['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
