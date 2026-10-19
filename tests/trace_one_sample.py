"""Trace one sample through both implementations and record every intermediate.

This is the evidence behind the side-by-side panel in index.html: the same
condition and the same input value, fed to the upstream PyTorch marginal and to
the Alpaka kernel, with the numbers from each stage printed for comparison.

Uses the trained gg_ttbar checkpoint so the raw logits are the ones a real model
produces, not random test weights.

    PYTHONPATH=build/python python tests/trace_one_sample.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

UPSTREAM = Path(os.environ.get("MANGO_UPSTREAM", "C:/Users/laiji/test/conditional-nf-integration"))
for path in (UPSTREAM, PROJECT / "build" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from nf.conditional_cubic_1d import ConditionalCubicFlow1D  # noqa: E402

import mango_alpaka_ext as mango  # noqa: E402

CHECKPOINT = UPSTREAM / "results" / "gg_ttbar_copula_thorough_gpu" / "madgraph_gg_ttbar_2_final_model.pt"
ENERGY = 400.0
MINIMUM, MAXIMUM = 360.0, 2000.0
N_BINS = 32
U = 0.73140000


def load_marginal():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint["marginal_state_dict"]
    hidden = state["marginals.0.net.0.weight"].shape[0]
    model = ConditionalCubicFlow1D(cond_dim=3, hidden=hidden, n_bins=N_BINS).double()
    prefix = "marginals.0."
    model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    model.eval()
    return model


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    model = load_marginal()
    z = (ENERGY - MINIMUM) / (MAXIMUM - MINIMUM)
    cond = torch.tensor([[z, z * z, math.log1p(z) / math.log(2.0)]], dtype=torch.float64)

    with torch.no_grad():
        raw = model.net(cond)
        widths, heights, cumw, cumh = model._params(cond)
        slopes = model._derivatives(widths, heights)
        x_t, logdet_t = model.forward(torch.tensor([U], dtype=torch.float64), cond)
        u_t, inv_logdet_t = model.inverse(x_t, cond)

    u_np = np.array([U], dtype=np.float64)
    x_k, logdet_k = mango.forward(raw.numpy(), u_np, backend="cuda")
    u_k, inv_logdet_k = mango.inverse(raw.numpy(), x_k, backend="cuda")

    # The bin the reference selects, recomputed here for the trace.
    idx = int(min(int((u_np >= cumw[0, 1:].numpy()).sum()), N_BINS - 1))
    t_val = float((U - cumw[0, idx])) / float(widths[0, idx])

    trace = {
        "energy_gev": ENERGY,
        "condition": [float(v) for v in cond[0]],
        "raw_first8": [float(v) for v in raw[0, :8]],
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "u": U,
        "bin_index": idx,
        "width_first4": [float(v) for v in widths[0, :4]],
        "height_first4": [float(v) for v in heights[0, :4]],
        "knots_first5": [float(v) for v in cumw[0, :5]],
        "slopes_first5": [float(v) for v in slopes[0, :5]],
        "selected": {
            "width": float(widths[0, idx]),
            "height": float(heights[0, idx]),
            "u0": float(cumw[0, idx]),
            "x0": float(cumh[0, idx]),
            "slope0": float(slopes[0, idx]),
            "slope1": float(slopes[0, idx + 1]),
            "t": t_val,
        },
        "torch": {"x": float(x_t[0]), "logdet": float(logdet_t[0]),
                  "u_back": float(u_t[0]), "inverse_logdet": float(inv_logdet_t[0])},
        "alpaka": {"x": float(x_k[0]), "logdet": float(logdet_k[0]),
                   "u_back": float(u_k[0]), "inverse_logdet": float(inv_logdet_k[0])},
    }
    trace["difference"] = {
        "x": abs(trace["torch"]["x"] - trace["alpaka"]["x"]),
        "logdet": abs(trace["torch"]["logdet"] - trace["alpaka"]["logdet"]),
        "u_back": abs(trace["torch"]["u_back"] - trace["alpaka"]["u_back"]),
    }

    print(f"energy      {ENERGY:g} GeV  (z = {z:.6f})")
    print(f"raw range   [{trace['raw_min']:.4f}, {trace['raw_max']:.4f}]")
    print(f"input u     {U}")
    print(f"bin         {idx} of {N_BINS}")
    print()
    print(f"{'stage':<22}{'PyTorch':>22}{'Alpaka':>22}{'|diff|':>11}")
    print("-" * 77)
    rows = [
        ("widths[0:3]", widths[0, :3], None),
        ("knots[1:4]", cumw[0, 1:4], None),
        ("slopes[0:3]", slopes[0, :3], None),
        ("x", x_t, x_k),
        ("log|dx/du|", logdet_t, logdet_k),
        ("u recovered", u_t, u_k),
        ("inverse log|dx/du|", inv_logdet_t, inv_logdet_k),
    ]
    for name, left, right in rows:
        l = float(left[0]) if np.ndim(left) == 0 else float(left[0])
        if right is None:
            print(f"{name:<22}{l:>22.12f}{'(same kernel input)':>22}{'':>11}")
        else:
            r = float(right[0])
            print(f"{name:<22}{l:>22.12f}{r:>22.12f}{abs(l - r):>11.2e}")

    out = PROJECT / "results" / "sample_trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
