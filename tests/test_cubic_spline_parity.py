"""Parity between the Alpaka kernel and the upstream PyTorch marginal.

The reference is ``ConditionalCubicFlow1D`` in
``conditional-nf-integration/nf/conditional_cubic_1d.py``. Its conditioning MLP
is evaluated in Python and its raw outputs are handed to the kernel, so this
test isolates the spline itself — softmax, cumulative knots, PCHIP slopes, bin
search, Hermite evaluation and the inverse solve.

Both precisions are covered. ``float64`` is the parity reference and holds to
1e-12; ``float32`` is what the pipeline runs and holds to a tolerance derived
from measurement, not assumed — see ``RESULTS.md`` §3.

Run with the built module on the path:

    PYTHONPATH=build/python pytest tests/test_cubic_spline_parity.py -v

Set ``MANGO_UPSTREAM`` to point at the reference repo if it is not at the
default location.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

UPSTREAM = Path(os.environ.get("MANGO_UPSTREAM", "C:/Users/laiji/test/conditional-nf-integration"))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

BUILD_PYTHON = PROJECT / "build" / "python"
if str(BUILD_PYTHON) not in sys.path:
    sys.path.insert(0, str(BUILD_PYTHON))

from nf.conditional_cubic_1d import ConditionalCubicFlow1D  # noqa: E402

mango = pytest.importorskip("mango_alpaka_ext", reason="build the module first: cmake --build build")

N_BINS = 32
COND_DIM = 3
HIDDEN = 128

# float64: differences in operation order and exp/log implementations only.
TOL_F64 = 1e-12
# float32: measured worst case over the model's logit range is 4.8e-7 for x,
# 5.4e-7 for u and 9.5e-6 for the log-determinant. Two orders of margin.
TOL_F32 = 2e-5

DTYPES = [
    pytest.param(torch.float64, np.float64, TOL_F64, id="float64"),
    pytest.param(torch.float32, np.float32, TOL_F32, id="float32"),
]


def make_reference(dtype: torch.dtype, seed: int = 0) -> ConditionalCubicFlow1D:
    torch.manual_seed(seed)
    model = ConditionalCubicFlow1D(cond_dim=COND_DIM, hidden=HIDDEN, n_bins=N_BINS).to(dtype)
    model.eval()
    return model


def raw_and_cond(model, n: int, dtype: torch.dtype, seed: int = 1):
    """Raw MLP outputs (the kernel input) and the conditioning tensor."""
    generator = torch.Generator().manual_seed(seed)
    cond = torch.rand((n, COND_DIM), generator=generator).to(dtype)
    with torch.no_grad():
        raw = model.net(cond)
    return raw.detach().numpy(), cond, raw


@pytest.fixture(scope="module", params=[torch.float64, torch.float32], ids=["float64", "float32"])
def reference(request):
    return make_reference(request.param), request.param


@pytest.mark.parametrize("n", [1, 7, 512, 5_000])
def test_forward_matches_reference(reference, n):
    model, dtype = reference
    tol = TOL_F64 if dtype == torch.float64 else TOL_F32
    raw_np, cond, _ = raw_and_cond(model, n, dtype, seed=n)

    generator = torch.Generator().manual_seed(n + 100)
    u = torch.rand(n, generator=generator).to(dtype)

    with torch.no_grad():
        x_ref, logdet_ref = model.forward(u, cond)

    for backend in mango.backends():
        x, logdet = mango.forward(raw_np, u.numpy(), backend=backend)
        assert x.dtype == raw_np.dtype, "output dtype must match the input dtype"
        np.testing.assert_allclose(x, x_ref.numpy(), atol=tol, rtol=tol,
                                   err_msg=f"x mismatch on {backend}, n={n}, {dtype}")
        np.testing.assert_allclose(logdet, logdet_ref.numpy(), atol=tol, rtol=tol,
                                   err_msg=f"logdet mismatch on {backend}, n={n}, {dtype}")


@pytest.mark.parametrize("n", [1, 7, 512, 5_000])
def test_inverse_matches_reference(reference, n):
    model, dtype = reference
    tol = TOL_F64 if dtype == torch.float64 else TOL_F32
    raw_np, cond, _ = raw_and_cond(model, n, dtype, seed=n + 1)

    generator = torch.Generator().manual_seed(n + 200)
    x = torch.rand(n, generator=generator).to(dtype)

    with torch.no_grad():
        u_ref, logdet_ref = model.inverse(x, cond)

    for backend in mango.backends():
        u, logdet = mango.inverse(raw_np, x.numpy(), backend=backend)
        np.testing.assert_allclose(u, u_ref.numpy(), atol=tol, rtol=tol,
                                   err_msg=f"u mismatch on {backend}, n={n}, {dtype}")
        np.testing.assert_allclose(logdet, logdet_ref.numpy(), atol=tol, rtol=tol,
                                   err_msg=f"inverse logdet mismatch on {backend}, n={n}, {dtype}")


def test_round_trip_is_identity(reference):
    model, dtype = reference
    n = 2_048
    raw_np, cond, _ = raw_and_cond(model, n, dtype, seed=7)
    generator = torch.Generator().manual_seed(11)
    u = torch.rand(n, generator=generator).to(dtype)
    # The bisection has ~2^-40 resolution in t; float32 limits the round trip
    # far earlier than that, so the bound is dtype-dependent.
    tol = 1e-10 if dtype == torch.float64 else 1e-5

    for backend in mango.backends():
        x, _ = mango.forward(raw_np, u.numpy(), backend=backend)
        u_back, _ = mango.inverse(raw_np, x, backend=backend)
        np.testing.assert_allclose(u_back, u.numpy(), atol=tol, rtol=tol,
                                   err_msg=f"round trip failed on {backend}, {dtype}")


def test_log_density_matches_reference(reference):
    model, dtype = reference
    tol = TOL_F64 if dtype == torch.float64 else TOL_F32
    n = 512
    raw_np, cond, _ = raw_and_cond(model, n, dtype, seed=13)
    generator = torch.Generator().manual_seed(17)
    x = torch.rand(n, generator=generator).to(dtype)

    with torch.no_grad():
        log_prob_ref = model.log_prob(x, cond)

    for backend in mango.backends():
        log_prob = mango.log_density(raw_np, x.numpy(), backend=backend)
        np.testing.assert_allclose(log_prob, log_prob_ref.numpy(), atol=tol, rtol=tol,
                                   err_msg=f"log_prob mismatch on {backend}, {dtype}")


def test_contract_sample_and_log_prob(reference):
    """The Brown-facing contract returns (x, log|dx/du|) consistently."""
    model, dtype = reference
    n = 256
    raw_np, _, _ = raw_and_cond(model, n, dtype, seed=19)
    generator = torch.Generator().manual_seed(23)
    u = torch.rand(n, generator=generator).to(dtype).numpy()

    for backend in mango.backends():
        x, logdet = mango.sample_and_log_prob(raw_np, u, backend=backend)
        x_direct = mango.sample(raw_np, u, backend=backend)
        np.testing.assert_array_equal(x, x_direct)
        assert np.all(np.isfinite(logdet))


def test_backends_agree_with_each_other(reference):
    """CPU and GPU must agree to the same tolerance the reference demands."""
    model, dtype = reference
    tol = TOL_F64 if dtype == torch.float64 else TOL_F32
    n = 4_096
    raw_np, _, _ = raw_and_cond(model, n, dtype, seed=29)
    generator = torch.Generator().manual_seed(31)
    u = torch.rand(n, generator=generator).to(dtype).numpy()

    results = {backend: mango.forward(raw_np, u, backend=backend) for backend in mango.backends()}
    names = list(results)
    for other in names[1:]:
        np.testing.assert_allclose(results[other][0], results[names[0]][0], atol=tol, rtol=tol)
        np.testing.assert_allclose(results[other][1], results[names[0]][1], atol=tol, rtol=tol)


def test_dtype_mismatch_raises(reference):
    """Mixing dtypes silently would hide a caller bug."""
    model, dtype = reference
    raw_np, _, _ = raw_and_cond(model, 8, dtype, seed=37)
    other = np.float32 if raw_np.dtype == np.float64 else np.float64
    with pytest.raises(ValueError, match="same dtype"):
        mango.forward(raw_np, np.zeros(8, dtype=other), backend="cuda")
