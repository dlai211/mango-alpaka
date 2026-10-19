"""Parity between the Alpaka kernel and the upstream PyTorch marginal.

The reference is ``ConditionalCubicFlow1D`` in
``conditional-nf-integration/nf/conditional_cubic_1d.py``. Its conditioning MLP
is evaluated in Python and its raw outputs are handed to the kernel, so this
test isolates the spline itself — softmax, cumulative knots, PCHIP slopes, bin
search, Hermite evaluation and the inverse solve.

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

# Absorbing the difference between PyTorch's exp/log and libdevice's is the
# whole point of this threshold. Anything larger indicates a real mismatch.
TOL_FORWARD = 1e-12
TOL_INVERSE = 1e-12
TOL_LOGDET = 1e-12


def make_reference(seed: int = 0):
    torch.manual_seed(seed)
    model = ConditionalCubicFlow1D(cond_dim=COND_DIM, hidden=HIDDEN, n_bins=N_BINS)
    # The kernel computes in double; the module defaults to float.
    model = model.double()
    model.eval()
    return model


def raw_and_cond(model, n: int, seed: int = 1):
    """Raw MLP outputs (the kernel input) and the conditioning tensor."""
    generator = torch.Generator().manual_seed(seed)
    cond = torch.rand((n, COND_DIM), generator=generator, dtype=torch.float64)
    with torch.no_grad():
        raw = model.net(cond)
    return raw.detach().numpy().astype(np.float64), cond, raw


@pytest.fixture(scope="module")
def reference():
    return make_reference()


@pytest.mark.parametrize("n", [1, 7, 512, 5000])
def test_forward_matches_reference(reference, n):
    raw_np, cond, raw_t = raw_and_cond(reference, n, seed=n)

    generator = torch.Generator().manual_seed(n + 100)
    u = torch.rand(n, generator=generator, dtype=torch.float64)

    with torch.no_grad():
        x_ref, logdet_ref = reference.forward(u, cond)

    x_ref = x_ref.numpy()
    logdet_ref = logdet_ref.numpy()

    for backend in mango.backends():
        x, logdet = mango.forward(raw_np, u.numpy(), backend=backend)
        np.testing.assert_allclose(x, x_ref, atol=TOL_FORWARD, rtol=TOL_FORWARD,
                                   err_msg=f"x mismatch on {backend}, n={n}")
        np.testing.assert_allclose(logdet, logdet_ref, atol=TOL_LOGDET, rtol=TOL_LOGDET,
                                   err_msg=f"logdet mismatch on {backend}, n={n}")


@pytest.mark.parametrize("n", [1, 7, 512, 5000])
def test_inverse_matches_reference(reference, n):
    raw_np, cond, _ = raw_and_cond(reference, n, seed=n + 1)

    generator = torch.Generator().manual_seed(n + 200)
    x = torch.rand(n, generator=generator, dtype=torch.float64)

    with torch.no_grad():
        u_ref, logdet_ref = reference.inverse(x, cond)

    u_ref = u_ref.numpy()
    logdet_ref = logdet_ref.numpy()

    for backend in mango.backends():
        u, logdet = mango.inverse(raw_np, x.numpy(), backend=backend)
        np.testing.assert_allclose(u, u_ref, atol=TOL_INVERSE, rtol=TOL_INVERSE,
                                   err_msg=f"u mismatch on {backend}, n={n}")
        np.testing.assert_allclose(logdet, logdet_ref, atol=TOL_LOGDET, rtol=TOL_LOGDET,
                                   err_msg=f"inverse logdet mismatch on {backend}, n={n}")


def test_round_trip_is_identity(reference):
    """x -> u -> x must return the original value."""
    n = 2048
    raw_np, cond, _ = raw_and_cond(reference, n, seed=7)

    generator = torch.Generator().manual_seed(11)
    u = torch.rand(n, generator=generator, dtype=torch.float64)

    for backend in mango.backends():
        x, _ = mango.forward(raw_np, u.numpy(), backend=backend)
        u_back, _ = mango.inverse(raw_np, x, backend=backend)
        np.testing.assert_allclose(u_back, u.numpy(), atol=1e-10, rtol=1e-10,
                                   err_msg=f"round trip failed on {backend}")


def test_log_density_matches_reference(reference):
    n = 512
    raw_np, cond, _ = raw_and_cond(reference, n, seed=13)

    generator = torch.Generator().manual_seed(17)
    x = torch.rand(n, generator=generator, dtype=torch.float64)

    with torch.no_grad():
        log_prob_ref = reference.log_prob(x, cond)

    for backend in mango.backends():
        log_prob = mango.log_density(raw_np, x.numpy(), backend=backend)
        np.testing.assert_allclose(log_prob, log_prob_ref.numpy(),
                                   atol=TOL_LOGDET, rtol=TOL_LOGDET,
                                   err_msg=f"log_prob mismatch on {backend}")


def test_contract_sample_and_log_prob(reference):
    """The Brown-facing contract returns (x, log|dx/du|) consistently."""
    n = 256
    raw_np, _, _ = raw_and_cond(reference, n, seed=19)

    generator = torch.Generator().manual_seed(23)
    u = torch.rand(n, generator=generator, dtype=torch.float64).numpy()

    for backend in mango.backends():
        x, logdet = mango.sample_and_log_prob(raw_np, u, backend=backend)
        x_direct = mango.sample(raw_np, u, backend=backend)
        np.testing.assert_array_equal(x, x_direct)
        assert np.all(np.isfinite(logdet))


def test_backends_agree_with_each_other(reference):
    """CPU and GPU must agree to the same tolerance the reference demands."""
    n = 4096
    raw_np, _, _ = raw_and_cond(reference, n, seed=29)
    generator = torch.Generator().manual_seed(31)
    u = torch.rand(n, generator=generator, dtype=torch.float64).numpy()

    results = {backend: mango.forward(raw_np, u, backend=backend) for backend in mango.backends()}
    names = list(results)
    base_name = names[0]
    for other in names[1:]:
        np.testing.assert_allclose(results[other][0], results[base_name][0],
                                   atol=TOL_FORWARD, rtol=TOL_FORWARD)
        np.testing.assert_allclose(results[other][1], results[base_name][1],
                                   atol=TOL_LOGDET, rtol=TOL_LOGDET)
