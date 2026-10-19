# Intro — what this repository is and why

A one-sitting orientation. Read this first; `PRD.md` has the requirements,
`RUNBOOK.md` has the build, `RESULTS.md` has the numbers.

---

## 1. The problem MANGO is solving

Monte Carlo integration for particle physics has a computing problem. The
HL-LHC needs roughly **10× more simulated events** than the current CPU-based
stack can deliver in the same budget.

Classical importance sampling (VEGAS) is inherently serial: each sampled point
depends on grid updates computed from earlier points. That defeats the SIMD
parallelism GPUs rely on, so throwing GPUs at VEGAS does not work.

MANGO's answer is to **learn the proposal distribution** as a conditional
normalizing flow, evaluate it natively on GPUs via Alpaka, and keep the
estimator exact.

## 2. Why an exact estimator matters

Importance sampling says

```
I = ∫ f(x) dx = E_{r}[ f(x)/r(x) ]        Î = (1/N) Σ f(xₙ)/r(xₙ)
```

This is **unbiased for any proposal r that is positive wherever f is**. A bad
proposal costs variance, never correctness. Quality is measured by the
effective sample size,

```
ESS/N = (Σw)² / (N Σw²)          wₙ = f(xₙ)/r(xₙ)
```

So the flow never has to be *right*; it only has to concentrate samples where
`f` is large. That is what makes a learned proposal safe to deploy.

The flow is **conditional on the collision energy**, so one model covers the
whole energy range instead of one model per point.

## 3. The factorisation this repository touches

The upstream pipeline splits the conditional density into marginals and a
copula:

```
r(x | c) = [ Π_i marginal_i(x_i | c) ] × copula(rank x | c)
```

The **marginal** is a monotone map of `[0,1]` to itself, one per dimension, and
it is what this repository implements. The copula captures the correlations
between dimensions and stays in Python for now.

### The marginal, in four steps

Given a conditioning vector `c`, a small MLP produces `2·n_bins` raw numbers.
Everything after that happens inside the kernel, per element:

1. **Bins.** Split the raw numbers into width and height logits. A scaled
   softmax turns each into `n_bins` positive bin sizes summing to ~1:

   ```
   width[i] = min_bin_width + (1 - n_bins·min_bin_width) · softmax(logit_w)[i]
   ```

   The cumulative sums give the knots, with the final knot forced to exactly 1
   so softmax roundoff does not accumulate at the boundary.

2. **Slopes.** A PCHIP construction gives the derivative at each knot from the
   neighbouring secants. Positive secants produce positive slopes, which is
   what makes the map monotone and therefore invertible.

3. **Forward.** Find the bin containing `u`, normalise to `t ∈ [0,1]` within the
   bin, and evaluate the cubic Hermite polynomial. Its derivative gives
   `log|dx/du|` directly.

4. **Inverse.** There is no closed form, so the reference does 40 bisection
   iterations on `t`, then one Newton correction. This is the expensive part —
   and in PyTorch it is 40 rounds of small kernel launches.

## 4. Why Alpaka, and why it wins here

Alpaka is a header-only C++17 abstraction that compiles the same kernel for
CUDA, HIP, SYCL and CPUs. For MANGO that is a strategic requirement: the
project must not bet on one vendor.

The measured reason it wins on this workload is **not arithmetic**. The PyTorch
marginal costs the same at batch size 1 and batch size 10 000:

| operation, CUDA, dim=2 | N=1 | N=10 000 |
|---|---:|---:|
| forward | 3.68 ms | 3.70 ms |
| inverse | 69.8 ms | 68.7 ms |

Those times are **launch overhead**, not compute. The Alpaka kernel evaluates
each element end to end in a single launch, which is why the inverse is about
**130× faster at N=1** — and only **2.9× faster at N=50 000**, where the
arithmetic finally matters more than the launches.

The forward is a different story: above N ≈ 1 000 the kernel is *slower* than
PyTorch (0.4× at N=10 000), because it computes in `float64` (1/32 rate on a
consumer GPU) and copies per call. Making it competitive is the next kernel
task. The honest summary of Deliverable B is therefore: **validated for the
inverse, not yet for the forward.**

## 5. Repository map

| path | what it is |
|---|---|
| `include/mango_interface.hpp` | The frozen contract. No Alpaka, no pybind11 — Brown can include it with a plain C++17 compiler. |
| `include/kernels/cubic_spline_1d.hpp` | The Alpaka kernel: one thread per element, everything fused. |
| `src/launcher.cu` | Instantiates the kernel per backend. Compiled by nvcc. |
| `src/bindings.cpp` | pybind11 module `mango_alpaka_ext`. Compiled by MSVC, never sees Alpaka. |
| `tests/test_cubic_spline_parity.py` | Kernel vs PyTorch at 1e-12. |
| `tests/benchmark_fixed_cost.py` | The pre-kernel baseline. |
| `tests/benchmark_kernel_vs_reference.py` | The before/after. |
| `docs/INTERFACE_CONTRACT.md` | What Brown and BNL code against. |
| `third_party/alpaka-1.3.0/` | Not committed; fetched by `scripts/fetch_dependencies.sh`. |

## 6. Five-minute start

```cmd
scripts\fetch_dependencies.sh          :: or clone alpaka 1.3.0 by hand
conda install -n <env> -c conda-forge boost-cpp
build.bat
set PYTHONPATH=build\python
python -m pytest tests\test_cubic_spline_parity.py -q
```

Then:

```python
import mango_alpaka_ext as mango
print(mango.backends())                              # ['cpu_serial', 'cpu_omp2_blocks', 'cuda']
x, logdet = mango.forward(raw, u, backend="cuda")    # raw: (n, 2*n_bins) float64
```

`raw` is the **unsplit** output of the conditioning MLP:
`[width logits (n_bins) | height logits (n_bins)]`.

## 7. Glossary

| term | meaning |
|---|---|
| **marginal** | The 1-D monotone map for one coordinate; the unit this repository implements. |
| **copula** | The part of the model capturing correlations between coordinates. |
| **PCHIP** | Piecewise Cubic Hermite Interpolating Polynomial — the slope rule that preserves monotonicity. |
| **ESS/N** | Effective sample size. 1.0 means the proposal is perfect; 0 means it is useless. |
| **logabsdet** | `log|dx/du|`, the Jacobian term every change of variables needs. |
| **NFE** | Number of function evaluations, the currency of ODE-solver cost in a CNF. |
| **Talapas** | University of Oregon's cluster: 9 500 cores, 120 GPUs, 3 PB. |
| **sm_86** | The CUDA architecture of the RTX 3060 used for all local measurements. |

## 8. Where to go next

- **To change the kernel** — read `docs/INTERFACE_CONTRACT.md` §1 first; the
  MLP boundary is deliberate.
- **To swap in a CNF** — §3 of the contract lists what a replacement must
  provide. The exactness-of-log-density point is the one that bites.
- **To reproduce the numbers** — `REPRODUCTION_COMMANDS.md`.
- **To know what is left** — `PRD.md` §9.
