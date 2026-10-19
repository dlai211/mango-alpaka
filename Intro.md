# Intro — what this repository is, and how the code works

A one-sitting orientation. `PRD.md` has requirements, `RUNBOOK.md` has the
build, `RESULTS.md` has the numbers.

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

---

## 3. What we are trying to accomplish

### 3.1 The structure of the model

The upstream pipeline factorises the conditional density:

```
r(x | c) = [ Π_i marginal_i(x_i | c) ] × copula(rank x | c)
```

- **`marginal_i`** — a monotone map of `[0,1]` to itself for coordinate `i`,
  with a tractable log-Jacobian. There are `d` of them (one per dimension).
- **`copula`** — captures the correlations *between* coordinates.

**This repository implements the marginal only.** The copula, the conditioning
network and the training loop stay in the upstream Python.

### 3.2 Why the marginal is the right first target

It is the component evaluated most often. In the upstream training loop,

```python
def marginal_log_prob(bank, x, cond):
    total_logq = torch.zeros(x.shape[0])
    for j in range(DIM):
        _, log_jac = bank.marginals[j].inverse(x[:, j:j+1], cond)
        total_logq += -log_jac.reshape(-1)
    return total_logq
```

that is `d` **sequential** calls per evaluation — 8 separate inverse calls for
the 8-D process, each with its own Python dispatch and kernel launches. Nothing
about the eight is coupled, but the loop serialises them, and in PyTorch each
one launches a few dozen small kernels.

### 3.3 The three goals

| goal | why |
|---|---|
| **Fuse** each marginal into one kernel launch | the reference is launch-bound, not arithmetic-bound (§8) |
| **Keep it bit-comparable** | 1e-12 against the reference, so a swap is invisible to physics validation |
| **Freeze an interface** | Brown is about to replace the model architecture; the kernel boundary should not move with it |

### 3.4 What "done" means

- the kernel reproduces the reference to 1e-12 on every backend — **done**;
- it is *faster* where the pipeline uses it — **partially**: the inverse wins at
  every size, the forward does not (§8.3);
- the contract is written and stable — **done**.

---

## 4. The reference model, line by line

The upstream class is `ConditionalCubicFlow1D` in
`nf/conditional_cubic_1d.py`, subclassing `ConditionalLinearFlow1D` in
`nf/conditional_linear_1d.py`. Read in this order.

### 4.1 The conditioning network

```python
self.net = nn.Sequential(
    nn.Linear(cond_dim, hidden), nn.ReLU(),
    nn.Linear(hidden, hidden),   nn.ReLU(),
    nn.Linear(hidden, 2 * n_bins),
)
```

Three affine layers. For the pipeline's settings (`cond_dim=3`, `hidden=128`,
`n_bins=32`) it maps `(n, 3) → (n, 64)`: **32 width logits and 32 height
logits per element**. Nothing else in the model needs `cond`.

### 4.2 From logits to bins — `_params(cond)`

```python
raw_widths, raw_heights = self.net(cond).chunk(2, dim=-1)
widths  = min_bin_width  + (1 - n_bins*min_bin_width)  * softmax(raw_widths,  -1)
heights = min_bin_height + (1 - n_bins*min_bin_height) * softmax(raw_heights, -1)
cumwidths  = pad(cumsum(widths,  -1), (1, 0))
cumheights = pad(cumsum(heights, -1), (1, 0))
cumwidths[:, -1]  = 1.0          # forced, not computed
cumheights[:, -1] = 1.0
```

Four things to notice, because each is a place to get the port wrong:

1. **The softmax is over each element's own 32 logits**, not across the batch.
2. **`min_bin_*` is added before the cumulative sum**, so it shifts every bin,
   not just the last.
3. **The last knot is set to exactly 1.** `cumsum` of the softmax does not hit 1
   exactly in floating point, and the comment says so.
4. **`widths` and `cumwidths` are kept separately.** The final bin's width is
   `widths[31]`, while `cumwidths[32] - cumwidths[31]` would differ slightly
   because of the forced 1. Both sides must use `widths[idx]`.

### 4.3 PCHIP slopes — `_derivatives(widths, heights)`

```python
secants = heights / widths
weight1 = 2*right_w + left_w
weight2 = right_w + 2*left_w
interior = (weight1 + weight2) / (weight1/secants[:-1] + weight2/secants[1:])
d = cat([secants[:, :1], interior, secants[:, -1:]])
```

One derivative per knot, `n_bins + 1` of them. This is the Fritsch–Carlson
construction: positive secants produce positive slopes, which is what makes the
map monotone and therefore invertible. Note the interior is a **harmonic** mean,
not arithmetic — using the arithmetic mean can overshoot and break monotonicity.

### 4.4 Selecting a bin — `_selected(values, cond, inverse)`

```python
knots = cumheights if inverse else cumwidths
idx = (values[:, None] >= knots[:, 1:]).sum(-1).clamp_max(n_bins - 1)
```

A count of how many interior knots the value has passed. `clamp_max` handles
`value == 1`, which would otherwise index past the last bin. Direction matters:
the **forward** searches on `cumwidths` (the input space), the **inverse** on
`cumheights` (the output space).

### 4.5 The Hermite polynomial — `_hermite(t, y0, height, width, d0, d1)`

```python
t2, t3 = t*t, t*t*t
x = ((2*t3 - 3*t2 + 1)*y0 + (t3 - 2*t2 + t)*width*d0
     + (-2*t3 + 3*t2)*(y0 + height) + (t3 - t2)*width*d1)
dx_dt = ((6*t2 - 6*t)*y0 + (3*t2 - 4*t + 1)*width*d0
         + (-6*t2 + 6*t)*(y0 + height) + (3*t2 - 2*t)*width*d1)
```

`t` is the position **within the bin**, in `[0,1]`. `y0` is the bin's lower
output knot, `height` its output size, `width` its input size, `d0`/`d1` the
slopes at its two ends. `dx_dt` is returned alongside because it is exactly the
Jacobian: `log|dx/du| = log(dx_dt) - log(width)`.

### 4.6 Forward

```python
idx = bin index on cumwidths
t = ((u - u0) / width).clamp(0, 1)
x, dx_dt = hermite(t, x0, height, width, d0, d1)
logabsdet = log(dx_dt.clamp_min(tiny)) - log(width)
```

### 4.7 Inverse

```python
idx = bin index on cumheights
low, high = 0, 1
for _ in range(40):
    mid = (low + high)/2
    x_mid, _ = hermite(mid, ...)
    low, high = where(x_mid < x, mid, low), where(x_mid < x, high, mid)
t = (low + high)/2
t = clamp(t - (hermite(t) - x)/dx_dt.clamp_min(tiny), 0, 1)   # one Newton step
u = (u0 + width*t).clamp(0, 1)
```

There is no closed-form inverse of a cubic, so this is a bracketed solve:
**40 bisection iterations** for global safety, then **one Newton correction** so
the result follows the implicit root. In PyTorch that is ~40 rounds of small
kernel launches per call — the reason the inverse costs ~32 ms regardless of
batch size.

### 4.8 What the model exposes

```python
sample(n, cond)      # u ~ Uniform(0,1) then forward — draw from the marginal
log_prob(x, cond)    # -inverse(x).logabsdet   — uniform base density
forward(u, cond)     # -> (x, logabsdet)
inverse(x, cond)     # -> (u, logabsdet)
```

---

## 5. The port, function by function

The C++ side mirrors 4.2–4.7 exactly. `include/kernels/cubic_spline_1d.hpp`:

| C++ | Python it mirrors | notes |
|---|---|---|
| `scaled_softmax<T,N>(logits, min_bin, out)` | the two `softmax` lines in `_params` | numerically stable (subtract max) |
| `cumulative<T,N>(bins, cum)` | `pad(cumsum)` + forced last knot | writes `N+1` knots, last = 1 |
| `knot_derivatives<T,N>(width, height, d)` | `_derivatives` | PCHIP harmonic mean |
| `hermite<T>(t, y0, h, w, d0, d1, x, dxdt)` | `_hermite` | returns value and derivative |
| `bin_index<T,N>(value, knots)` | the `(values[:,None] >= knots[:,1:]).sum(-1)` expression | counts, then clamps |
| `prepare<T,N>(raw, …)` | `_params` + `_derivatives` | the per-element preamble |
| `ForwardKernel<T, IdxT, N>` | `forward` | one thread per element |
| `InverseKernel<T, IdxT, N>` | `inverse` | bisection + Newton, in thread-local registers |

`n_bins` is a **template parameter**, not a runtime value: it lets the local
arrays (`width[32]`, `cumw[33]`, …) live in registers/local memory instead of
heap. `launcher.cu` instantiates 8, 16, 32 and 64; the model uses 32.

### 5.1 The kernels

Both kernels are one loop:

```cpp
for (auto i : alpaka::uniformElements(acc, n)) {
    const T* raw = raw_all + i * (2 * N_BINS);
    T width[N_BINS], height[N_BINS], cumw[N_BINS+1], cumh[N_BINS+1], d[N_BINS+1];
    prepare<T, N_BINS>(raw, min_w, min_h, width, height, cumw, cumh, d);
    // forward: bin on cumw, hermite, write x and logdet
    // inverse: bin on cumh, 40x bisection, 1x Newton, write u and logdet
}
```

Everything an element needs is computed and discarded inside one iteration.
That is the entire optimisation: the reference recomputes 32 bins and 33 knots
in a fresh set of kernel launches *per operation*, and there are ~40 operations
in the inverse.

### 5.2 The launcher — `src/launcher.cu`

Compiled by `nvcc`, because it includes Alpaka's CUDA headers.

| function | role |
|---|---|
| `run_backend<Acc, N_BINS, INVERSE>` | allocates host and device buffers, copies in, launches, copies out |
| `dispatch_bins<Acc, INVERSE>` | switches on `n_bins` → 8 / 16 / 32 / 64 |
| `dispatch_backend<INVERSE>` | switches on `Backend` → `CpuSerial` / `CpuOmp2Blocks` / `Cuda` |
| `cubic_spline_forward/inverse` | the plain C++ entry points declared in the contract header |
| `backend_name`, `backend_available` | introspection; an unavailable backend **throws** |

It is compiled **only** with nvcc. That is a deliberate split, not an accident:

```
bindings.cpp  (MSVC)  ->  pybind11, mango_interface.hpp        no Alpaka
launcher.cu   (nvcc)  ->  Alpaka, kernels, mango_interface.hpp  no pybind11
```

The two meet through `mango_interface.hpp`, which includes neither. That keeps
pybind11 out of nvcc (which fights over compile flags) and keeps Alpaka out of
anything Brown has to build.

### 5.3 The Python module — `src/bindings.cpp`

| function | semantics |
|---|---|
| `forward(raw, u)` | → `(x, logabsdet)` |
| `inverse(raw, x)` | → `(u, logabsdet)` |
| `sample(raw, u)` | → `x` (the contract name) |
| `log_density(raw, x)` | → `-logabsdet` from the inverse (uniform base) |
| `sample_and_log_prob(raw, u)` | → `(x, logabsdet)` |
| `backends()` | which backends this build enabled |

`n_bins` is inferred from `raw.shape[1] / 2` unless passed explicitly.

---

## 6. The boundary: where `cond` stops and `raw` begins

```
cond (n, 3)
   │   Python / PyTorch:  Linear(3→128) → ReLU → Linear(128→128) → ReLU → Linear(128→64)
   ▼
raw (n, 64)   ◄── THE INTERFACE (docs/INTERFACE_CONTRACT.md)
   │   C++ / Alpaka:     softmax → knots → slopes → bin → Hermite → bisection
   ▼
x (n) or u (n),  log|dx/du| (n)
```

The MLP stays in PyTorch for two reasons:

1. **It is not the bottleneck.** It costs 0.25–3 ms in the measurements, against
   ~32 ms for the inverse spline.
2. **It is what Brown will replace.** Putting the MLP inside the kernel would
   bake a specific architecture into the interface, right before the
   architecture changes.

The cost of the boundary is visible in `RESULTS.md` §2: the `alpaka realistic`
column includes the MLP plus a host copy of `raw`, and it is measurably worse
than the `alpaka spline` column. That gap is the price of the split, and the
next optimisation task.

---

## 7. What this repository does and does not cover

| covered | not covered |
|---|---|
| the 1-D conditional monotone cubic-Hermite marginal | the copula |
| forward and inverse with log-Jacobian | the conditioning MLP |
| three Alpaka backends | CNFs / flow matching (Brown) |
| parity, benchmarks, interface contract | training |
| CPU and CUDA | Linux / Talapas build |

---

## 8. Why Alpaka, and where it wins

Alpaka is a header-only C++17 abstraction that compiles one kernel for CUDA,
HIP, SYCL and CPUs. For MANGO that is strategic: the project must not bet on one
vendor.

### 8.1 The reference is launch-bound

| operation, CUDA, dim=2 | N=1 | N=10 000 |
|---|---:|---:|
| forward | 3.68 ms | 3.70 ms |
| inverse | 69.8 ms | 68.7 ms |

Flat in batch size. Those times are **launch overhead**, not arithmetic.

### 8.2 The kernel removes it — for the inverse

Like-for-like on the spline, one marginal per call, same GPU:

| N | torch inverse | Alpaka | speedup |
|---:|---:|---:|---:|
| 1 | 35.1 ms | 0.27 ms | **130×** |
| 1 000 | 31.3 ms | 0.82 ms | 38× |
| 10 000 | 30.3 ms | 2.91 ms | 10× |
| 50 000 | 33.1 ms | 11.4 ms | **2.9×** |

### 8.3 The forward is a different answer

| N | torch forward | Alpaka | speedup |
|---:|---:|---:|---:|
| 1 | 1.26 ms | 0.25 ms | 5.1× |
| 1 000 | 1.27 ms | 0.64 ms | 2.0× |
| 10 000 | 1.42 ms | 3.34 ms | **0.4×** |
| 50 000 | 2.23 ms | 11.8 ms | **0.2×** |

Above N ≈ 1 000 the kernel is *slower*. It computes in `float64` (1/32 the
rate of `float32` on a consumer GPU), recomputes all 32 bins per element, and
copies parameters per call. PyTorch's forward is a few vectorised ops with
almost no launch cost to amortise.

**Deliverable B is validated for the inverse, not for the forward.** That is
stated in `PRD.md` §9 and is the next kernel task.

---

## 9. Repository map

| path | what it is |
|---|---|
| `include/mango_interface.hpp` | The frozen contract. No Alpaka, no pybind11 — Brown can include it with a plain C++17 compiler. |
| `include/kernels/cubic_spline_1d.hpp` | The Alpaka kernel: one thread per element, everything fused. |
| `src/launcher.cu` | Instantiates the kernel per backend. Compiled by nvcc. |
| `src/bindings.cpp` | pybind11 module `mango_alpaka_ext`. Compiled by MSVC, never sees Alpaka. |
| `tests/test_cubic_spline_parity.py` | Kernel vs PyTorch at 1e-12. |
| `tests/benchmark_fixed_cost.py` | The pre-kernel baseline. |
| `tests/benchmark_kernel_vs_reference.py` | The before/after, four scopes. |
| `docs/INTERFACE_CONTRACT.md` | What Brown and BNL code against. |
| `third_party/alpaka-1.3.0/` | Not committed; fetched by `scripts/fetch_dependencies.sh`. |

---

## 10. Five-minute start

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

---

## 11. Glossary

| term | meaning |
|---|---|
| **marginal** | The 1-D monotone map for one coordinate; the unit this repository implements. |
| **copula** | The part of the model capturing correlations between coordinates. |
| **PCHIP** | Piecewise Cubic Hermite Interpolating Polynomial — the slope rule that preserves monotonicity. |
| **secant** | `height/width` of a bin; the average slope across it. |
| **knot** | A bin boundary; the cumulative widths or heights. |
| **ESS/N** | Effective sample size. 1.0 means the proposal is perfect; 0 means it is useless. |
| **logabsdet** | `log|dx/du|`, the Jacobian term every change of variables needs. |
| **NFE** | Number of function evaluations, the currency of ODE-solver cost in a CNF. |
| **Talapas** | University of Oregon's cluster: 9 500 cores, 120 GPUs, 3 PB. |
| **sm_86** | The CUDA architecture of the RTX 3060 used for all local measurements. |

---

## 12. Where to go next

- **To change the kernel** — read `docs/INTERFACE_CONTRACT.md` §1 first; the
  MLP boundary is deliberate.
- **To swap in a CNF** — §3 of the contract lists what a replacement must
  provide. The exactness-of-log-density point is the one that bites.
- **To reproduce the numbers** — `REPRODUCTION_COMMANDS.md`.
- **To know what is left** — `PRD.md` §9.
