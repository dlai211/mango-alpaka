# MANGO C++ / Python interface contract

**Deliverable C.** The interface Brown's continuous-flow work and BNL's physics
validation can code against, and the boundary Oregon's Alpaka kernels implement.

Version: **0.1.0** — provisional. The value lists and the semantics below are
what we ask to be frozen; the implementation behind them can change freely.

---

## 1. What this layer is responsible for

A **conditional monotone cubic-Hermite marginal**: a bijection of `[0, 1]` to
itself, conditioned on a real-valued vector, with a tractable log-Jacobian.

It is not the whole model. The upstream pipeline factorises as

```
r(x | c) = [ Π_i marginal_i(x_i | c) ] × copula(rank x | c)
```

and this layer covers the **marginal**. The copula, the conditioning network and
the training loop stay in Python for now.

### Why the conditioning network is outside the kernel

The marginal's parameters come from a small MLP (`3 → 128 → 128 → 2·n_bins`).
That MLP is a separate, dense operation; folding it into the kernel would tie
the interface to one architecture, which is exactly what Brown's CNF work will
replace. So the contract takes the **raw MLP outputs** and owns everything
after them: softmax, bin construction, PCHIP slopes, bin search, Hermite
evaluation, and the inverse solve.

If a later kernel wants to own the MLP too, the contract does not change — only
the meaning of `raw` becomes internal.

---

## 2. API surface

### 2.1 Entry points

| name | signature | returns |
|---|---|---|
| `sample` | `(raw, u, backend="cuda", n_bins=0)` | `x` |
| `log_density` | `(raw, x, backend="cuda", n_bins=0)` | `log q(x)` |
| `sample_and_log_prob` | `(raw, u, backend="cuda", n_bins=0)` | `(x, log|dx/du|)` |
| `forward` | `(raw, u, backend, n_bins)` | `(x, log|dx/du|)` |
| `inverse` | `(raw, x, backend, n_bins)` | `(u, log|dx/du|)` |

`forward` and `inverse` are the primitive pair; `sample`, `log_density` and
`sample_and_log_prob` are the contract names built on them.

### 2.2 Arguments

**`raw`** — `float64`, shape `(n, 2 * n_bins)`, C-contiguous.
Row `i` is `[width logits (n_bins) | height logits (n_bins)]`, exactly the
unsplit output of the conditioning MLP.

**`u`** — `float64`, shape `(n,)`, values in `[0, 1]`. Base-distribution samples.

**`x`** — `float64`, shape `(n,)`, values in `[0, 1]`. Physical samples.

**`backend`** — one of `"cpu_serial"`, `"cpu_omp2_blocks"`, `"cuda"`
(aliases: `"cpu"`, `"cpu_omp2"`, `"gpu"`). Query the build with
`mango_alpaka_ext.backends()`; a backend absent from that list raises
`invalid_argument` rather than silently falling back.

**`n_bins`** — `int`, optional. `0` (the default) infers it from
`raw.shape[1] / 2`; a non-zero value is validated against the shape. Supported
values are `8, 16, 32, 64`; the reference model uses **32**.

### 2.3 Semantics

The base distribution is **uniform on `[0,1]`**, matching the reference
implementation. That gives

```
log_density(x) = -log |dx/du|
```

If Brown changes the base distribution, `log_density` must subtract the base
log-density — this is the one semantic a CNF replacement will *not* inherit
automatically, and it is worth deciding before any kernel is rewritten.

`inverse` returns the log-Jacobian **evaluated at the recovered `u`**, not at
the input `x`; they agree analytically but not to the last bit, and the
reference's `inverse` returns the former.

---

## 3. What a replacement model must provide

For a continuous normalising flow to drop into this slot:

1. `sample(raw_or_params, u) -> x` — solve the ODE from base to data.
2. `log_density(raw_or_params, x) -> log q(x)` — integrate the divergence.
3. **Exactness of the likelihood.** The integration estimator is unbiased only
   if `log_density` is the exact log-density; an approximate or truncated ODE
   solve breaks that silently.
4. **Determinism** given inputs, so the parity tests remain meaningful.

Item 3 is the one to raise early: a CNF solved to finite tolerance does not
have an exact log-density, and the difference is a systematic bias in the
integral rather than noise.

---

## 4. Guarantees this implementation makes

- **Parity with the reference** at `1e-12` absolute and relative, in `float64`,
  for forward, inverse, log-Jacobian and `log_density`, at batch sizes
  1 → 5 000, on every enabled backend.
  See `tests/test_cubic_spline_parity.py` (12 tests).
- **Backends agree with each other** to the same tolerance.
- **Round trip** `x → u → x` reproduces the input to `1e-10`.
- **No silent backend fallback.** An unavailable backend raises.

`float32` is not currently exposed. The reference model is `float32` in
training and `float64` in evaluation; if a `float32` path is needed, the
tolerance must be re-derived rather than assumed.

---

## 5. Measured behaviour

### Fixed cost in the reference implementation

From `tests/benchmark_fixed_cost.py` (first 40 warmup-free calls, best of 20):

| backend | dim | N=1 | N=10 000 | N=50 000 |
|---:|---:|---:|---:|---:|
| cpu | 2 | 0.94 ms | 13.3 ms | 78.7 ms |
| cuda | 2 | **3.68 ms** | **3.70 ms** | 13.1 ms |

`sample` on CUDA is flat from N=1 to N=10 000: the arithmetic is free and the
launch cost is everything.

The inverse is worse — it runs 40 bisection iterations per element:

| backend | dim | N=1 | N=50 000 |
|---:|---:|---:|---:|
| cpu | 2 | 13.1 ms | 127 ms |
| cuda | 2 | **65.9 ms** | **78.8 ms** |

### Against the kernel

`tests/benchmark_kernel_vs_reference.py`, dim=2, `float64`, end to end through
the Python binding (both columns include host↔device copies):

| N | torch forward | Alpaka forward | speedup | torch inverse | Alpaka inverse | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.30 ms | 1.21 ms | 1.1× | 16.09 ms | 1.35 ms | **11.9×** |
| 1 000 | 5.19 ms | 2.18 ms | 2.4× | 28.65 ms | 2.30 ms | **12.5×** |
| 50 000 | 193.2 ms | 29.3 ms | 6.6× | 306.1 ms | 29.0 ms | **10.6×** |

The inverse gain is roughly constant across batch sizes, which is the signature
of removing a fixed per-call cost rather than winning on arithmetic.

---

## 6. What is deliberately not here yet

- **The conditioning MLP** (§1).
- **The copula.** The contract covers one marginal; a joint interface needs a
  decision on how the autoregressive copula receives ranks.
- **Streams / batching policy.** Every call allocates buffers and synchronises.
  A production interface should accept caller-owned buffers and a queue.
- **`float32`.**
- **A Python package.** The module is loaded from `build/python`; packaging is
  deliberate later, because the interface will move first.

---

## 7. Change policy

`Backend` values, the `raw` layout and the uniform-base `log_density` semantics
are frozen at 0.1.0. Anything else may change without a version bump while this
remains provisional. Changes to the frozen set require agreement from Brown
(architecture) and BNL (validation), because both consume them.
