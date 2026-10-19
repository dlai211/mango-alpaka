# PRD — mango-alpaka

**Product requirements for the Oregon workstream's Deliverables B and C.**

| | |
|---|---|
| Project | MANGO — Monte Carlo Acceleration via Normalizing Flows using GPU Optimization |
| Sponsor | DOE Genesis Mission, DE-FOA-0003612, Focus Area 14C |
| Phase | Phase I, Months 1–3 (setup sprint) |
| Owner | University of Oregon (Majewski, Le Goff) |
| Status | Deliverables B and C **complete and measured** |
| Version | 0.1.0 — provisional interface |

---

## 1. Context

MANGO replaces classical importance-sampling integration (VEGAS) with
conditional normalizing flows evaluated natively on GPUs, using the **Alpaka**
hardware-abstraction library so the same code runs on NVIDIA, AMD, Intel and
CPU backends.

The physics pipeline is the upstream `conditional-nf-integration` repository,
which is treated here as **read-only reference**. It implements a conditional
flow as a marginal × copula factorisation:

```
r(x | c) = [ Π_i marginal_i(x_i | c) ] × copula(rank x | c)
```

Oregon owns the GPU framework. Brown owns the continuous-flow architecture and
BNL owns physics validation, so this repository has to be usable by both:
a C++ kernel Oregon can optimise, behind an interface Brown and BNL can code
against without a GPU toolchain.

---

## 2. Problem statement

Two problems, both measured rather than assumed.

### 2.1 The reference implementation is overhead-bound

Timing the upstream `ConditionalCubicFlow1D` marginal on an RTX 3060, batch
size from 1 to 50 000:

| operation, dim=2, CUDA | N=1 | N=10 000 | N=50 000 |
|---|---:|---:|---:|
| `sample` (forward) | 3.68 ms | **3.70 ms** | 13.1 ms |
| `log_density` (inverse) | **69.8 ms** | 68.7 ms | **78.8 ms** |

The cost is **flat in batch size**. The arithmetic is effectively free; the
per-call cost is a few dozen small kernel launches, and the inverse runs 40
bisection iterations per element, each a handful of launches.

That is the opening for Alpaka: not faster arithmetic, **fewer launches**.

### 2.2 There is no frozen interface

Brown is about to change the model architecture (discrete coupling layers →
continuous normalizing flows). If the GPU-side interface is not fixed first,
every kernel written against the current one is thrown away. BNL needs the same
boundary stable for validation.

---

## 3. Goals

| # | Goal |
|---|---|
| G1 | A C++17 Alpaka implementation of the monotone cubic-Hermite marginal that is bit-comparable to PyTorch |
| G2 | A build that works on hardware Oregon controls: CUDA today, CPU for portability, room for AMD/Intel backends |
| G3 | A frozen Python ↔ C++ contract Brown and BNL can consume |
| G4 | A measured before/after on the fixed cost, since that is the whole claim |

## 4. Non-goals

- **The copula.** This repository covers one marginal. The autoregressive
  copula stays in Python until the marginal kernel is established.
- **The conditioning MLP.** The contract takes its raw output; folding it in
  would tie the interface to one architecture.
- **Continuous normalizing flows.** That is Brown's deliverable. This
  repository provides the slot a CNF will drop into.
- **Training.** Kernels are inference-side (sample / log-density) only.
- **Packaging.** The module is built into `build/python`; it is not on PyPI and
  should not be until the interface settles.

---

## 5. Requirements

### Functional

| ID | Requirement |
|---|---|
| F1 | `forward(u) -> x` and `inverse(x) -> u` for the conditional marginal |
| F2 | `logabsdet` returned with both directions |
| F3 | Contract names `sample`, `log_density`, `sample_and_log_prob` |
| F4 | Three backends: `cpu_serial`, `cpu_omp2_blocks`, `cuda` |
| F5 | Backend chosen per call; unavailable backend **raises**, never silently falls back |
| F6 | `n_bins` ∈ {8, 16, 32, 64}; the reference model uses **32** |
| F7 | Min bin width/height and inverse iteration count are parameters, not constants |

### Non-functional

| ID | Requirement |
|---|---|
| N1 | Numerical parity with PyTorch at **1e-12** absolute and relative, `float64` |
| N2 | Backends agree with each other to the same tolerance |
| N3 | The interface header includes **neither Alpaka nor pybind11** |
| N4 | The project rebuilds from a clean checkout with one command |
| N5 | Toolchain quirks are documented, not rediscovered |

---

## 6. Acceptance criteria

Deliverable B is accepted when:

1. the kernel reproduces the PyTorch marginal to 1e-12 on all enabled backends
   — **met, 26/26 tests (13 per precision)**, and independently confirmed against a NumPy
   implementation written from the formulas (1e-15 to 1e-13 inside the model's
   logit range);
2. the fixed per-call cost is measured before and after, **on the same device
   and at the same scope** — **met**, with a mixed verdict:

   float32, the pipeline's dtype, after the 0.2.0 optimisations:

   | N | torch inverse (spline) | Alpaka | like-for-like | realistic integration |
   |---:|---:|---:|---:|---:|
   | 1 | 31.4 ms | 0.12 ms | **254×** | 77× |
   | 1 000 | 32.6 ms | 0.19 ms | 176× | 60× |
   | 10 000 | 32.2 ms | 0.97 ms | 33× | 20× |
   | 50 000 | 31.3 ms | 2.48 ms | **12.7×** | 5.9× |

   **Inverse: accepted**, at every batch size.

   | N | torch forward (spline) | Alpaka | like-for-like | realistic integration |
   |---:|---:|---:|---:|---:|
   | 1 | 1.26 ms | 0.12 ms | 10.4× | 2.9× |
   | 1 000 | 1.32 ms | 0.19 ms | 7.1× | 2.4× |
   | 10 000 | 1.30 ms | 1.00 ms | 1.3× | 0.8× |
   | 50 000 | 1.39 ms | 2.48 ms | **0.6×** | 0.3× |

   **Forward: accepted below ~10 000 points, not above.** The residual gap at
   large N is the host→device copy of `raw` plus the MLP; closing it needs
   device pointers (zero-copy), an interface change rather than a kernel one.

Deliverable C is accepted when:

3. `docs/INTERFACE_CONTRACT.md` states the frozen surface, the semantics a
   replacement must preserve, and what is deliberately excluded — **met**;
4. a replacement model (Brown's CNF) can be described against it without
   reopening the kernel — **met as a written contract, untested by a second
   implementation**.

---

## 7. Constraints

| constraint | value |
|---|---|
| OS | Windows 11, MSVC 19.41 (VS 2022 Build Tools) |
| CUDA | 12.4.131 |
| Alpaka | **1.3.0 exactly** — 1.0.0 does not compile under nvcc 12.4 |
| Boost headers | ≥ 1.74 (conda-forge `boost-cpp` 1.74.0 used) |
| pybind11 | 3.1 |
| CMake | ≥ 3.24 (3.28.1 used) |
| GPU | RTX 3060 Laptop, 6 GB, sm_86 |
| Python | 3.10.14, torch 2.3.0+cu121 |

The Alpaka version is a hard constraint, not a preference. See
`RUNBOOK.md` §5.

---

## 8. Risks

**The CNF has no exact log-density.** A continuous flow solved to finite ODE
tolerance gives an approximate density. The integration estimator is unbiased
only if `log_density` is exact, so the error becomes a systematic bias rather
than noise. This must be settled with Brown before the kernel is rewritten.

**A straight port would inherit the launch cost.** The Alpaka implementation
wins because it evaluates each element in one kernel. Any future change that
splits the computation back into stages gives the gain away.

**The forward does not currently win, and it is the half that runs most often.**
The validated result is for the inverse; a pipeline samples with the forward and
scores with the inverse. Using the kernel for sampling today would be several
times slower than PyTorch. `float32` and a copy-free calling convention are
prerequisites for calling Deliverable B done end to end.

**Parity is bounded by the logit range.** The 1e-12 claim holds where the model
operates (logits within `[-1.62, 1.31]`). Beyond `|raw| ≈ 5` the softmax
saturates and inverse log-determinants diverge from PyTorch by ~1e-11, rising
to ~1e-6 at extreme scales. Any future model that produces larger logits needs
the tolerance re-derived.

**The interface may be frozen too early.** 0.1.0 deliberately excludes the
copula and streaming. If those turn out to be coupled to the marginal's layout,
the frozen part will need a version bump.

**Windows-only toolchain.** The kernel is portable C++17; the build script is
not. A Linux/Talapas build is a prerequisite for any multi-node result and has
not been written.

---

## 9. Status

| item | state |
|---|---|
| Deliverable B — Alpaka kernel | **complete**, 26/26 parity tests (13 per precision) |
| Deliverable C — interface + contract | **complete** |
| `float32` support | **done** (0.2.0) — 2.5–3.9× faster than `float64` in the same code |
| Inverse (`log_density`) speedup | **validated** — 254× at N=1, 12.7× at N=50 000, like-for-like |
| Forward (`sample`) speedup | **partial** — 10.4× at N=1, 1.3× at N=10 000, 0.6× at N=50 000 |
| Linux / Talapas build | not started |
| Copula in C++ | not started |
| CNF drop-in | blocked on Brown |
