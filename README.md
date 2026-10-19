# mango-alpaka

Alpaka implementation of the MANGO normalising-flow marginal, and the C++/Python
interface other MANGO institutions code against.

This is **Deliverables B and C** of the Oregon workstream's Phase 1. The
upstream pipeline lives in `../conditional-nf-integration` and is treated as a
read-only reference.

---

## Documentation

| file | read it for |
|---|---|
| **`Intro.md`** | what this is and why — orientation, the maths in one page, glossary |
| **`PRD.md`** | requirements, acceptance criteria, scope, risks, status |
| **`RUNBOOK.md`** | building it, and the five toolchain traps |
| **`REPRODUCTION_COMMANDS.md`** | exact commands to regenerate every number |
| **`RESULTS.md`** | the measurements and what they do and do not claim |
| `docs/INTERFACE_CONTRACT.md` | the frozen Python ↔ C++ contract (Deliverable C) |
| `docs/CODE_COMPARISON.md` | the PyTorch marginal and the Alpaka kernel, explained and shown side by side |
| **`index.html`** | interactive anatomy: the pipeline, the flow, Alpaka, and where the kernel wins or loses |

New here? Start with `Intro.md`. Rebuilding? `RUNBOOK.md`. Reproducing a
number? `REPRODUCTION_COMMANDS.md`.

---

## What is here

```
include/mango_interface.hpp        the frozen contract (no Alpaka, no pybind11)
include/kernels/cubic_spline_1d.hpp Deliverable B: the fused Alpaka kernel
src/launcher.cu                    kernel instantiation (compiled by nvcc)
src/bindings.cpp                   pybind11 module (compiled by MSVC)
tests/benchmark_fixed_cost.py      pre-kernel fixed-cost baseline
tests/benchmark_kernel_vs_reference.py  before/after
tests/test_cubic_spline_parity.py  kernel vs PyTorch at 1e-12
docs/INTERFACE_CONTRACT.md         Deliverable C's written note
docs/CODE_COMPARISON.md            PyTorch vs Alpaka, explanation and code side by side
```

---

## Build

Requirements, all present on the development machine:

| component | version used |
|---|---|
| CMake | 3.28 |
| MSVC | 19.41 (VS 2022 Build Tools) |
| CUDA | 12.4 |
| Alpaka | **1.3.0** (vendored in `third_party/`) |
| Boost headers | ≥ 1.74 (from conda-forge `boost-cpp`) |
| pybind11 | 3.1 |

```cmd
build.bat
```

`build.bat` sources `vcvars64.bat` first. That is required, not cosmetic: `cl.exe`
is on PATH but the Windows SDK tools (`rc.exe`, `mt.exe`) that CMake's compiler
check needs are not, so configuring without it fails with "No CUDA toolset
found" under the Visual Studio generator and a resource-compiler error under
Ninja.

The module lands in `build/python/mango_alpaka_ext.cp310-win_amd64.pyd`, with
`cudart64_12.dll` copied beside it — CPython 3.8+ does not search PATH for the
dependencies of an extension module.

---

## Use

```python
import sys; sys.path.insert(0, "build/python")
import mango_alpaka_ext as mango

print(mango.backends())                       # ['cpu_serial', 'cpu_omp2_blocks', 'cuda']
x, logdet = mango.forward(raw, u, backend="cuda")   # raw: (n, 2*n_bins) float64
logq = mango.log_density(raw, x, backend="cuda")    # uniform base
```

`raw` is the unsplit output of the conditioning MLP:
`[width logits (n_bins) | height logits (n_bins)]`.

---

## Test

```cmd
set PYTHONPATH=build/python
python -m pytest tests/test_cubic_spline_parity.py -v
```

26 tests (13 per precision): forward, inverse, log-density, round trip, contract
conformance and cross-backend agreement — at `1e-12` in `float64` and a measured
`2e-5` in `float32`.

---

## Measured

**The reference implementation is overhead-bound, not arithmetic-bound.**
`sample` on CUDA costs 3.68 ms at N=1 and 3.70 ms at N=10 000 — the arithmetic
is free and the launch cost is everything. The inverse is worse: it runs 40
bisection iterations per element, so `log_density` costs 65.9 ms at N=1 and
78.8 ms at N=50 000.

End to end through the Python binding, **both sides on the GPU**, like-for-like
on the spline (torch's conditioning MLP is excluded from its column — see
`RESULTS.md` §2):

| N | inverse | forward | inverse, realistic | forward, realistic |
|---:|---:|---:|---:|---:|
| 1 | **254×** | 10.4× | 77× | 2.9× |
| 1 000 | 176× | 7.1× | 60× | 2.4× |
| 10 000 | 33× | 1.3× | 20× | **0.8×** |
| 50 000 | 12.7× | **0.6×** | 5.9× | **0.3×** |

"realistic" = MLP in torch → host copy → kernel, which is what an integration
would do today.

**The inverse is a validated win at every size. The forward wins below ~10 000
points and loses above.** The residual forward gap at large N is the host→device
copy of `raw` plus the MLP in the realistic path; closing it needs device
pointers (zero-copy), which is an interface change. See `PRD.md` §9.

---

## Notes for the reader coming from the Python side

- **Scope.** The kernel covers the *marginal*, not the copula and not the
  conditioning MLP. See `docs/INTERFACE_CONTRACT.md` §1 for why.
- **Base distribution is uniform on [0,1].** `log_density(x) = -log|dx/du|`.
  A CNF replacement will have to revisit this.
- **Alpaka 1.3.0 is required.** 1.0.0 does not compile its CUDA buffer header
  under nvcc 12.4 (`Idx is not a template`, an unqualified dependent alias).
- **Do not name a global alias `Idx` in a translation unit that includes
  Alpaka.** It shadows `alpaka::Idx<T>` during Alpaka's own header parse and
  produces the same error.
