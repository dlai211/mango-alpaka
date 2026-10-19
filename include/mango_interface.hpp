// mango_interface.hpp — the frozen Python <-> C++ contract (Deliverable C).
//
// This header is deliberately free of Alpaka and pybind11 includes so that it
// can be consumed by:
//   * src/bindings.cpp   (compiled by MSVC, includes pybind11)
//   * src/launcher.cu    (compiled by nvcc, includes Alpaka)
//   * Brown University  (a plain C++17 dependency, no GPU toolchain needed)
//
// The Python-facing semantics are defined in docs/INTERFACE_CONTRACT.md.
//
// Spline scope: the *marginal* transform only — the monotone cubic-Hermite map
// from [0,1] to [0,1] conditioned on a vector. The conditioning network (a
// 3 -> 128 -> 128 -> 2*n_bins MLP) is outside the kernel for now; its raw
// outputs are the input here. See the note in the contract doc.

#pragma once

#include <cstdint>

namespace mango
{

    //! Compute backend selected at call time.
    //!
    //! Values are fixed so they can cross a pybind11 boundary as integers.
    enum class Backend : int
    {
        CpuSerial = 0,
        CpuOmp2Blocks = 1,
        Cuda = 2,
    };

    //! Describes one batch of splines.
    struct SplineSpec
    {
        //! Number of independent elements in the batch.
        std::int64_t n = 0;

        //! Bins per spline. Supported values are instantiated at compile time:
        //! 8, 16, 32, 64. The reference model uses 32.
        int n_bins = 32;

        //! Lower bounds on bin width and height, as in the PyTorch reference.
        double min_bin_width = 1e-4;
        double min_bin_height = 1e-4;

        //! Bisection iterations used by the inverse. The reference uses 40.
        int inverse_iterations = 40;
    };

    //! Forward map u -> x through the monotone cubic-Hermite spline.
    //!
    //! \param raw     (n, 2 * n_bins) row-major: [width logits | height logits].
    //!                These are the raw outputs of the conditioning MLP.
    //! \param u       (n,) values in [0, 1].
    //! \param x       (n,) output values.
    //! \param logabsdet (n,) log |dx/du| at each element.
    void cubic_spline_forward(
        Backend backend,
        const double* raw,
        const double* u,
        double* x,
        double* logabsdet,
        const SplineSpec& spec);

    //! Inverse map x -> u, by per-bin bisection plus one Newton correction.
    //!
    //! \param logabsdet receives log |dx/du| evaluated at the recovered u, which
    //!        is the quantity the reference returns from `inverse`.
    void cubic_spline_inverse(
        Backend backend,
        const double* raw,
        const double* x,
        double* u,
        double* logabsdet,
        const SplineSpec& spec);

    //! Human-readable name, for error messages and logging.
    char const* backend_name(Backend backend);

    //! True if this build enabled the backend.
    bool backend_available(Backend backend);

} // namespace mango
