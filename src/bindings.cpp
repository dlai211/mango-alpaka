// bindings.cpp — the Python entry points of Deliverable C.
//
// This file is compiled by MSVC and includes only pybind11 plus the plain
// mango_interface.hpp. It never sees Alpaka. The kernels live in launcher.cu,
// built by nvcc, and are linked in as ordinary C++ symbols.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mango_interface.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace
{
    using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

    mango::Backend parse_backend(std::string const& name)
    {
        if(name == "cpu" || name == "cpu_serial")
            return mango::Backend::CpuSerial;
        if(name == "cpu_omp2" || name == "cpu_omp2_blocks")
            return mango::Backend::CpuOmp2Blocks;
        if(name == "cuda" || name == "gpu")
            return mango::Backend::Cuda;
        throw std::invalid_argument("unknown backend '" + name + "'");
    }

    //! Validate the (n, 2*n_bins) raw-parameter array and return n_bins.
    int check_raw(Array const& raw, int requested_bins)
    {
        if(raw.ndim() != 2)
            throw std::invalid_argument("raw must have shape (n, 2 * n_bins)");
        auto const columns = static_cast<int>(raw.shape(1));
        if(columns % 2 != 0)
            throw std::invalid_argument("raw.shape[1] must be even (2 * n_bins)");
        int const bins = columns / 2;
        if(requested_bins != 0 && requested_bins != bins)
            throw std::invalid_argument(
                "n_bins=" + std::to_string(requested_bins) + " does not match raw.shape[1]/2="
                + std::to_string(bins));
        return bins;
    }

    struct Outputs
    {
        py::array_t<double> value;
        py::array_t<double> logabsdet;
    };

    Outputs make_outputs(py::ssize_t n)
    {
        return Outputs{py::array_t<double>(n), py::array_t<double>(n)};
    }

    //! Forward map. Returns (x, log|dx/du|).
    Outputs forward_impl(Array raw, Array u, std::string const& backend, int n_bins)
    {
        auto const n = static_cast<std::int64_t>(raw.shape(0));
        if(u.ndim() != 1 || u.shape(0) != n)
            throw std::invalid_argument("u must have shape (n,) matching raw.shape[0]");

        mango::SplineSpec spec;
        spec.n = n;
        spec.n_bins = check_raw(raw, n_bins);

        auto out = make_outputs(n);
        mango::cubic_spline_forward(
            parse_backend(backend),
            raw.data(),
            u.data(),
            out.value.mutable_data(),
            out.logabsdet.mutable_data(),
            spec);
        return out;
    }

    //! Inverse map. Returns (u, log|dx/du|) — the reference's `inverse` contract.
    Outputs inverse_impl(Array raw, Array x, std::string const& backend, int n_bins)
    {
        auto const n = static_cast<std::int64_t>(raw.shape(0));
        if(x.ndim() != 1 || x.shape(0) != n)
            throw std::invalid_argument("x must have shape (n,) matching raw.shape[0]");

        mango::SplineSpec spec;
        spec.n = n;
        spec.n_bins = check_raw(raw, n_bins);

        auto out = make_outputs(n);
        mango::cubic_spline_inverse(
            parse_backend(backend),
            raw.data(),
            x.data(),
            out.value.mutable_data(),
            out.logabsdet.mutable_data(),
            spec);
        return out;
    }

} // namespace

PYBIND11_MODULE(mango_alpaka_ext, module)
{
    module.doc() = "MANGO Alpaka kernels: monotone cubic-Hermite marginal spline.";

    module.def(
        "forward",
        [](Array raw, Array u, std::string const& backend, int n_bins) {
            auto result = forward_impl(raw, u, backend, n_bins);
            return py::make_tuple(result.value, result.logabsdet);
        },
        py::arg("raw"),
        py::arg("u"),
        py::arg("backend") = "cuda",
        py::arg("n_bins") = 0,
        R"doc(Forward map u -> x through the conditional cubic-Hermite spline.

raw : (n, 2*n_bins) float64, [width logits | height logits]
u   : (n,) float64 in [0, 1]
Returns (x, logabsdet) where logabsdet = log|dx/du|.
)doc");

    module.def(
        "inverse",
        [](Array raw, Array x, std::string const& backend, int n_bins) {
            auto result = inverse_impl(raw, x, backend, n_bins);
            return py::make_tuple(result.value, result.logabsdet);
        },
        py::arg("raw"),
        py::arg("x"),
        py::arg("backend") = "cuda",
        py::arg("n_bins") = 0,
        R"doc(Inverse map x -> u by per-bin bisection plus one Newton correction.

Returns (u, logabsdet) with logabsdet = log|dx/du| at the recovered u.
)doc");

    // --- the Brown-facing contract ------------------------------------------

    module.def(
        "sample",
        [](Array raw, Array u, std::string const& backend, int n_bins) {
            return forward_impl(raw, u, backend, n_bins).value;
        },
        py::arg("raw"),
        py::arg("u"),
        py::arg("backend") = "cuda",
        py::arg("n_bins") = 0,
        "Contract: draw from the flow by mapping base samples u -> x.");

    module.def(
        "log_density",
        [](Array raw, Array x, std::string const& backend, int n_bins) {
            // Uniform base: log q(x) = -log|dx/du|, evaluated through the inverse.
            auto result = inverse_impl(raw, x, backend, n_bins);
            auto logdet = result.logabsdet;
            auto* data = logdet.mutable_data();
            for(py::ssize_t i = 0; i < logdet.size(); ++i)
                data[i] = -data[i];
            return logdet;
        },
        py::arg("raw"),
        py::arg("x"),
        py::arg("backend") = "cuda",
        py::arg("n_bins") = 0,
        "Contract: log q(x) for a uniform base distribution.");

    module.def(
        "sample_and_log_prob",
        [](Array raw, Array u, std::string const& backend, int n_bins) {
            auto result = forward_impl(raw, u, backend, n_bins);
            return py::make_tuple(result.value, result.logabsdet);
        },
        py::arg("raw"),
        py::arg("u"),
        py::arg("backend") = "cuda",
        py::arg("n_bins") = 0,
        "Contract: sample and return (x, log|dx/du|) in one call.");

    // --- introspection -------------------------------------------------------

    module.def(
        "backend_name",
        [](std::string const& name) { return std::string(mango::backend_name(parse_backend(name))); },
        py::arg("backend"));

    module.def(
        "backend_available",
        [](std::string const& name) { return mango::backend_available(parse_backend(name)); },
        py::arg("backend"));

    module.def("backends", []() {
        std::vector<std::string> enabled;
        for(auto const* name : {"cpu_serial", "cpu_omp2_blocks", "cuda"})
            if(mango::backend_available(parse_backend(name)))
                enabled.emplace_back(name);
        return enabled;
    });
}
