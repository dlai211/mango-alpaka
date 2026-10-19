// bindings.cpp — the Python entry points of Deliverable C.
//
// Compiled by MSVC and includes only pybind11 plus the plain
// mango_interface.hpp. It never sees Alpaka. The kernels live in launcher.cu,
// built by nvcc, and are linked in as ordinary C++ symbols.
//
// float32 and float64 are both accepted. The dtype of `raw` selects the compute
// precision, every other buffer must match, and the outputs come back in the
// same dtype.

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

    mango::Precision precision_of(py::array const& array, char const* what)
    {
        auto const dtype = array.dtype();
        if(dtype.is(py::dtype::of<double>()))
            return mango::Precision::Float64;
        if(dtype.is(py::dtype::of<float>()))
            return mango::Precision::Float32;
        throw std::invalid_argument(
            std::string(what) + ": expected float32 or float64, got "
            + py::str(dtype).cast<std::string>());
    }

    void require_contiguous(py::array const& array, char const* what)
    {
        if(!(array.flags() & py::array::c_style))
            throw std::invalid_argument(std::string(what) + " must be C-contiguous");
    }

    py::array make_output(mango::Precision precision, py::ssize_t n)
    {
        if(precision == mango::Precision::Float64)
            return py::array_t<double>(n);
        return py::array_t<float>(n);
    }

    //! Validate the (n, 2*n_bins) raw-parameter array and return n_bins.
    int check_raw(py::array const& raw, int requested_bins)
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
        py::array value;
        py::array logabsdet;
        mango::Precision precision;
    };

    //! Forward map. Returns (x, log|dx/du|).
    Outputs forward_impl(py::array raw, py::array u, std::string const& backend, int n_bins)
    {
        require_contiguous(raw, "raw");
        require_contiguous(u, "u");
        auto const precision = precision_of(raw, "raw");
        if(precision_of(u, "u") != precision)
            throw std::invalid_argument("u must have the same dtype as raw");

        auto const n = static_cast<std::int64_t>(raw.shape(0));
        if(u.ndim() != 1 || u.shape(0) != n)
            throw std::invalid_argument("u must have shape (n,) matching raw.shape[0]");

        mango::SplineSpec spec;
        spec.n = n;
        spec.n_bins = check_raw(raw, n_bins);
        spec.precision = precision;

        Outputs out{make_output(precision, n), make_output(precision, n), precision};
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
    Outputs inverse_impl(py::array raw, py::array x, std::string const& backend, int n_bins)
    {
        require_contiguous(raw, "raw");
        require_contiguous(x, "x");
        auto const precision = precision_of(raw, "raw");
        if(precision_of(x, "x") != precision)
            throw std::invalid_argument("x must have the same dtype as raw");

        auto const n = static_cast<std::int64_t>(raw.shape(0));
        if(x.ndim() != 1 || x.shape(0) != n)
            throw std::invalid_argument("x must have shape (n,) matching raw.shape[0]");

        mango::SplineSpec spec;
        spec.n = n;
        spec.n_bins = check_raw(raw, n_bins);
        spec.precision = precision;

        Outputs out{make_output(precision, n), make_output(precision, n), precision};
        mango::cubic_spline_inverse(
            parse_backend(backend),
            raw.data(),
            x.data(),
            out.value.mutable_data(),
            out.logabsdet.mutable_data(),
            spec);
        return out;
    }

    void negate_inplace(py::array& array, mango::Precision precision)
    {
        auto const count = array.size();
        if(precision == mango::Precision::Float64)
        {
            auto* data = static_cast<double*>(array.mutable_data());
            for(py::ssize_t i = 0; i < count; ++i)
                data[i] = -data[i];
        }
        else
        {
            auto* data = static_cast<float*>(array.mutable_data());
            for(py::ssize_t i = 0; i < count; ++i)
                data[i] = -data[i];
        }
    }

} // namespace

PYBIND11_MODULE(mango_alpaka_ext, module)
{
    module.doc() = "MANGO Alpaka kernels: monotone cubic-Hermite marginal spline.";

    module.def(
        "forward",
        [](py::array raw, py::array u, std::string const& backend, int n_bins) {
            auto result = forward_impl(raw, u, backend, n_bins);
            return py::make_tuple(result.value, result.logabsdet);
        },
        py::arg("raw"),
        py::arg("u"),
        py::arg("backend") = "cuda",
        py::arg("n_bins") = 0,
        R"doc(Forward map u -> x through the conditional cubic-Hermite spline.

raw : (n, 2*n_bins), float32 or float64, [width logits | height logits]
u   : (n,) same dtype, in [0, 1]
Returns (x, logabsdet) where logabsdet = log|dx/du|.
)doc");

    module.def(
        "inverse",
        [](py::array raw, py::array x, std::string const& backend, int n_bins) {
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
        [](py::array raw, py::array u, std::string const& backend, int n_bins) {
            return forward_impl(raw, u, backend, n_bins).value;
        },
        py::arg("raw"),
        py::arg("u"),
        py::arg("backend") = "cuda",
        py::arg("n_bins") = 0,
        "Contract: draw from the flow by mapping base samples u -> x.");

    module.def(
        "log_density",
        [](py::array raw, py::array x, std::string const& backend, int n_bins) {
            // Uniform base: log q(x) = -log|dx/du|, evaluated through the inverse.
            auto result = inverse_impl(raw, x, backend, n_bins);
            negate_inplace(result.logabsdet, result.precision);
            return result.logabsdet;
        },
        py::arg("raw"),
        py::arg("x"),
        py::arg("backend") = "cuda",
        py::arg("n_bins") = 0,
        "Contract: log q(x) for a uniform base distribution.");

    module.def(
        "sample_and_log_prob",
        [](py::array raw, py::array u, std::string const& backend, int n_bins) {
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

    module.def("precisions", []() {
        std::vector<std::string> enabled;
        if(mango::precision_available(mango::Precision::Float64))
            enabled.emplace_back("float64");
        if(mango::precision_available(mango::Precision::Float32))
            enabled.emplace_back("float32");
        return enabled;
    });
}
