// launcher.cu — instantiates the Alpaka kernels and exposes the plain C++ API
// declared in include/mango_interface.hpp.
//
// Compiled by nvcc (it includes Alpaka's CUDA headers). The pybind11 module is
// built separately from src/bindings.cpp by MSVC, so the two worlds never meet
// in one translation unit.

#include "mango_interface.hpp"

#include <alpaka/alpaka.hpp>

#include "kernels/cubic_spline_1d.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace mango
{
    namespace
    {
        using Dim = alpaka::DimInt<1u>;
        using IdxT = std::size_t;

        //! Run one batch on one backend. Shared by forward and inverse.
        template<typename TAcc, int N_BINS, bool INVERSE>
        void run_backend(
            const double* raw,
            const double* values,
            double* out_value,
            double* out_logabsdet,
            const SplineSpec& spec)
        {
            IdxT const n = static_cast<IdxT>(spec.n);
            if(n == 0)
                return;

            auto const platform = alpaka::Platform<TAcc>{};
            auto const dev = alpaka::getDevByIdx(platform, 0);
            using QueueAcc = alpaka::Queue<TAcc, alpaka::Blocking>;
            QueueAcc queue(dev);

            using DevHost = alpaka::DevCpu;
            auto const platformHost = alpaka::PlatformCpu{};
            auto const devHost = alpaka::getDevByIdx(platformHost, 0);

            auto const extent = alpaka::Vec<Dim, IdxT>{n};
            auto const extentRaw = alpaka::Vec<Dim, IdxT>{n * IdxT(2 * N_BINS)};

            auto rawHost = alpaka::allocBuf<double, IdxT>(devHost, extentRaw);
            auto valueHost = alpaka::allocBuf<double, IdxT>(devHost, extent);
            auto outHost = alpaka::allocBuf<double, IdxT>(devHost, extent);
            auto logHost = alpaka::allocBuf<double, IdxT>(devHost, extent);

            std::memcpy(
                alpaka::getPtrNative(rawHost),
                raw,
                sizeof(double) * static_cast<std::size_t>(n) * 2 * N_BINS);
            std::memcpy(alpaka::getPtrNative(valueHost), values, sizeof(double) * n);

            auto rawDev = alpaka::allocBuf<double, IdxT>(dev, extentRaw);
            auto valueDev = alpaka::allocBuf<double, IdxT>(dev, extent);
            auto outDev = alpaka::allocBuf<double, IdxT>(dev, extent);
            auto logDev = alpaka::allocBuf<double, IdxT>(dev, extent);

            alpaka::memcpy(queue, rawDev, rawHost);
            alpaka::memcpy(queue, valueDev, valueHost);

            auto const elementsPerThread = alpaka::Vec<Dim, IdxT>::ones();
            alpaka::KernelCfg<TAcc> const kernelCfg = {extent, elementsPerThread};

            if constexpr(INVERSE)
            {
                kernel::InverseKernel<double, IdxT, N_BINS> const kernel{
                    alpaka::getPtrNative(rawDev),
                    alpaka::getPtrNative(valueDev),
                    alpaka::getPtrNative(outDev),
                    alpaka::getPtrNative(logDev),
                    n,
                    spec.min_bin_width,
                    spec.min_bin_height,
                    spec.inverse_iterations};
                auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, dev, kernel);
                alpaka::exec<TAcc>(queue, workDiv, kernel);
            }
            else
            {
                kernel::ForwardKernel<double, IdxT, N_BINS> const kernel{
                    alpaka::getPtrNative(rawDev),
                    alpaka::getPtrNative(valueDev),
                    alpaka::getPtrNative(outDev),
                    alpaka::getPtrNative(logDev),
                    n,
                    spec.min_bin_width,
                    spec.min_bin_height};
                auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, dev, kernel);
                alpaka::exec<TAcc>(queue, workDiv, kernel);
            }

            alpaka::memcpy(queue, outHost, outDev);
            alpaka::memcpy(queue, logHost, logDev);
            alpaka::wait(queue);

            std::memcpy(out_value, alpaka::getPtrNative(outHost), sizeof(double) * n);
            std::memcpy(out_logabsdet, alpaka::getPtrNative(logHost), sizeof(double) * n);
        }

        //! Dispatch on the spline bin count, which is a compile-time constant in
        //! the kernel. The reference model uses 32; the others exist for tests.
        template<typename TAcc, bool INVERSE>
        void dispatch_bins(
            const double* raw,
            const double* values,
            double* out_value,
            double* out_logabsdet,
            const SplineSpec& spec)
        {
            switch(spec.n_bins)
            {
            case 8:
                run_backend<TAcc, 8, INVERSE>(raw, values, out_value, out_logabsdet, spec);
                break;
            case 16:
                run_backend<TAcc, 16, INVERSE>(raw, values, out_value, out_logabsdet, spec);
                break;
            case 32:
                run_backend<TAcc, 32, INVERSE>(raw, values, out_value, out_logabsdet, spec);
                break;
            case 64:
                run_backend<TAcc, 64, INVERSE>(raw, values, out_value, out_logabsdet, spec);
                break;
            default:
                throw std::invalid_argument(
                    "unsupported n_bins: " + std::to_string(spec.n_bins)
                    + " (supported: 8, 16, 32, 64)");
            }
        }

        template<bool INVERSE>
        void dispatch_backend(
            Backend backend,
            const double* raw,
            const double* values,
            double* out_value,
            double* out_logabsdet,
            const SplineSpec& spec)
        {
            using AccCpu = alpaka::AccCpuSerial<Dim, IdxT>;
            switch(backend)
            {
            case Backend::CpuSerial:
                dispatch_bins<AccCpu, INVERSE>(raw, values, out_value, out_logabsdet, spec);
                return;
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
            case Backend::CpuOmp2Blocks:
                dispatch_bins<alpaka::AccCpuOmp2Blocks<Dim, IdxT>, INVERSE>(
                    raw,
                    values,
                    out_value,
                    out_logabsdet,
                    spec);
                return;
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
            case Backend::Cuda:
                dispatch_bins<alpaka::AccGpuCudaRt<Dim, IdxT>, INVERSE>(
                    raw,
                    values,
                    out_value,
                    out_logabsdet,
                    spec);
                return;
#endif
            default:
                throw std::invalid_argument(
                    std::string("backend not enabled in this build: ") + backend_name(backend));
            }
        }

    } // namespace

    void cubic_spline_forward(
        Backend backend,
        const double* raw,
        const double* u,
        double* x,
        double* logabsdet,
        const SplineSpec& spec)
    {
        dispatch_backend<false>(backend, raw, u, x, logabsdet, spec);
    }

    void cubic_spline_inverse(
        Backend backend,
        const double* raw,
        const double* x,
        double* u,
        double* logabsdet,
        const SplineSpec& spec)
    {
        dispatch_backend<true>(backend, raw, x, u, logabsdet, spec);
    }

    char const* backend_name(Backend backend)
    {
        switch(backend)
        {
        case Backend::CpuSerial:
            return "cpu_serial";
        case Backend::CpuOmp2Blocks:
            return "cpu_omp2_blocks";
        case Backend::Cuda:
            return "cuda";
        }
        return "unknown";
    }

    bool backend_available(Backend backend)
    {
        switch(backend)
        {
        case Backend::CpuSerial:
            return true;
        case Backend::CpuOmp2Blocks:
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
            return true;
#else
            return false;
#endif
        case Backend::Cuda:
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
            return true;
#else
            return false;
#endif
        }
        return false;
    }

} // namespace mango
