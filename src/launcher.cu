// launcher.cu — instantiates the Alpaka kernels and exposes the plain C++ API
// declared in include/mango_interface.hpp.
//
// Compiled by nvcc (it includes Alpaka's CUDA headers). The pybind11 module is
// built separately from src/bindings.cpp by MSVC, so the two worlds never meet
// in one translation unit.
//
// Every kernel is instantiated for both float64 and float32. float64 is the
// parity reference; float32 is what the pipeline trains and evaluates in, and
// on a consumer GPU it runs at 32x the float64 rate.

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

        //! Run one batch on one backend in one precision.
        template<typename TAcc, typename T, int N_BINS, bool INVERSE>
        void run_backend(
            const void* raw,
            const void* values,
            void* out_value,
            void* out_logabsdet,
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
            auto const extentBoth = alpaka::Vec<Dim, IdxT>{n * IdxT(2)};

            // Views over the caller's memory instead of host buffers: this
            // removes four allocations and two memcpys per call, which matters
            // at the batch sizes the pipeline uses.
            using ViewHost = alpaka::ViewPlainPtr<DevHost, T, Dim, IdxT>;
            ViewHost rawHost(const_cast<T*>(static_cast<const T*>(raw)), devHost, extentRaw);
            ViewHost valueHost(const_cast<T*>(static_cast<const T*>(values)), devHost, extent);
            ViewHost outHost(static_cast<T*>(out_value), devHost, extent);
            ViewHost logHost(static_cast<T*>(out_logabsdet), devHost, extent);

            auto rawDev = alpaka::allocBuf<T, IdxT>(dev, extentRaw);
            auto valueDev = alpaka::allocBuf<T, IdxT>(dev, extent);
            auto outDev = alpaka::allocBuf<T, IdxT>(dev, extent);
            auto logDev = alpaka::allocBuf<T, IdxT>(dev, extent);

            alpaka::memcpy(queue, rawDev, rawHost);
            alpaka::memcpy(queue, valueDev, valueHost);

            auto const elementsPerThread = alpaka::Vec<Dim, IdxT>::ones();
            alpaka::KernelCfg<TAcc> const kernelCfg = {extent, elementsPerThread};

            if constexpr(INVERSE)
            {
                kernel::InverseKernel<T, IdxT, N_BINS> const kernel{
                    alpaka::getPtrNative(rawDev),
                    alpaka::getPtrNative(valueDev),
                    alpaka::getPtrNative(outDev),
                    alpaka::getPtrNative(logDev),
                    n,
                    static_cast<T>(spec.min_bin_width),
                    static_cast<T>(spec.min_bin_height),
                    spec.inverse_iterations};
                auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, dev, kernel);
                alpaka::exec<TAcc>(queue, workDiv, kernel);
            }
            else
            {
                kernel::ForwardKernel<T, IdxT, N_BINS> const kernel{
                    alpaka::getPtrNative(rawDev),
                    alpaka::getPtrNative(valueDev),
                    alpaka::getPtrNative(outDev),
                    alpaka::getPtrNative(logDev),
                    n,
                    static_cast<T>(spec.min_bin_width),
                    static_cast<T>(spec.min_bin_height)};
                auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, dev, kernel);
                alpaka::exec<TAcc>(queue, workDiv, kernel);
            }

            alpaka::memcpy(queue, outHost, outDev);
            alpaka::memcpy(queue, logHost, logDev);
            alpaka::wait(queue);
            // outHost/logHost are views over the caller's buffers, so the
            // device-to-host copies above are the last step.
        }

        //! Bin count is a template parameter because the kernel uses fixed-size
        //! per-thread arrays. The reference model uses 32.
        template<typename TAcc, typename T, bool INVERSE>
        void dispatch_bins(
            const void* raw,
            const void* values,
            void* out_value,
            void* out_logabsdet,
            const SplineSpec& spec)
        {
            switch(spec.n_bins)
            {
            case 8:
                run_backend<TAcc, T, 8, INVERSE>(raw, values, out_value, out_logabsdet, spec);
                break;
            case 16:
                run_backend<TAcc, T, 16, INVERSE>(raw, values, out_value, out_logabsdet, spec);
                break;
            case 32:
                run_backend<TAcc, T, 32, INVERSE>(raw, values, out_value, out_logabsdet, spec);
                break;
            case 64:
                run_backend<TAcc, T, 64, INVERSE>(raw, values, out_value, out_logabsdet, spec);
                break;
            default:
                throw std::invalid_argument(
                    "unsupported n_bins: " + std::to_string(spec.n_bins)
                    + " (supported: 8, 16, 32, 64)");
            }
        }

        template<typename TAcc, bool INVERSE>
        void dispatch_precision(
            const void* raw,
            const void* values,
            void* out_value,
            void* out_logabsdet,
            const SplineSpec& spec)
        {
            switch(spec.precision)
            {
            case Precision::Float64:
                dispatch_bins<TAcc, double, INVERSE>(raw, values, out_value, out_logabsdet, spec);
                break;
            case Precision::Float32:
                dispatch_bins<TAcc, float, INVERSE>(raw, values, out_value, out_logabsdet, spec);
                break;
            default:
                throw std::invalid_argument("unsupported precision");
            }
        }

        template<bool INVERSE>
        void dispatch_backend(
            Backend backend,
            const void* raw,
            const void* values,
            void* out_value,
            void* out_logabsdet,
            const SplineSpec& spec)
        {
            using AccCpu = alpaka::AccCpuSerial<Dim, IdxT>;
            switch(backend)
            {
            case Backend::CpuSerial:
                dispatch_precision<AccCpu, INVERSE>(raw, values, out_value, out_logabsdet, spec);
                return;
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
            case Backend::CpuOmp2Blocks:
                dispatch_precision<alpaka::AccCpuOmp2Blocks<Dim, IdxT>, INVERSE>(
                    raw, values, out_value, out_logabsdet, spec);
                return;
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
            case Backend::Cuda:
                dispatch_precision<alpaka::AccGpuCudaRt<Dim, IdxT>, INVERSE>(
                    raw, values, out_value, out_logabsdet, spec);
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
        const void* raw,
        const void* u,
        void* x,
        void* logabsdet,
        const SplineSpec& spec)
    {
        dispatch_backend<false>(backend, raw, u, x, logabsdet, spec);
    }

    void cubic_spline_inverse(
        Backend backend,
        const void* raw,
        const void* x,
        void* u,
        void* logabsdet,
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

    bool precision_available(Precision precision)
    {
        return precision == Precision::Float64 || precision == Precision::Float32;
    }

} // namespace mango
