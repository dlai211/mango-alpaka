// cubic_spline_1d.hpp — Deliverable B.
//
// Fused Alpaka kernel for the conditional monotone cubic-Hermite marginal of
// nf/conditional_cubic_1d.py in the upstream conditional-nf-integration repo.
//
// One thread evaluates one element end to end: softmax -> bin widths/heights ->
// cumulative knots -> PCHIP slopes -> bin search -> Hermite evaluation. There
// is no intermediate state in global memory, which is the whole point: the
// PyTorch reference issues a few dozen small kernels per call, and at the batch
// sizes this pipeline uses the launch overhead dominates the arithmetic.
//
// MATHEMATICAL CONTRACT (must match the Python exactly):
//
//   widths[i]  = min_w + (1 - N*min_w) * softmax(raw_w)[i]
//   heights[i] = min_h + (1 - N*min_h) * softmax(raw_h)[i]
//   cumw[0] = 0; cumw[i+1] = cumw[i] + widths[i]; cumw[N] = 1   (forced)
//   cumh likewise.
//
// PCHIP knot slopes (n_bins == 1 is the degenerate case):
//   secant[i] = heights[i] / widths[i]
//   d[0] = secant[0];  d[N] = secant[N-1]
//   d[k] = (w1 + w2) / (w1/secant[k-1] + w2/secant[k])
//          w1 = 2*widths[k] + widths[k-1]
//          w2 = widths[k] + 2*widths[k-1]
//
// Hermite (t in [0,1] within a bin):
//   x(t)    = (2t^3 - 3t^2 + 1) y0 + (t^3 - 2t^2 + t) w d0
//           + (-2t^3 + 3t^2)(y0 + h) + (t^3 - t^2) w d1
//   dx/dt   = (6t^2 - 6t) y0 + (3t^2 - 4t + 1) w d0
//           + (-6t^2 + 6t)(y0 + h) + (3t^2 - 2t) w d1
//   log|det| = log(max(dx/dt, tiny)) - log(w)
//
// Bin index: count of (value >= knot[j]) for j = 1..N, clamped to N-1.

#pragma once

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

namespace mango::kernel
{

    //! Smallest positive normal, matching torch.finfo(dtype).tiny.
    template<typename T>
    ALPAKA_FN_ACC inline T tiny()
    {
        return std::numeric_limits<T>::min();
    }

    template<typename T>
    ALPAKA_FN_ACC inline T clamp01(T v)
    {
        return v < T(0) ? T(0) : (v > T(1) ? T(1) : v);
    }

    //! widths[i] = min_bin + (1 - N*min_bin) * softmax(logits)[i]
    template<typename T, int N_BINS>
    ALPAKA_FN_ACC inline void scaled_softmax(T const* logits, T min_bin, T* out)
    {
        T max_logit = logits[0];
        for(int i = 1; i < N_BINS; ++i)
            max_logit = logits[i] > max_logit ? logits[i] : max_logit;

        T sum = T(0);
        for(int i = 0; i < N_BINS; ++i)
            sum += std::exp(logits[i] - max_logit);

        T const room = T(1) - T(N_BINS) * min_bin;
        for(int i = 0; i < N_BINS; ++i)
            out[i] = min_bin + room * (std::exp(logits[i] - max_logit) / sum);
    }

    //! Cumulative sum with the right boundary forced to 1, as the reference does
    //! to avoid softmax roundoff accumulating at the last knot.
    template<typename T, int N_BINS>
    ALPAKA_FN_ACC inline void cumulative(T const* bins, T* cum)
    {
        cum[0] = T(0);
        for(int i = 0; i < N_BINS; ++i)
            cum[i + 1] = cum[i] + bins[i];
        cum[N_BINS] = T(1);
    }

    //! PCHIP knot slopes; positive secants give positive slopes, hence monotonicity.
    template<typename T, int N_BINS>
    ALPAKA_FN_ACC inline void knot_derivatives(T const* width, T const* height, T* d)
    {
        T secant[N_BINS];
        for(int i = 0; i < N_BINS; ++i)
            secant[i] = height[i] / width[i];

        if constexpr(N_BINS == 1)
        {
            d[0] = secant[0];
            d[1] = secant[0];
        }
        else
        {
            d[0] = secant[0];
            d[N_BINS] = secant[N_BINS - 1];
            for(int k = 1; k < N_BINS; ++k)
            {
                T const left = width[k - 1];
                T const right = width[k];
                T const w1 = T(2) * right + left;
                T const w2 = right + T(2) * left;
                d[k] = (w1 + w2) / (w1 / secant[k - 1] + w2 / secant[k]);
            }
        }
    }

    template<typename T>
    ALPAKA_FN_ACC inline void hermite(
        T t,
        T y0,
        T height,
        T width,
        T d0,
        T d1,
        T& x,
        T& dxdt)
    {
        T const t2 = t * t;
        T const t3 = t2 * t;
        x = (T(2) * t3 - T(3) * t2 + T(1)) * y0 + (t3 - T(2) * t2 + t) * width * d0
            + (-T(2) * t3 + T(3) * t2) * (y0 + height) + (t3 - t2) * width * d1;
        dxdt = (T(6) * t2 - T(6) * t) * y0 + (T(3) * t2 - T(4) * t + T(1)) * width * d0
               + (-T(6) * t2 + T(6) * t) * (y0 + height) + (T(3) * t2 - T(2) * t) * width * d1;
    }

    //! Index of the bin containing `value` given knots[0..N_BINS].
    template<typename T, int N_BINS>
    ALPAKA_FN_ACC inline int bin_index(T value, T const* knots)
    {
        int idx = 0;
        for(int j = 1; j <= N_BINS; ++j)
            idx += (value >= knots[j]) ? 1 : 0;
        return idx > N_BINS - 1 ? N_BINS - 1 : idx;
    }

    //! Per-element parameter preparation shared by the two directions.
    template<typename T, int N_BINS>
    ALPAKA_FN_ACC inline void prepare(
        T const* raw,
        T min_bin_width,
        T min_bin_height,
        T* width,
        T* height,
        T* cumw,
        T* cumh,
        T* d)
    {
        scaled_softmax<T, N_BINS>(raw, min_bin_width, width);
        scaled_softmax<T, N_BINS>(raw + N_BINS, min_bin_height, height);
        cumulative<T, N_BINS>(width, cumw);
        cumulative<T, N_BINS>(height, cumh);
        knot_derivatives<T, N_BINS>(width, height, d);
    }

    //! Forward: u -> x, log|dx/du|.
    template<typename T, typename IdxT, int N_BINS>
    struct ForwardKernel
    {
        T const* raw;
        T const* u;
        T* x;
        T* logabsdet;
        IdxT n;
        T min_bin_width;
        T min_bin_height;

        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc) const
        {
            for(auto i : alpaka::uniformElements(acc, n))
            {
                auto const offset = static_cast<std::int64_t>(i) * (2 * N_BINS);
                T const* element_raw = raw + offset;

                T width[N_BINS];
                T height[N_BINS];
                T cumw[N_BINS + 1];
                T cumh[N_BINS + 1];
                T d[N_BINS + 1];
                prepare<T, N_BINS>(
                    element_raw,
                    min_bin_width,
                    min_bin_height,
                    width,
                    height,
                    cumw,
                    cumh,
                    d);

                T const value = u[i];
                int const idx = bin_index<T, N_BINS>(value, cumw);
                T const w = width[idx];
                T const h = height[idx];
                T const y0 = cumh[idx];

                T const t = clamp01((value - cumw[idx]) / w);

                T xv = T(0);
                T dxdt = T(0);
                hermite(t, y0, h, w, d[idx], d[idx + 1], xv, dxdt);

                x[i] = xv;
                logabsdet[i] = std::log(dxdt > tiny<T>() ? dxdt : tiny<T>()) - std::log(w);
            }
        }
    };

    //! Inverse: x -> u by bisection, one Newton correction, then log|dx/du|.
    template<typename T, typename IdxT, int N_BINS>
    struct InverseKernel
    {
        T const* raw;
        T const* x;
        T* u;
        T* logabsdet;
        IdxT n;
        T min_bin_width;
        T min_bin_height;
        int inverse_iterations;

        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc) const
        {
            for(auto i : alpaka::uniformElements(acc, n))
            {
                auto const offset = static_cast<std::int64_t>(i) * (2 * N_BINS);
                T const* element_raw = raw + offset;

                T width[N_BINS];
                T height[N_BINS];
                T cumw[N_BINS + 1];
                T cumh[N_BINS + 1];
                T d[N_BINS + 1];
                prepare<T, N_BINS>(
                    element_raw,
                    min_bin_width,
                    min_bin_height,
                    width,
                    height,
                    cumw,
                    cumh,
                    d);

                T const target = x[i];
                int const idx = bin_index<T, N_BINS>(target, cumh);
                T const w = width[idx];
                T const h = height[idx];
                T const y0 = cumh[idx];
                T const u0 = cumw[idx];

                // Bisection on t in [0, 1] — globally safe, no derivative needed.
                T low = T(0);
                T high = T(1);
                for(int iteration = 0; iteration < inverse_iterations; ++iteration)
                {
                    T const mid = (low + high) * T(0.5);
                    T xmid = T(0);
                    T dxmid = T(0);
                    hermite(mid, y0, h, w, d[idx], d[idx + 1], xmid, dxmid);
                    if(xmid < target)
                        low = mid;
                    else
                        high = mid;
                }

                T t = (low + high) * T(0.5);

                // One Newton correction so the result follows the implicit root.
                T xt = T(0);
                T dxt = T(0);
                hermite(t, y0, h, w, d[idx], d[idx + 1], xt, dxt);
                T const safe_dx = dxt > tiny<T>() ? dxt : tiny<T>();
                t = clamp01(t - (xt - target) / safe_dx);

                T dummy = T(0);
                T dxdt = T(0);
                hermite(t, y0, h, w, d[idx], d[idx + 1], dummy, dxdt);

                T mapped = u0 + w * t;
                u[i] = clamp01(mapped);
                logabsdet[i] = std::log(dxdt > tiny<T>() ? dxdt : tiny<T>()) - std::log(w);
            }
        }
    };

} // namespace mango::kernel
