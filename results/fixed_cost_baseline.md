# Fixed-cost baseline — PyTorch marginal spline

Measured by `tests/benchmark_fixed_cost.py`. Times are the best of 20 repeats after 5 warmup calls, with `torch.cuda.synchronize()` around every CUDA measurement.

A bank of `dim` independent `ConditionalCubicFlow1D` marginals is evaluated per call, matching the pipeline's `MarginalBank`.

| dim | device | N | sample (ms) | sample (µs/point) | log_density (ms) | log_density (µs/point) |
|---:|---|---:|---:|---:|---:|---:|
| 2 | cpu | 1 | 0.942 | 471.05 | 13.064 | 6532.05 |
| 2 | cpu | 100 | 1.627 | 8.13 | 18.971 | 94.86 |
| 2 | cpu | 512 | 2.401 | 2.34 | 21.310 | 20.81 |
| 2 | cpu | 1,000 | 3.471 | 1.74 | 22.576 | 11.29 |
| 2 | cpu | 10,000 | 13.342 | 0.67 | 40.340 | 2.02 |
| 2 | cpu | 50,000 | 78.650 | 0.79 | 127.342 | 1.27 |
| 5 | cpu | 1 | 3.119 | 623.80 | 34.235 | 6847.04 |
| 5 | cpu | 100 | 4.056 | 8.11 | 49.693 | 99.39 |
| 5 | cpu | 512 | 6.271 | 2.45 | 54.098 | 21.13 |
| 5 | cpu | 1,000 | 8.369 | 1.67 | 59.409 | 11.88 |
| 5 | cpu | 10,000 | 32.739 | 0.65 | 101.509 | 2.03 |
| 5 | cpu | 50,000 | 192.739 | 0.77 | 320.489 | 1.28 |
| 8 | cpu | 1 | 4.401 | 550.18 | 54.257 | 6782.12 |
| 8 | cpu | 100 | 6.554 | 8.19 | 79.905 | 99.88 |
| 8 | cpu | 512 | 10.278 | 2.51 | 86.290 | 21.07 |
| 8 | cpu | 1,000 | 14.057 | 1.76 | 93.855 | 11.73 |
| 8 | cpu | 10,000 | 52.741 | 0.66 | 160.052 | 2.00 |
| 8 | cpu | 50,000 | 315.532 | 0.79 | 512.895 | 1.28 |
| 2 | cuda | 1 | 3.684 | 1841.90 | 69.785 | 34892.65 |
| 2 | cuda | 100 | 3.803 | 19.02 | 65.871 | 329.35 |
| 2 | cuda | 512 | 3.872 | 3.78 | 66.054 | 64.51 |
| 2 | cuda | 1,000 | 3.732 | 1.87 | 66.455 | 33.23 |
| 2 | cuda | 10,000 | 3.703 | 0.19 | 68.692 | 3.43 |
| 2 | cuda | 50,000 | 13.108 | 0.13 | 78.849 | 0.79 |
| 5 | cuda | 1 | 9.278 | 1855.56 | 164.878 | 32975.60 |
| 5 | cuda | 100 | 9.894 | 19.79 | 168.809 | 337.62 |
| 5 | cuda | 512 | 9.306 | 3.64 | 165.550 | 64.67 |
| 5 | cuda | 1,000 | 9.094 | 1.82 | 165.083 | 33.02 |
| 5 | cuda | 10,000 | 9.193 | 0.18 | 161.335 | 3.23 |
| 5 | cuda | 50,000 | 35.200 | 0.14 | 177.909 | 0.71 |
| 8 | cuda | 1 | 15.332 | 1916.49 | 268.266 | 33533.20 |
| 8 | cuda | 100 | 15.593 | 19.49 | 265.974 | 332.47 |
| 8 | cuda | 512 | 15.824 | 3.86 | 271.212 | 66.21 |
| 8 | cuda | 1,000 | 14.947 | 1.87 | 265.040 | 33.13 |
| 8 | cuda | 10,000 | 15.700 | 0.20 | 275.553 | 3.44 |
| 8 | cuda | 50,000 | 55.490 | 0.14 | 286.245 | 0.72 |

## Fixed cost

Milliseconds at the smallest batch, where arithmetic is negligible:

```
   cpu dim=2  0.94 ms at N=1  ->  78.7 ms at N=50,000   (83.5x for 50000x the points)
   cpu dim=5  3.12 ms at N=1  ->  192.7 ms at N=50,000   (61.8x for 50000x the points)
   cpu dim=8  4.40 ms at N=1  ->  315.5 ms at N=50,000   (71.7x for 50000x the points)
  cuda dim=2  3.68 ms at N=1  ->  13.1 ms at N=50,000   (3.6x for 50000x the points)
  cuda dim=5  9.28 ms at N=1  ->  35.2 ms at N=50,000   (3.8x for 50000x the points)
  cuda dim=8  15.33 ms at N=1  ->  55.5 ms at N=50,000   (3.6x for 50000x the points)
   cpu dim=2  13.06 ms at N=1  ->  127.3 ms at N=50,000   (9.7x for 50000x the points)
   cpu dim=5  34.24 ms at N=1  ->  320.5 ms at N=50,000   (9.4x for 50000x the points)
   cpu dim=8  54.26 ms at N=1  ->  512.9 ms at N=50,000   (9.5x for 50000x the points)
  cuda dim=2  69.79 ms at N=1  ->  78.8 ms at N=50,000   (1.1x for 50000x the points)
  cuda dim=5  164.88 ms at N=1  ->  177.9 ms at N=50,000   (1.1x for 50000x the points)
  cuda dim=8  268.27 ms at N=1  ->  286.2 ms at N=50,000   (1.1x for 50000x the points)
```
