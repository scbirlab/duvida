# Benchmarks

`bench_hessian_diagonal.py` compares the `exact_diagonal` methods (`forward` =
nested forward-mode, `hvp` = Hessian-vector products) across backends,
parameter counts and output sizes, with peak memory (max RSS) per run:

```bash
python benchmarks/bench_hessian_diagonal.py --backends torch jax \
    --params 100 1000 10000 --outputs scalar loss64 loss512 8 64
```

`outputs` is `scalar` (loss on one data point), `lossN` (scalar loss summed over N
data points, as in `fisher_information_diagonal`), or an integer m (a model
evaluated at m data points, as in `parameter_hessian_diagonal`).

`results_2core_cpu_float64.csv` was produced on a 2-core, 7 GB CPU machine in
float64 (torch 2.15, jax 0.12). Absolute times will differ on your hardware; the
ratios are what matter. Summary (seconds for `hvp` / `forward`, higher ratio = forward faster):

| Workload | torch | jax |
| --- | --- | --- |
| vector output, m = 8 | 2.6x - 6.8x | 4.2x - 11.8x |
| vector output, m = 64 | 15x - 22x | 16x - 24x |
| scalar loss, 1 data point | 1.0x - 2.3x | 1.8x - 3.2x |
| scalar loss, 64-512 data points | 0.3x - 0.7x (hvp faster) | 0.9x - 1.4x |

`method="auto"` follows these results: `forward` for vector outputs on both backends
and for scalar outputs on JAX; `hvp` for scalar outputs on PyTorch.
Memory for the two methods is similar once chunking is active (see the
`peak_rss_gb` column).
