"""Benchmark `exact_diagonal` methods across backends, parameter counts and output sizes.

Each configuration runs in its own subprocess so peak memory (max RSS) is
attributable to that run alone. Usage::

    python benchmarks/bench_hessian_diagonal.py --backends torch jax \
        --params 100 1000 10000 --outputs scalar 8 64

"outputs" is the number of model outputs: `scalar` differentiates a scalar loss on one
data point and `lossN` a scalar loss summed over N data points (as in
`fisher_information_diagonal`); an integer m differentiates a model evaluated at m 
data points (as in `parameter_hessian_diagonal`).
A configuration is skipped when its projected time (assuming time ~ p**2 from the
last measured p) exceeds `--max-seconds`.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time


def worker(args):
    # Put the requested checkout first on sys.path. PYTHONPATH is not reliable here:
    # an installed (e.g. editable) duvida can take precedence over it, and `-I`
    # ignores it entirely.
    sys.path.insert(0, os.path.abspath(args.repo))
    os.environ["DUVIDA_BACKEND"] = args.backend
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    if args.backend == "jax":
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        normal = lambda k, s: jax.random.normal(jax.random.key(k), s)
        block = jax.block_until_ready
        tanh, to_scalar = jnp.tanh, float
    else:
        import torch
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        normal = lambda k, s: torch.randn(*s)
        block = lambda x: x
        tanh, to_scalar = torch.tanh, float
    import duvida
    from duvida.hessians import exact_diagonal
    import duvida.numpy as dnp
    if not os.path.abspath(duvida.__file__).startswith(os.path.abspath(args.repo) + os.sep):
        raise ImportError(
            f"Imported duvida from {duvida.__file__}, not from the requested checkout {args.repo}"
        )

    d_in = 8
    h = max(2, round((args.params) ** .5))
    params = (normal(1, (d_in, h)) / d_in ** .5, normal(2, (h, h)) / h ** .5, normal(3, (h,)) / h ** .5)
    p = sum(int(dnp.get_array_size(q)) for q in params)
    is_loss = args.outputs == "scalar" or args.outputs.startswith("loss")
    n = (1 if args.outputs == "scalar" else int(args.outputs[4:])) if is_loss else int(args.outputs)
    x = normal(4, (n, d_in))

    def model(prm):
        w1, w2, w3 = prm
        return tanh(tanh(x @ w1) @ w2) @ w3     # (n,)

    f = (lambda prm: dnp.sum(model(prm) ** 2)) if is_loss else model
    fn = exact_diagonal(f, method=args.method)
    block(fn(params))                            # warm-up / compile
    t0 = time.perf_counter(); block(fn(params)); t = time.perf_counter() - t0
    t0 = time.perf_counter(); block(f(params)); t_eval = time.perf_counter() - t0
    print(json.dumps(dict(
        backend=args.backend, p=p, outputs=args.outputs, method=args.method,
        duvida=os.path.dirname(duvida.__file__),
        seconds=t, one_eval_seconds=t_eval,
        peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    )))


def orchestrate(args):
    rows = []
    for backend in args.backends:
        for outputs in args.outputs:
            for method in ("forward", "hvp"):
                last = None  # (p, seconds)
                for params in sorted(args.params):
                    if last and last[1] * (params / last[0]) ** 2 > args.max_seconds:
                        print(f"skip {backend} {outputs} {method} p~{params} (projected too slow)", file=sys.stderr)
                        continue
                    cmd = [sys.executable, "-I", os.path.abspath(__file__), "--worker",
                           "--backend", backend, "--params", str(params),
                           "--outputs", str(outputs), "--method", method,
                           "--repo", args.repo]
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.max_seconds * 4)
                    if r.returncode != 0:
                        print(f"FAILED {backend} {outputs} {method} p~{params}: {r.stderr.strip().splitlines()[-1:]}", file=sys.stderr)
                        rows.append(dict(backend=backend, p=params, outputs=outputs, method=method, seconds=None, peak_rss_gb=None))
                        continue
                    row = json.loads(r.stdout.strip().splitlines()[-1])
                    rows.append(row); last = (row["p"], row["seconds"])
    if rows and any(r.get("duvida") for r in rows):
        print("# duvida imported from:", sorted({r["duvida"] for r in rows if r.get("duvida")}), file=sys.stderr)
    print("backend,outputs,p,method,seconds,one_eval_seconds,peak_rss_gb")
    for r in rows:
        print(",".join(str(r.get(k)) if r.get(k) is None or isinstance(r.get(k), (str, int)) else f"{r[k]:.4g}"
                       for k in ("backend", "outputs", "p", "method", "seconds", "one_eval_seconds", "peak_rss_gb")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--backends", nargs="+", default=["torch", "jax"])
    ap.add_argument("--params", nargs="+", type=int, default=[100, 1000, 10000])
    ap.add_argument("--outputs", nargs="+", default=["scalar", "8", "64"])
    ap.add_argument("--method", default="forward")
    ap.add_argument("--max-seconds", type=float, default=60.)
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    args = ap.parse_args()
    if args.worker:
        args.params = args.params[0]
        args.outputs = args.outputs[0]
        worker(args)
    else:
        orchestrate(args)
