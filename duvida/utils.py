"""Generic utilities for JAX and PyTorch."""

from collections.abc import Callable, Iterable
import os
from functools import partial
from itertools import accumulate

from carabiner import print_err

from .config import config
from .types import Array, ArrayLike

__backend__ = config.backend


def _check_chunk_size(chunk_size: int | None) -> None:
    """Raise `ValueError` unless `chunk_size` is `None` or a positive integer."""
    if chunk_size is not None and (
        isinstance(chunk_size, bool) 
        or not isinstance(chunk_size, int) 
        or chunk_size < 1
    ):
        raise ValueError(
            f"`chunk_size` must be None or a positive integer, got {chunk_size!r}."
        )


def _env_positive_int(name: str, default: int) -> int:
    """Read a positive integer from the environment, raising a clear error otherwise."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"Environment variable {name} must be a positive integer, got {raw!r}.")
    return value


# Budget, in array elements, for the batch of basis vectors that exact Hessian
# diagonals process at once (2**22 elements = 32 MB in float64). This bounds
# the basis-vector batch only, not total memory: activations and autodiff state
# scale with the chunk size too, and the output itself is never chunked.
MAX_BATCH_ELEMENTS = _env_positive_int("DUVIDA_MAX_BATCH_ELEMENTS", 2 ** 22)


def default_chunk_size(size: int, cap: int | None = None) -> int:
    """Largest number of length-`size` vectors that fit in `MAX_BATCH_ELEMENTS`
    (at least 1, at most `cap` if given)."""
    chunk = max(1, MAX_BATCH_ELEMENTS // max(int(size), 1))
    return min(chunk, cap) if cap is not None else chunk


if __backend__ == 'jax':
    from jax import (
        grad,
        hessian,
        jacrev as jax_jacrev,
        jit,
        jvp,
        random,
        vmap
    )
    from jax import lax, vmap as jax_vmap
    from jax.numpy import (
        concatenate,
        moveaxis,
        split
    )
    from jax.tree_util import (
        tree_flatten,
        tree_map,
        tree_unflatten as jax_tree_unflatten,
    )

    def tree_unflatten(spec, leaves):
        return jax_tree_unflatten(spec, leaves)

    def output_size(f: Callable, *args) -> int:
        """Number of elements in the output of `f(*args)`, found by tracing 
        shapes only (no computation)."""
        from jax import eval_shape
        from math import prod
        return sum(
            prod(leaf.shape) for leaf in tree_flatten(eval_shape(f, *args))[0]
        )

    def jacrev(f: Callable, *args, chunk_size: int | None = None, **kwargs) -> Callable:
        """Reverse-mode Jacobian. `chunk_size` is accepted for API parity with the
        PyTorch backend and ignored."""
        _check_chunk_size(chunk_size)
        return jax_jacrev(f, *args, **kwargs)

    def chunked_vmap(
        f: Callable,
        chunk_size: int | None = None,
        in_axes: int | Iterable[int | None] = 0,
        out_axes: int = 0,
    ) -> Callable:
        """Vectorize `f` over its mapped arguments, `chunk_size` elements at a time.

        This is deliberately *not* a drop-in for `jax.vmap`. It is a bounded-memory
        map for the simple case where each entry of `in_axes` is an `int` or `None`
        (an `int` applies to every leaf of that argument) and `out_axes` is an `int`.
        Anything else (nested `in_axes` pytrees, `axis_name`, `spmd_axis_name`, ...)
        is rejected with a `ValueError` rather than silently handled differently.
        With `chunk_size=None` it is exactly `jax.vmap(f, in_axes, out_axes)`.

        Chunks are processed sequentially with `jax.lax.map(..., batch_size=...)`,
        so peak memory scales with `chunk_size` instead of the full batch.

        Examples
        ========
        >>> import duvida.numpy as dnp
        >>> double = lambda z: z * 2
        >>> chunked_vmap(double, chunk_size=2)(dnp.arange(5)).tolist()
        [0, 2, 4, 6, 8]

        """
        _check_chunk_size(chunk_size)
        if not isinstance(out_axes, int):
            raise ValueError(f"`out_axes` must be an int, got {out_axes!r}.")
        if isinstance(in_axes, (tuple, list)):
            if not all(a is None or isinstance(a, int) for a in in_axes):
                raise ValueError(
                    "Each `in_axes` entry must be an int or None, got "
                    f"{in_axes!r}."
                )
        elif not isinstance(in_axes, int):
            raise ValueError(f"`in_axes` must be an int or a sequence of int/None, got {in_axes!r}.")
        if chunk_size is None:
            return jax_vmap(f, in_axes, out_axes)

        def _chunked(*f_args):
            axes = (
                tuple(in_axes) 
                if isinstance(in_axes, (tuple, list)) 
                else (in_axes,) * len(f_args)
            )
            if len(axes) != len(f_args):
                raise ValueError(
                    f"`in_axes` has {len(axes)} entries for {len(f_args)} arguments."
                )
            mapped = [i for i, a in enumerate(axes) if a is not None]
            xs = tuple(
                tree_map(lambda leaf: moveaxis(leaf, axes[i], 0), f_args[i])
                for i in mapped
            )

            def _per_element(element):
                call_args = list(f_args)
                for i, e in zip(mapped, element):
                    call_args[i] = e
                return f(*call_args)

            out = lax.map(_per_element, xs, batch_size=chunk_size)
            return tree_map(lambda leaf: moveaxis(leaf, 0, out_axes), out)

        return _chunked

    def random_normal(
        seed: int, 
        device=None
    ) -> Callable:
        """Generate a sample from the Normal distribution.

        Examples
        ========
        >>> gen = random_normal(seed=0)
        >>> a, b = gen((3,)), gen((3,))
        >>> (a == b).all()
        Array(True, dtype=bool)

        """
        key = random.key(seed)
        def _normal(shape, *args, **kwargs):
            return random.normal(
                key, 
                shape=shape, 
                *args, **kwargs
            )
        return _normal
        

elif __backend__ == 'torch':

    from functools import wraps
    from torch import concat as concatenate, compile, randn, Generator, split
    from torch.random import manual_seed
    from torch.func import (
        jacrev as jacrev_torch, 
        jvp, 
        grad, 
        hessian, 
        vmap as vmap_torch
    )
    from torch.utils._pytree import (
        tree_flatten,
        tree_map,
        tree_unflatten as torch_tree_unflatten,
    )
    from torch._dynamo import config as dynamo_config
    dynamo_config.suppress_errors = True
    dynamo_config.capture_scalar_outputs = True

    _COMPILE_WARNINGS = set()


    def tree_unflatten(spec, leaves):
        return torch_tree_unflatten(leaves, spec)

    def output_size(f: Callable, *args) -> int:
        """Number of elements in the output of `f(*args)` (one evaluation 
        without gradient tracking)."""
        from torch import no_grad
        with no_grad():
            return sum(leaf.numel() for leaf in tree_flatten(f(*args))[0])

    _JACREV_MAX_CHUNK_SIZE = _env_positive_int("DUVIDA_JACREV_CHUNK_SIZE", 64)

    def jacrev(
        f: Callable, 
        *args, 
        chunk_size: int | None = None,
        **kwargs
    ) -> Callable:
        """Reverse-mode Jacobian with chunked backward passes.

        The backward pass for each output element is batched in chunks of 
        `chunk_size`, which is much faster than one at a time but uses up to
        `chunk_size` times more memory. An explicit `chunk_size` is used as given
        and is *not* limited by the memory budget (`DUVIDA_MAX_BATCH_ELEMENTS`); only 
        the automatic choice is. If `chunk_size` is `None` it is chosen 
        per call as `min(64, MAX_BATCH_ELEMENTS // p)` where `p` is the size of 
        the differentiated argument (64 can be changed with the 
        `DUVIDA_JACREV_CHUNK_SIZE` environment variable), so large parameter 
        sets automatically get smaller chunks.
        """
        _check_chunk_size(chunk_size)
        argnums = kwargs.get("argnums", args[0] if args else 0)

        def _jacobian(*f_args, **f_kwargs):
            chunk = chunk_size
            if chunk is None:
                selected = (
                    [f_args[i] for i in argnums] 
                    if isinstance(argnums, (tuple, list)) 
                    else [f_args[argnums]]
                )
                size = sum(
                    leaf.numel() 
                    for arg in selected 
                    for leaf in tree_flatten(arg)[0]
                )
                chunk = default_chunk_size(size, cap=_JACREV_MAX_CHUNK_SIZE)
            return jacrev_torch(
                f, *args, chunk_size=chunk, **kwargs,
            )(*f_args, **f_kwargs)

        return _jacobian

    def vmap(
        f: Callable, 
        in_axes: int | Iterable[int] = 0, 
        out_axes: int | Iterable[int] = 0, 
        *args, **kwargs
    ) -> Callable:
        """Vectorizes function over axis of its arguments.

        Examples
        ========
        >>> import duvida.numpy as dnp
        >>> double = lambda z: z * 2
        >>> vmap(double)(dnp.array([1., 2., 3.])).tolist()
        [2.0, 4.0, 6.0]

        """
        return vmap_torch(
            f, 
            in_dims=in_axes, 
            out_dims=out_axes, 
            randomness='different',
            *args, **kwargs
        )


    def chunked_vmap(
        f: Callable, 
        chunk_size: int | None = None,
        in_axes: int | Iterable[int | None] = 0, 
        out_axes: int = 0, 
    ) -> Callable:
        """Vectorizes function over axis of its arguments, `chunk_size` elements 
        at a time, so peak memory scales with `chunk_size` instead of the full batch.
        With `chunk_size=None` the whole batch is processed at once.

        Examples
        ========
        >>> import duvida.numpy as dnp
        >>> double = lambda z: z * 2
        >>> chunked_vmap(double, chunk_size=2)(dnp.arange(5)).tolist()
        [0, 2, 4, 6, 8]

        """
        _check_chunk_size(chunk_size)
        return vmap_torch(
            f, 
            in_dims=in_axes, 
            out_dims=out_axes, 
            randomness='different',
            chunk_size=chunk_size,
        )
        

    def jit(fn):
        try:
            compiled = compile(
                fn,
                fullgraph=True,
                mode="max-autotune",
            )
        except RuntimeError as e:
            warning = f"[torch.compile] Compiling `{fn}` failed; running eagerly"
            if warning not in _COMPILE_WARNINGS:
                _COMPILE_WARNINGS.add(warning)
                print_err(warning + "\n" + str(e))
            return fn

        @wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return compiled(*args, **kwargs)
            except Exception as e:
                warning = f"[torch.compile] Running compiled `{fn}` failed; falling back"
                if warning not in _COMPILE_WARNINGS:
                    _COMPILE_WARNINGS.add(warning)
                    print_err(warning + "\n" + str(e))
                return fn(*args, **kwargs)
        return wrapped


    def random_normal(seed: int, device='cpu') -> Callable:
        """Generate a sample from the Normal distribution.

        Examples
        ========
        >>> gen = random_normal(seed=0)
        >>> a, b = gen((3,)), gen((3,))
        >>> a, b
        [1], [2]
        >>> (a == b).all()
        tensor(True)

        """
        generator = Generator(device=device)
        generator.manual_seed(seed)

        def _normal(shape, dtype=None, *args, **kwargs):
            return randn(
                *shape,
                generator=generator,
                device=device,
                dtype=dtype,
                *args, **kwargs
            )

        return _normal


    def ravel_pytree(params):
        """Torch pytree flattener.

        Returns
        -------
        flat : 1-D ndarray / Tensor
        unflatten : Callable[[1-D ndarray], original-shaped pytree]

        Examples
        --------
        >>> import duvida.numpy as dnp
        >>> flat, unravel = ravel_pytree([dnp.ones((2,)), dnp.zeros((3,))])
        >>> flat.shape
        (5,)
        >>> (unravel(flat)[0] == 1.).all()
        True
        """
        leaves, spec = tree_flatten(params)  # 
        sizes = [p.numel() for p in leaves]
        flat = concatenate([p.flatten() for p in leaves])

        def unravel(vec):
            chunks = split(vec, sizes)
            rebuilt = [
                chunk.reshape_as(p) 
                for chunk, p in zip(chunks, leaves)
            ]
            return tree_unflatten(spec, rebuilt)

        return flat, unravel
else:
    raise ValueError(f"Invalid backed: {config.backend}")
    

def reciprocal(f: Callable[[ArrayLike], Array]) -> Callable[[ArrayLike], Array]:

    """Convert a function returning a numeric value to return its reciprocal instead.
    
    Parameters
    ----------
    f : Callable
        Function to convert.
    
    Returns
    -------
    Callable
        Function returning the reciprocal of `f`.

    Examples
    --------
    >>> f = lambda x: x + 1.
    >>> f(3.)
    4.0
    >>> reciprocal(f)(1.)
    0.5

    """

    def _inverter(x):
        return 1. / x

    def _inverted(*args, **kwargs):
        return tree_map(_inverter, f(*args, **kwargs))

    return _inverted


def get_eps(dtype=None):

    """Get machine precision, for `dtype` if given or else the backend's default float.
    
    """
    
    if config.backend == 'jax':
        from jax.numpy import asarray, finfo

        def converter(x): 
            return asarray(x)

    else:
        from torch import as_tensor, finfo

        def converter(x): 
            return as_tensor(x)
            
    if dtype is None:
        dtype = converter(1.).dtype
    return converter(finfo(dtype).eps)



def ravel_pytree(params):
    """Flatten a pytree while preserving leading dimensions on unravel."""

    from .numpy import numpy as dnp

    leaves, spec = tree_flatten(params)

    sizes = [dnp.get_array_size(leaf) for leaf in leaves]

    flat = dnp.concatenate([
        leaf.reshape(-1)
        for leaf in leaves
    ])

    def unravel(vec):
        chunks = dnp.split(
            vec,
            sizes,
            axis=-1,
        )
        leading_shape = dnp.get_array_shape(vec)[:-1]
        rebuilt = [
            chunk.reshape(
                *leading_shape,
                *dnp.get_array_shape(leaf),
            )
            for chunk, leaf
            in zip(chunks, leaves)
        ]

        return tree_unflatten(
            spec,
            rebuilt,
        )

    return flat, unravel


def ravel_arg(
    f: Callable,
    args,
    argnums: int = 0,
    kwargs=None
):
    """Flatten one pytree argument of a function."""
    if kwargs is None:
        kwargs = {}
    args = tuple(args)

    flat_arg, unravel = ravel_pytree(args[argnums])

    def flat_f(flat_arg):
        f_args = list(args)
        f_args[argnums] = unravel(flat_arg)
        return f(*f_args, **kwargs)

    return flat_f, flat_arg, unravel


def ravel_pytree_like(
    pytree,
    reference
):
    """Ravel pytree parameter dimensions while preserving leading dimensions."""

    from .numpy import numpy as dnp

    leaves, spec = tree_flatten(pytree)
    reference_leaves, reference_spec = tree_flatten(reference)

    if spec != reference_spec:
        raise ValueError(
            "Pytree and reference must have the same structure."
        )

    leading_shape = None
    flat_leaves = []

    for leaf, reference_leaf in zip(
        leaves,
        reference_leaves,
    ):
        leaf_shape = dnp.get_array_shape(leaf)
        reference_shape = dnp.get_array_shape(reference_leaf)
        reference_size = dnp.get_array_size(reference_leaf)

        if reference_shape:
            if leaf_shape[-len(reference_shape):] != reference_shape:
                raise ValueError(
                    "Pytree leaf trailing dimensions must match "
                    "the corresponding reference leaf."
                )

            leaf_leading_shape = leaf_shape[:-len(reference_shape)]
        else:
            leaf_leading_shape = leaf_shape

        if leading_shape is None:
            leading_shape = leaf_leading_shape
        elif leaf_leading_shape != leading_shape:
            raise ValueError(
                "Pytree leaves must have the same leading dimensions."
            )

        flat_leaves.append(
            leaf.reshape(
                *leaf_leading_shape,
                reference_size,
            )
        )

    return dnp.concatenate(flat_leaves, axis=-1)
