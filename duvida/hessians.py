"""JAX transforms for Hessians diagonals and their approximations."""
import warnings
from typing import Any
from collections.abc import Callable

from functools import partial

from .config import config
from .hvp import hvp
from .types import Approximator, Array
from .utils import (
    _check_chunk_size,
    chunked_vmap,
    default_chunk_size,
    output_size,
    jacrev,
    jvp,
    grad,
    random_normal,
    ravel_pytree,
    ravel_pytree_like,
    ravel_arg,
    get_eps,
    vmap,
)
from .numpy import numpy as dnp

_EPS = get_eps() ** .5
_DEFAULT_APPROXIMATOR = 'exact_diagonal'
_DIAGONAL_METHODS = ('auto', 'forward', 'hvp')


def get_approximators(
    key: str | Callable | None = None,
    *args, **kwargs
) -> tuple[str] | dict[str, Approximator]:

    """Return a list of available Hessian diagonal approximators, or one of the
    named approximator functions.

    Parameters
    ----------
    key : str, optional
        If provided, return the function with the name `key`.

    Returns
    -------
    Union[Tuple[str], Dict[str, Approximator]]
        If key is provided, returns the named approximator function. Otherwise
        returns a tuple of named approximators.

    Raises
    ------
    NotImplementedError
        When `key` is provided but is not in the list of approximators.

    """

    APPROXIMATORS = {
        'squared_jacobian': squared_jacobian,
        'exact_diagonal': exact_diagonal,
        'bekas': bekas,
    }
    # Resolvable by name for backward compatibility, but not advertised: these 
    # estimate H @ 1, which is not the Hessian diagonal when parameters interact.
    NON_DIAGONAL = {
        'directional_hessian_finite_difference': directional_hessian_finite_difference,
        'rough_finite_difference': rough_finite_difference,
    }

    if key is None:
        return tuple(APPROXIMATORS)
    elif isinstance(key, str):
        if key in NON_DIAGONAL:
            warnings.warn(
                f"Approximator '{key}' estimates H @ 1 (Hessian row sums), not the "
                "Hessian diagonal, and is wrong when parameters interact. Use "
                "'exact_diagonal' or 'bekas' for the diagonal.",
                UserWarning,
                stacklevel=2,
            )
            return partial(NON_DIAGONAL[key], *args, **kwargs)
        try:
            return partial(APPROXIMATORS[key], *args, **kwargs)
        except KeyError:
            raise NotImplementedError(
                f"Approximator called '{key}' is not implemented. "
                f"Choose from {', '.join(sorted(APPROXIMATORS))}"
            )
    elif isinstance(key, Callable):
        return key
    else:
        raise ValueError(f"Key must be a str or Callable, but was actually '{type(key)}'.")


def squared_jacobian(
    f: Callable[[Any], Array],
    argnums: int = 0,
    *args, **kwargs
) -> Callable[[Any], Array]:

    """Functional transform to the square of the Jacobian of a scalar-output function.
    
    Additional aguments are passed to `jax.grad()`. Set `argnums` to
    differentiate `f` with respect to parameters other than the first.

    Parameters
    ----------
    f : Callable
        Function with scalar-output to transform.
    
    Returns
    -------
    Callable
        Function which takes `f`'s original arguments. Returns the 
        square of the gradient of `f` with respect to its first parameter.

    Examples
    --------
    >>> from duvida.utils import hessian
    >>> import duvida.numpy as dnp  
    >>> f = lambda x: dnp.sum(x ** 3. + x ** 2. + 4.)
    >>> a = dnp.array([1., 2.])
    >>> f(a)
    Array(22., dtype=float64)
    >>> dnp.diag(hessian(f)(a))
    Array([ 8., 14.], dtype=float64)
    >>> squared_jacobian(f)(a) == dnp.square(grad(f)(a))
    Array([ True,  True], dtype=bool)
    >>> squared_jacobian(f)(a)  # Not very accurate!
    Array([ 25., 256.], dtype=float64)
    
    """
    def _squared_jacobian(
        *f_args,
        **f_kwargs,
    ):
        flat_f, flat_arg, unravel = ravel_arg(
            f,
            f_args,
            argnums=argnums,
            kwargs=f_kwargs,
        )
        jacobian = jacrev(flat_f)(flat_arg)
        return unravel(dnp.square(jacobian))

    return _squared_jacobian


def exact_diagonal(
    f: Callable[[Any], Array], 
    argnums: int = 0,
    device: str = 'cpu',
    *args, 
    chunk_size: int | None = None,
    method: str = 'auto',
    **kwargs
) -> Callable[[Any], Array]:

    """Functional transform to the Hessian diagonal of a scalar-output function.
    
    Additional aguments are passed to `hvp()`. Set `argnums` to
    differentiate `f` with respect to parameters other than the first.

    Parameters
    ----------
    f : Callable
        Function with scalar-output to transform.
    argnums : int, optional
        Which argument number to take the second derivative for. Default: 0.
    chunk_size : int, optional
        Number of parameters (basis vectors) to process at once; must be 
        positive. Smaller values use less memory but run more sequentially. 
        Default: as many as fit in a budget of `DUVIDA_MAX_BATCH_ELEMENTS` 
        (default 2**22) basis-vector elements. Note this bounds the *basis-vector
        batch*, not total memory: activations and autodiff state also scale with 
        the chunk size, and the returned diagonal (one value per output element 
        per parameter) is allocated in full regardless.
    method : str, optional
        "forward" for nested forward-mode second directional derivatives, 
        "hvp" for Hessian-vector products, or "auto". Default: "auto", which 
        uses "forward" for functions with more than one output element (3-24x 
        faster in benchmarks on both backends) and for scalar outputs on JAX 
        (as fast or faster), but "hvp" for scalar outputs on PyTorch, where it 
        is 1.4-3x faster for losses over tens of data points or more (the typical 
        case for Fisher information; "forward" is faster for a very cheap scalar 
        function). See `benchmarks/` to measure your own workload.
    
    Returns
    -------
    Callable
        Function which takes `f`'s original arguments. Returns the 
        diagonal of the Hessian of `f`. 

    Notes
    -----
    With `method="forward"`, each diagonal element is the second directional
    derivative of `f` along a basis vector, calculated by nesting two 
    forward-mode passes. Unlike a Hessian-vector product this needs no 
    reverse-mode pass per output element, which is far faster for 
    vector-output `f` such as a model evaluated on many data points.

    Examples
    --------
    >>> from duvida.utils import hessian
    >>> import duvida.numpy as dnp  
    >>> f = lambda x: dnp.sum(x ** 3. + x ** 2. + 4.)
    >>> a = dnp.array([1., 2.])
    >>> exact_diagonal(f)(a) == dnp.diag(hessian(f)(a))
    Array([ True,  True], dtype=bool)
    >>> g = lambda x: dnp.sum(dnp.sum(x) ** 3. + x ** 2. + 4.)
    >>> exact_diagonal(g)(a) == dnp.diag(hessian(g)(a))
    Array([ True,  True], dtype=bool)
    
    """
    if method not in _DIAGONAL_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {', '.join(_DIAGONAL_METHODS)}."
        )
    _check_chunk_size(chunk_size)
    hvp_f = hvp(f, argnums=argnums, *args, **kwargs)

    def _diagonal_forward(flat_f, flat_arg, unravel) -> Array:
        size = dnp.get_array_size(flat_arg)
        chunk = default_chunk_size(size) if chunk_size is None else chunk_size
        chunk = None if chunk >= size else chunk  # single batch: skip chunking machinery

        def _second_derivative(i):
            v = dnp.dtype_like(
                dnp.one_hot(i, size, device=device), 
                flat_arg,
            )
            def _first_derivative(a):
                return jvp(flat_f, (a,), (v,))[1]
            return jvp(_first_derivative, (flat_arg,), (v,))[1]

        v_second_derivative = chunked_vmap(
            _second_derivative, 
            chunk_size=chunk,
            in_axes=0,
            out_axes=-1,
        )
        idx = dnp.arange(size, device=device)
        return unravel(v_second_derivative(idx))

    def _diagonal_hvp(f_args, f_kwargs, flat_arg, n_outputs) -> Array:
        params = f_args[argnums]
        _, unravel = ravel_pytree(params)
        size = dnp.get_array_size(flat_arg)
        # Each basis vector here holds a (n_outputs, p) block of Hessian-vector 
        # products, so the budget is divided by the number of outputs too.
        chunk = (
            default_chunk_size(size * n_outputs) 
            if chunk_size is None else chunk_size
        )
        chunk = None if chunk >= size else chunk  # single batch: skip chunking machinery

        def get_hessian_element(i: int) -> float:
            unit_vec = unravel(dnp.dtype_like(
                dnp.one_hot(i, size, device=device), 
                flat_arg,
            ))
            r = hvp_f(unit_vec, *f_args, **f_kwargs)
            flat_r = ravel_pytree_like(r, params)
            return dnp.take(flat_r, i, axis=-1)

        v_hvp_f = chunked_vmap(
            get_hessian_element, 
            chunk_size=chunk,
            in_axes=0,
            out_axes=-1,
        )
        idx = dnp.arange(size, device=device)
        return unravel(v_hvp_f(idx))

    def _hessian_diagonal(*f_args, **f_kwargs) -> Array:
        flat_f, flat_arg, unravel = ravel_arg(
            f,
            f_args,
            argnums=argnums,
            kwargs=f_kwargs,
        )
        n_outputs = (
            1 if method == 'forward' 
            else output_size(flat_f, flat_arg)  # shape-only (JAX) or one cheap evaluation
        )
        use_forward = (
            method == 'forward' 
            or (
                method == 'auto' 
                and (n_outputs > 1 or config.backend == 'jax')
            )
        )
        if use_forward:
            return _diagonal_forward(flat_f, flat_arg, unravel)
        return _diagonal_hvp(f_args, f_kwargs, flat_arg, n_outputs)

    return _hessian_diagonal


def bekas(
    f: Callable[[Any], Array], 
    n: int = 1, 
    seed: int = 0,
    argnums: int = 0,
    device: str = 'cpu',
    *args, **kwargs
) -> Callable[[Any], Array]:

    """Functional transform to the Bekas estimator of the Hessian diagonal of a scalar-output function.
    
    Additional aguments are passed to `hvp()`. Set `argnums` to
    differentiate `f` with respect to parameters other than the first.

    Bekas estimator is described at https://doi.org/10.1016/j.apnum.2007.01.003.

    Parameters
    ----------
    f : Callable
        Function with scalar-output to transform.
    n : int, optional
        Number of Gaussian random samples to take. Default: 1.
    seed : int, optional
        Controls the random behavior. Set a different seed to get a 
        different result. Default: 0.
    argnums : int, optional
        Which argument number to take the second derivative for. Default: 0.
    device : str, optional
        For PyTorch backend. Which device to use: "cpu" or "cuda". Default: "cpu".
    
    Returns
    -------
    Callable
        Function which takes `f`'s original arguments. Returns the 
        Bekas approximation of the diagonal of the Hessian of `f`. 

    Examples
    --------
    >>> from duvida.utils import hessian
    >>> import duvida.numpy as dnp  
    >>> f = lambda x: dnp.sum(x ** 3. + x ** 2. + 4.)
    >>> a = dnp.array([1., 2.])
    >>> f(a)
    Array(22., dtype=float64)
    >>> dnp.diag(hessian(f)(a))
    Array([ 8., 14.], dtype=float64)
    >>> bekas(f)(a)  # Apparently very accurate
    Array([ 8., 14.], dtype=float64)
    >>> dnp.allclose(bekas(f)(a), dnp.diag(hessian(f)(a)))
    Array(True, dtype=bool)
    >>> bekas(f, n=100_000, seed=1)(a)  # Increase samples for better accuracy
    Array([ 8., 14.], dtype=float64)
    >>> bekas(f, seed=1)(a) == bekas(f, seed=0)(a)  # Change the seed to alter the outcome
    Array([ True, False], dtype=bool)
    >>> g = lambda x: dnp.sum(dnp.sum(x) ** 3. + x ** 2. + 4.)
    >>> dnp.diag(hessian(g)(a))
    Array([38., 38.], dtype=float64)
    >>> bekas(g, n=1000, seed=0)(a)  # Less accurate when parameters interact
    Array([39.50905509, 39.81868951], dtype=float64)
    >>> bekas(g, n=1000, seed=1)(a)  # Change the seed to alter the outcome
    Array([37.25647008, 37.29415901], dtype=float64)
    
    """
    random_normal_fn = random_normal(seed, device=device)
    hvp_f = hvp(f, argnums=argnums, *args, **kwargs)

    def _approx_hessian_diagonal(*f_args, **f_kwargs) -> Array:
        params = f_args[argnums]
        flat_params, unravel = ravel_pytree(params)
        d_args_size = dnp.get_array_size(flat_params)

        def flat_hvp(flat_v, *f_args):
            v = unravel(flat_v)
            r = hvp_f(v, *f_args, **f_kwargs)
            return ravel_pytree_like(r, params)

        v_hvp_f = vmap(
            flat_hvp, 
            in_axes=(1, ) + (None, ) * len(f_args), 
            out_axes=-1,
        )
        v = random_normal_fn(shape=(d_args_size, n), dtype=flat_params.dtype)  # p, n  # TODO: Don't instantiate all at once - risk of memory blow-up
        samples = v * v_hvp_f(v, *f_args, **f_kwargs)   # p, n
        return unravel(dnp.sum(samples, axis=-1) / dnp.sum(dnp.square(v), axis=-1))

    return _approx_hessian_diagonal


def directional_hessian_finite_difference(
    f: Callable[[Any], Array], 
    argnums: int = 0, 
    eps: float | None = None,
    *args, **kwargs
) -> Callable[[Any], Array]:
    
    """Functional transform to a finite-difference estimate of `H @ 1`, the row sums
    of the Hessian of a scalar-output function.

    This uses the usual finite-difference algorithm from `scipy.optimize.fprime()`
    to calculate the Jacobian of the autodiff gradients of `f`, but changes 
    all values at once, instead of one-by-one. This is the directional derivative
    of the gradient along the all-ones vector, and costs two gradient evaluations 
    regardless of the number of parameters.
    
    **This is not an estimate of the Hessian diagonal in general.** Perturbing 
    every parameter at once estimates the Hessian-vector product `H @ 1`, 
    i.e. the row sums of the Hessian, which equals the diagonal only when 
    parameters don't interact (for example `f(x) = (sum(x)) ** 2` has a diagonal
    of 2 but returns `2 * len(x)`). Use `exact_diagonal` or `bekas` when 
    parameters interact.
    
    Additional aguments are passed to `grad()`. Set `argnums` to
    differentiate `f` with respect to parameters other than the first.

    Parameters
    ----------
    f : Callable
        Function with scalar-output to transform.
    argnums : int, optional
        Which argument number to take the second derivative for. Default: 0.
    eps : float, optional
        The small number to add to prevent instability. Default: square root 
        of machine precision of the parameters' dtype.
    
    Returns
    -------
    Callable
        Function which takes `f`'s original arguments. Returns the 
        finite-difference estimate of `H @ 1` (equal to the diagonal of the 
        Hessian of `f` only when parameters do not interact). 

    Examples
    --------
    >>> from duvida.utils import hessian
    >>> import duvida.numpy as dnp 
    >>> f = lambda x: dnp.sum(x ** 3. + x ** 2. + 4.)
    >>> a = dnp.array([1., 2.])
    >>> f(a)
    Array(22., dtype=float64)
    >>> dnp.diag(hessian(f)(a))
    Array([ 8., 14.], dtype=float64)
    >>> directional_hessian_finite_difference(f)(a)  # Relatively accurate  # doctest: +NORMALIZE_WHITESPACE
    Array([ 8.00000006, 14.        ], dtype=float64)
    >>> directional_hessian_finite_difference(f, eps=.01)(a)
    Array([ 8.03, 14.03], dtype=float64)
    >>> g = lambda x: dnp.sum(dnp.sum(x) ** 3. + x ** 2. + 4.)
    >>> dnp.diag(hessian(g)(a))
    Array([38., 38.], dtype=float64)
    >>> dnp.allclose(rough_finite_difference(g)(a), dnp.array([74., 74.]))  # Less accurate when parameters interact  # doctest: +NORMALIZE_WHITESPACE
    Array(True, dtype=bool)
    
    """
    def _approx_hessian_diagonal(
        *f_args,
        **f_kwargs,
    ):

        flat_f, flat_arg, unravel = ravel_arg(
            f,
            f_args,
            argnums=argnums,
            kwargs=f_kwargs,
        )
        step = eps if eps is not None else get_eps(flat_arg.dtype) ** .5
        jacobian = jacrev(flat_f)
        observed = (jacobian(flat_arg + step) - jacobian(flat_arg)) / step
        return unravel(observed)

    return _approx_hessian_diagonal


def rough_finite_difference(*args, **kwargs) -> Callable[[Any], Array]:

    """Deprecated alias of `directional_hessian_finite_difference`.

    The old name suggested a rough estimate of the Hessian diagonal, but this
    estimates `H @ 1` (the Hessian row sums), which is the diagonal only when 
    parameters don't interact.

    """
    warnings.warn(
        "`rough_finite_difference` is deprecated and will be removed; it estimates "
        "H @ 1 (Hessian row sums), not the Hessian diagonal. Use "
        "`directional_hessian_finite_difference` if that is what you want, or "
        "`exact_diagonal` / `bekas` for the diagonal.",
        DeprecationWarning,
        stacklevel=2,
    )
    return directional_hessian_finite_difference(*args, **kwargs)


if _DEFAULT_APPROXIMATOR not in get_approximators():  # Should never happen
    raise KeyError(f"Default approximator {_DEFAULT_APPROXIMATOR} not in approximator dictionary!")  
