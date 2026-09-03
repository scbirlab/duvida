"""JAX transforms for Hessians diagonals and their approximations."""
from typing import Any
from collections.abc import Callable

from functools import partial

from .hvp import hvp
from .types import Approximator, Array
from .utils import (
    jacrev,
    jvp,
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
        'rough_finite_difference': rough_finite_difference,
    }

    if key is None:
        return tuple(APPROXIMATORS)
    elif isinstance(key, str):
        try:
            return partial(APPROXIMATORS[key], *args, **kwargs)
        except KeyError:
            NotImplementedError(
                f"Approximator called '{key}' is not implemented. Choose from {', '.join(sorted(APPROXIMATORS))}"
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
    *args, **kwargs
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
    
    Returns
    -------
    Callable
        Function which takes `f`'s original arguments. Returns the 
        diagonal of the Hessian of `f`. 

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
    hvp_f = hvp(f, argnums=argnums, *args, **kwargs)

    def get_hessian_element(
        i: int, 
        size: int, 
        *f_args, **f_kwargs
    ) -> float:
        params = f_args[argnums]
        _, unravel = ravel_pytree(params)
        unit_vec = unravel(dnp.one_hot(i, size, device=device))
        r = hvp_f(unit_vec, *f_args, **f_kwargs)
        flat_r = ravel_pytree_like(r, params)
        return dnp.take(flat_r, i, axis=-1)

    def _hessian_diagonal(*f_args, **f_kwargs) -> Array:
        params = f_args[argnums]
        flat_params, unravel = ravel_pytree(params)

        d_args_size = dnp.get_array_size(flat_params)
        v_hvp_f = vmap(
            get_hessian_element, 
            in_axes=(0, None) + (None, ) * len(f_args),
            out_axes=-1,
        )
        idx = dnp.arange(d_args_size, device=device)
        hessian_diag = v_hvp_f(idx, d_args_size, *f_args, **f_kwargs)
        return unravel(hessian_diag)

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

        v_hvp_f = vmap(
            hvp_f, 
            in_axes=(1, ) + (None, ) * len(args), 
            out_axes=-1,
        )
        v = random_normal_fn(shape=(d_args_size, n))  # p, n  # TODO: Don't instantiate all at once - risk of memory blow-up
        samples = v * v_hvp_f(v, *f_args, **f_kwargs)   # p, n
        return ravel_pytree_like(dnp.sum(samples, axis=-1) / dnp.sum(dnp.square(v), axis=-1), params)

    return _approx_hessian_diagonal


def rough_finite_difference(
    f: Callable[[Any], Array], 
    argnums: int = 0, 
    eps: float = _EPS,
    *args, **kwargs
) -> Callable[[Any], Array]:
    
    """Functional transform to a rough Hessian diagonal of a scalar-output function.

    This uses the usual finite-difference algorithm from `scipy.optimize.fprime()`
    to calculate the Jacobian of the autodiff gradients of `f`, but changes 
    all values at once, instead of one-by-one. 
    
    This will work best when parameters don't have strong interactions with
    each other becuase it's essentially a column sum of the Hessian matrix.
    
    Additional aguments are passed to `grad()`. Set `argnums` to
    differentiate `f` with respect to parameters other than the first.

    Parameters
    ----------
    f : Callable
        Function with scalar-output to transform.
    argnums : int, optional
        Which argument number to take the second derivative for. Default: 0.
    eps : float, optional
        The small number to add to prevent instability.
    
    Returns
    -------
    Callable
        Function which takes `f`'s original arguments. Returns the 
        rough finite-difference approximation of the diagonal of the 
        Hessian of `f`. 

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
    >>> rough_finite_difference(f)(a)  # Relatively accurate
    Array([ 8.0010358, 14.0010358], dtype=float64)
    >>> rough_finite_difference(f, eps=.01)(a)
    Array([ 8.03, 14.03], dtype=float64)
    >>> g = lambda x: dnp.sum(dnp.sum(x) ** 3. + x ** 2. + 4.)
    >>> dnp.diag(hessian(g)(a))
    Array([38., 38.], dtype=float64)
    >>> rough_finite_difference(g)(a)  # Less accurate when parameters interact
    Array([74.00828641, 74.00828641], dtype=float64)
    
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
        jacobian = jacrev(flat_f)
        observed = (jacobian(flat_arg + eps) - jacobian(flat_arg)) / eps
        return unravel(observed)

    return _approx_hessian_diagonal


if _DEFAULT_APPROXIMATOR not in get_approximators():  # Should never happen
    raise KeyError(f"Default approximator {_DEFAULT_APPROXIMATOR} not in approximator dictionary!")  
