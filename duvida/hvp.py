"""Functional transform for Hessian vector product."""

from collections.abc import Callable
from functools import partial

from .numpy import numpy as dnp
from .types import Array, ArrayLike
from .utils import (
    jacrev,
    jvp,
    ravel_arg,
    ravel_pytree,
)


def hvp(
    f: Callable, 
    argnums: int = 0, 
    *args, **kwargs
) -> Callable:

    """Forward-over-reverse Hessian vector product transform for scalar-output functions.

    This does not behave exactly like other functional transforms. The 
    resulting function takes as its first argument a vector, and the 
    following argments are the original function's positional and 
    keyword arguments.

    Additional aguments are passed to `jax.grad()`.

    Parameters
    ----------
    f : Callable
        Function with scalar-output to transform.
    argnums : int, optional
        Which argument number to take the second derivative for. Default: 0.
    
    Returns
    -------
    Callable
        Function which takes a vector `v` as its first argument, and then `f`'s 
        original arguments. Returns the vector product between the Hessian
        of `f` and `v`.

    Examples
    --------
    >>> from duvida.config import config
    >>> config.set_backend("jax", precision="double")
    >>> from duvida.utils import grad, hessian
    >>> import duvida.numpy as dnp 
    >>> f = lambda x: dnp.sum(x ** 3. + x ** 2. + 4.)
    >>> a = dnp.array([1., 2.])
    >>> f(a)
    Array(22., dtype=float64)
    >>> hvp(f)(dnp.ones_like(a), a) == hessian(f)(a) @ dnp.ones_like(a)
    Array([ True,  True], dtype=bool)
    >>> g = lambda x, y: dnp.sum(x ** 2. + x ** 2. + 4. + y ** 3.)
    >>> b = dnp.array([3., 4.])
    >>> hvp(g)(dnp.ones_like(a), a, b) == hessian(g)(a, b) @ dnp.ones_like(a)
    Array([ True,  True], dtype=bool)
    >>> hvp(g, argnums=1)(dnp.ones_like(a), a, b) == hessian(g, argnums=1)(a, b) @ dnp.ones_like(a)
    Array([ True,  True], dtype=bool)

    """

    def _hvp(
        v: ArrayLike,
        *f_args,
        **f_kwargs,
    ) -> Array:
        flat_f, flat_arg, unravel = ravel_arg(
            f,
            f_args,
            argnums=argnums,
            kwargs=f_kwargs,
        )

        flat_v, _ = ravel_pytree(v)
        if dnp.get_array_shape(flat_v) != get_array_shape(flat_arg).shape:
            raise ValueError(
                "HVP vector must have the same flattened "
                "shape as the differentiated argument."
            )

        jacobian = jacrev(flat_f)

        flat_hvp = jvp(
            jacobian,
            (flat_arg,),
            (flat_v,),
        )[1]

        return unravel(flat_hvp)

    return _hvp
