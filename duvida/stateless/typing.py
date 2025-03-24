"""Generic types for JAX and PyTorch"""

from typing import  Any, Callable, Iterable, Union

from .config import config

if config.backend == 'jax':
    from jax import Array
    from jax.typing import ArrayLike
else:
    if config.precision == 'double':
        from torch import DoubleTensor as Array
    elif config.precision == 'float':
        from torch import FloatTensor as Array
    elif config.precision == 'half':
        from torch import HalfTensor as Array
    else:
        from torch import FloatTensor as Array
    ArrayLike = Union[Array, float]

Approximator = Callable[[Callable], Callable]
LossFunction = Callable[[ArrayLike, ArrayLike], float]
StatelessModel = Callable[[Iterable[ArrayLike], ArrayLike], Array]