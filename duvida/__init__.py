from importlib.metadata import version

appname = "duvida"
__version__ = version(appname)
__author__ = "Eachan Johnson"

from .hessians import (
    bekas,
    exact_diagonal,
    get_approximators, 
    rough_finite_difference,
    squared_jacobian
)
from .hvp import hvp
from .information import (
    doubtscore,
    fisher_score,
    fisher_information_diagonal,
    parameter_gradient,
    parameter_hessian_diagonal,
    information_sensitivity
)
from .utils import *
