# 🧐 duvida

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/scbirlab/duvida/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/duvida)
![PyPI](https://img.shields.io/pypi/v/duvida)

**duvida** (Portuguese for _doubt_) is a suite of python tools for calculating confidence and information metrics 
for deep learning. It provides lower-level function transforms for exact and approximate Hessian diagonals 
in JAX and pytorch. 

- [Installation](#installation)
- [Python API](#python-api)
- [Issues, problems, suggestions](#issues-problems-suggestions)
- [Documentation](#documentation)

## Installation

### The easy way

You can install the precompiled version directly using `pip`. You need to specify the machine learning framework
that you want to use:

```bash
$ pip install duvida[jax]
# or
$ pip install duvida[jax_cuda12]  # for JAX installing CUDA 12 for GPU support
# or
$ pip install duvida[jax_cuda12_local]  # for JAX using a locally-installed CUDA 12
# or
$ pip install duvida[torch]
```

We have implemented JAX and pytorch functional transformations for approximate and exact Hessian diagonals,
and doubtscore and information sensitivity. These can be used with JAX- and pytorch-based frameworks.

### From source

Clone the repository, then `cd` into it. Then run:

```bash
$ pip install -e .[torch]
```

## Python API

**duvida** provides functional transforms for JAX and pytorch that calculate 
either exact or approximate Hessian diagonals.

You can check which backend you're using:

```python
>>> from duvida.stateless.config import config
>>> config
Config(backend='jax', precision='double', fallback=True)
```

It can be changed:

```python
>>> config.set_backend("torch")
'torch'
>>> config
Config(backend='torch', precision='double', fallback=True)
```

Now you can calculate exact Hessian diagonals without calculating the 
full matrix:

```python
>>> from duvida.stateless.utils import hessian
>>> import duvida.stateless.numpy as dnp 
>>> f = lambda x: dnp.sum(x ** 3. + x ** 2. + 4.)
>>> a = dnp.array([1., 2.])
>>> exact_diagonal(f)(a) == dnp.diag(hessian(f)(a))
Array([ True,  True], dtype=bool)
```

Various approximations are also allowed.

```python
>>> from duvida.stateless.hessians import get_approximators
>>> get_approximators()  # Use no arguments to show what's available
('squared_jacobian', 'exact_diagonal', 'bekas', 'rough_finite_difference')
```

Now apply:

```python
>>> approx_hessian_diag = get_approximators("bekas")
>>> g = lambda x: dnp.sum(dnp.sum(x) ** 3. + x ** 2. + 4.)
>>> a = dnp.array([1., 2.])
>>> dnp.diag(hessian(g)(a))  # Exact
Array([38., 38.], dtype=float64)
>>> approx_hessian_diag(g, n=1000)(a)  # Less accurate when parameters interact
Array([38.52438307, 38.49679655], dtype=float64)
>>> approx_hessian_diag(g, n=1000, seed=1)(a)  # Change the seed to alter the outcome
Array([39.07878869, 38.97796601], dtype=float64)
```

## Issues, problems, suggestions

Add to the [issue tracker](https://www.github.com/scbirlab/duvida/issues).

## Documentation

(To come at [ReadTheDocs](https://duvida.readthedocs.org).)