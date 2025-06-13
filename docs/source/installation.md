# Installation

## The easy way

Install the package from PyPI. Choose the backend that suits your workflow:

```bash
$ pip install duvida[jax]
# or
$ pip install duvida[jax_cuda12]     # for CUDA 12 GPU support
# or
$ pip install duvida[jax_cuda12_local]
# or
$ pip install duvida[torch]
```

For chemistry specific tools, install with the `chem` extras:

```bash
$ pip install duvida[chem]
```

## From source

Clone the repository and install in editable mode:

```bash
$ git clone https://github.com/scbirlab/duvida.git
$ cd duvida
$ pip install -e .[torch]
```
