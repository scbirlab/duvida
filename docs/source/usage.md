# Usage

## Training a model in Python

`duvida` provides an easy way to construct and train models using the `AutoClass` helper.

```python
from duvida.autoclass import AutoClass

modelbox = AutoClass(
    "fingerprint",
    n_units=16,
    n_hidden=2,
    ensemble_size=10,
)

modelbox.load_training_data(
    filename="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train",
    inputs="smiles",
    labels="clogp",
)

modelbox.train(epochs=10, batch_size=128)
```

Predictions and metrics can then be obtained with the evaluation helpers:

```python
from duvida.evaluation import rmse

preds, metrics = modelbox.evaluate(
    filename="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test",
    metrics={"RMSE": rmse},
)
```

Exact or approximate Hessian diagonals are also available:

```python
from duvida.stateless.utils import hessian
import duvida.stateless.numpy as dnp

f = lambda x: dnp.sum(x ** 2)
a = dnp.array([1., 2.])
print(dnp.diag(hessian(f)(a)))
```

## Command-line interface

A command-line application is provided in `duvida.cli`. It is available via the `duvida` console script or by running `python -m duvida.cli`.

Show the available commands:

```bash
$ duvida --help
```

Train a model from the command line:

```bash
$ duvida train -1 train.csv -2 val.csv --output model.dv
```

Make predictions with uncertainty metrics:

```bash
$ duvida predict --test test.csv --checkpoint model.dv \
    --doubtscore --output predictions.parquet
```

You can also invoke the CLI programmatically:

```python
from duvida.cli import main
main(["train", "-1", "train.csv", "-2", "val.csv", "--output", "model.dv"])
```
