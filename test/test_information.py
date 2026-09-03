def test_fisher_score_preserves_parameter_pytree():

    import duvida.numpy as dnp

    from duvida import fisher_score

    def model(x, params):
        return x[:, None] * params["weight"] + params["bias"]

    def loss(
        prediction,
        target,
    ):
        return dnp.sum(dnp.square(prediction - target))

    params = {
        "weight": dnp.array([2., 4.]),
        "bias": dnp.array([1., -1.]),
    }

    x = dnp.array([1., 2., 3.])

    target = dnp.zeros(3, 2)

    observed = fisher_score(model, loss)(
        (params,),
        x,
        target,
    )[0]

    assert set(observed) == {"weight", "bias"}

    assert observed["weight"].shape == params["weight"].shape
    assert observed["bias"].shape == params["bias"].shape

    assert dnp.all(dnp.isfinite(observed["weight"]))
    assert dnp.all(dnp.isfinite(observed["bias"]))


def test_fisher_information_preserves_parameter_pytree():

    import duvida.numpy as dnp

    from duvida import fisher_information_diagonal

    def model(x, params):
        return x[:, None] * params["weight"] + params["bias"]

    def loss(
        prediction,
        target,
    ):
        return dnp.sum(dnp.square(prediction - target))

    params = {
        "weight": dnp.array([2., 4.]),
        "bias": dnp.array([1., -1.]),
    }

    x = dnp.array([1., 2., 3.])

    target = dnp.zeros(3, 2)
    observed = fisher_information_diagonal(model, loss)(
        (params,),
        x,
        target,
    )[0]

    assert set(observed) == {"weight", "bias"}

    assert observed["weight"].shape == params["weight"].shape
    assert observed["bias"].shape == params["bias"].shape

    assert dnp.all(dnp.isfinite(observed["weight"]))
    assert dnp.all(dnp.isfinite(observed["bias"]))


def test_doubtscore_preserves_parameter_pytree():

    import duvida.numpy as dnp

    from duvida import doubtscore

    def model(x, params):
        return x[:, None] * params["weight"] + params["bias"]

    def loss(
        prediction,
        target,
    ):
        return dnp.sum(dnp.square(prediction - target))

    params = {
        "weight": dnp.array([2., 4.]),
        "bias": dnp.array([1., -1.]),
    }

    x_true = dnp.array([1., 2., 3.])

    x = dnp.array([1.5, 2.5])

    y_true = dnp.zeros((3, 2))
  
    observed = doubtscore(model, loss)(
        (params,),
        x,
        x_true,
        y_true,
    )[0]

    assert set(observed) == {"weight", "bias"}
    assert observed["weight"].shape == (2, 2, 2)
    assert observed["bias"].shape == (2, 2, 2)


def test_information_sensitivity_preserves_parameter_pytree():

    import duvida.numpy as dnp

    from duvida import information_sensitivity

    def model(
        x,
        params,
    ):
        return (
            x[:, None]
            * params["weight"]
            + params["bias"]
        )

    def loss(
        prediction,
        target,
    ):
        return dnp.sum(
            dnp.square(
                prediction - target
            )
        )

    params = {
        "weight": dnp.array([2., 4.]),
        "bias": dnp.array([1., -1.]),
    }

    x_true = dnp.array([1., 2., 3.])

    x = dnp.array([1.5, 2.5])

    y_true = dnp.zeros((3, 2))

    observed = information_sensitivity(
        model,
        loss,
        approximator="squared_jacobian",
    )(
        (params,),
        x,
        x_true,
        y_true,
    )[0]

    assert set(observed) == {"weight", "bias"}
    assert observed["weight"].shape == (2, 2, 2)
    assert observed["bias"].shape == (2, 2, 2)
