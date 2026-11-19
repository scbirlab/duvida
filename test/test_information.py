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

    target = dnp.zeros((3, 2))

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

    target = dnp.zeros((3, 2))
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


def _is_fixture(scale=1., dtype=None):
    import numpy as np

    import duvida.numpy as dnp

    dtype = dtype or np.float64

    def model(x, weights):
        # Every output depends on every weight, so no gradient is zero.
        return x * dnp.sum(weights ** 3)

    def loss(prediction, target):
        return dnp.sum(dnp.square(prediction - target))

    def arr(values):
        return dnp.asarray(np.array(values, dtype=dtype))

    return (
        model, loss, 
        (arr([2., 4.]) * scale,), 
        arr([1., 2., 3.]), arr([1.5, 2.5]), arr([1., 1.]),
    )


def test_information_sensitivity_is_identical_to_its_two_terms():

    import numpy as np

    from duvida import information_sensitivity
    from duvida.information import (
        _information_sensitivity_term1,
        _information_sensitivity_term2,
    )

    for dtype in ("float32", "float64"):
        for scale in (1., 1e-2, 1e-4):
            model, loss, params, x, x_true, y_true = _is_fixture(scale, dtype)
            observed = information_sensitivity(model, loss)(
                params, x, x_true, y_true,
            )[0]
            term1 = _information_sensitivity_term1(model, loss)(
                params, x, x_true, y_true,
            )[0]
            term2 = _information_sensitivity_term2(model, loss)(
                params, x, x_true, y_true,
            )[0]
            # Same arithmetic, so bit-identical even where it is ill-conditioned
            assert np.array_equal(
                np.asarray(observed), np.asarray(term1 - term2), equal_nan=True,
            )


def test_information_sensitivity_zero_gradient_is_not_finite():

    import numpy as np

    from duvida import information_sensitivity

    model, loss, params, x, x_true, y_true = _is_fixture(0.)
    observed = np.asarray(
        information_sensitivity(model, loss)(params, x, x_true, y_true)[0]
    )
    assert not np.isfinite(observed).any()


def test_parameter_gradient_chunk_size_does_not_change_result():

    import numpy as np

    import duvida.numpy as dnp

    from duvida import parameter_gradient

    def model(x, w):
        return x[:, None] * w ** 2

    x = dnp.array([1., 2., 3., 4., 5.])
    params = (dnp.array([1., 2., 3.]),)

    expected = np.asarray(parameter_gradient(model)(params, x)[0])
    for chunk_size in (1, 2, 4, 100):
        observed = parameter_gradient(model, chunk_size=chunk_size)(params, x)[0]
        assert np.allclose(np.asarray(observed), expected)
