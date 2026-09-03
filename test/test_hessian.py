def test_parameter_hessian_preserves_output_dimensions():

    import duvida.numpy as dnp

    from duvida import parameter_hessian_diagonal

    def model(x, weight):
        return dnp.stack(
            (
                x * weight ** 2,
                x * weight ** 3,
            ),
            axis=-1,
        )

    x = dnp.array([1., 2., 3.])

    params = dnp.array([2.])

    observed = parameter_hessian_diagonal(model)(params, x)

    assert observed.shape == (3, 2, 1)

    expected = dnp.stack(
        (
            2. * x,
            6. * params * x,
        ),
        axis=-1,
    )[..., None]

    assert dnp.allclose(observed, expected)


def test_parameter_hessian_preserves_param_dimensions():

    import duvida.numpy as dnp

    from duvida import parameter_hessian_diagonal

    def model(
        x,
        weights,
    ):
        return (
            x[:, None]
            * weights ** 3
        )

    x = dnp.array([
        1.,
        2.,
        3.,
    ])

    params = dnp.array([
        2.,
        4.,
    ])

    observed = parameter_hessian_diagonal(
        model
    )(
        (params,),
        x,
    )[0]

    assert observed.shape == (
        3,
        2,
        2,
    )

    assert dnp.allclose(
        observed[..., 0],
        dnp.stack(
            (
                6. * x * params[0],
                dnp.zeros_like(x),
            ),
            axis=-1,
        ),
    )

    assert dnp.allclose(
        observed[..., 1],
        dnp.stack(
            (
                dnp.zeros_like(x),
                6. * x * params[1],
            ),
            axis=-1,
        ),
    )


def test_hessian_approximators_preserve_structure():

    import duvida.numpy as dnp

    from duvida import parameter_hessian_diagonal

    def model(
        x,
        weights,
    ):
        return (
            x[:, None]
            * weights ** 3
        )

    x = dnp.array([
        1.,
        2.,
        3.,
    ])

    params = (
        dnp.array([
            2.,
            4.,
        ]),
    )

    for approximator in (
        "exact_diagonal",
        "squared_jacobian",
        "bekas",
        "rough_finite_difference",
    ):

        observed = parameter_hessian_diagonal(
            model,
            approximator=approximator,
        )(
            params,
            x,
        )

        assert isinstance(
            observed,
            tuple,
        )

        assert observed[0].shape == (
            3,
            2,
            2,
        )


def test_hessian_approximators_accept_parameter_pytree():

    import duvida.numpy as dnp

    from duvida.hessians import get_approximators

    def model(params):
        return params["left"] ** 3 + dnp.sum(params["right"] ** 3)

    params = {
        "left": dnp.array([
            2.,
        ]),
        "right": dnp.array([
            3.,
            4.,
        ]),
    }

    for name in get_approximators():
        observed = get_approximators(name)(model)(params)

        assert set(observed) == {"left", "right"}
        assert observed["left"].shape == (1, 1)
        assert observed["right"].shape == (1, 2)
