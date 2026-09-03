def test_parameter_gradient_preserves_output_dimensions():

    import duvida.numpy as dnp

    from duvida import parameter_gradient

    def model(
        x,
        weight
    ):
        return dnp.stack(
            (
                x * weight,
                x * weight ** 2,
            ),
            axis=-1,
        )

    x = dnp.array([1., 2., 3.])

    params = dnp.array([2.])
    gradient = parameter_gradient(model)(
        params,
        x,
    )

    assert gradient.shape == (3, 2, 1)

    expected = dnp.stack(
        (x, 2. * params * x),
        axis=-1,
    )[...,None]

    assert dnp.allclose(gradient, expected)


def test_parameter_gradient_preserves_param_dimensions():

    import duvida.numpy as dnp

    from duvida import parameter_gradient

    def model(
        x,
        weights
    ):
        return x[:, None] * weights

    x = dnp.array([1., 2., 3.])

    params = dnp.array([2., 4.])
    gradient = parameter_gradient(model)(
        (params,),
        x,
    )[0]

    assert gradient.shape == (3, 2, 2)
