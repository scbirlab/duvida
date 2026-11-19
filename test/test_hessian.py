def test_parameter_hessian_preserves_output_dimensions():

    import duvida.numpy as dnp

    from duvida import parameter_hessian_diagonal

    def model(x, weight):
        return dnp.stack(
            (x * weight ** 2, x * weight ** 3),
            axis=-1,
        )

    x = dnp.array([1., 2., 3.])

    params = dnp.array([2.])

    observed = parameter_hessian_diagonal(model)(params, x)

    assert observed.shape == (3, 2, 1)

    expected = dnp.stack(
        (2. * x, 6. * params * x),
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


def test_exact_diagonal_methods_agree():

    import duvida.numpy as dnp

    from duvida.hessians import exact_diagonal

    def model(params):
        return dnp.stack(
            (
                dnp.sum(params ** 3) + params[0] * params[1],
                dnp.sum(params ** 2),
                params[1] ** 4,
            ),
            axis=-1,
        )

    params = dnp.array([1., 2., 3.])

    forward = exact_diagonal(model, method="forward")(params)
    hvp_based = exact_diagonal(model, method="hvp")(params)
    auto = exact_diagonal(model)(params)

    expected = dnp.stack(
        (
            6. * params,
            2. * dnp.ones_like(params),
            12. * params[1] ** 2 * dnp.array([0., 1., 0.]),
        ),
        axis=0,
    )

    assert forward.shape == (3, 3)
    assert dnp.allclose(forward, expected)
    assert dnp.allclose(hvp_based, expected)
    assert dnp.allclose(auto, expected)


def test_exact_diagonal_scalar_output_methods_agree():

    import duvida.numpy as dnp

    from duvida.hessians import exact_diagonal

    def f(params):
        return dnp.sum(params ** 3) + dnp.sum(params) ** 2

    params = dnp.array([1., 2., 3.])
    expected = 6. * params + 2.

    for method in ("auto", "forward", "hvp"):
        observed = exact_diagonal(f, method=method)(params)
        assert dnp.allclose(observed, expected)


def test_exact_diagonal_chunk_size_does_not_change_result():
    import duvida.numpy as dnp

    from duvida.hessians import exact_diagonal

    def model(params):
        return dnp.stack((params ** 3, params ** 2), axis=0)

    params = dnp.array([1., 2., 3., 4., 5.])

    full = exact_diagonal(model)(params)

    for method in ("forward", "hvp"):
        for chunk_size in (1, 2, 5, 7):  # incl. remainder and oversized
            chunked = exact_diagonal(
                model, chunk_size=chunk_size, method=method,
            )(params)
            assert chunked.shape == full.shape
            assert dnp.allclose(full, chunked)


def test_default_chunk_size_is_bounded_and_result_unchanged(monkeypatch):

    import duvida.numpy as dnp

    from duvida import hessians, utils

    assert utils.default_chunk_size(1) >= 1
    assert utils.default_chunk_size(10 ** 12) == 1  # never zero
    size = 1000
    chunk = utils.default_chunk_size(size)
    assert 1 <= chunk
    assert chunk * size <= max(utils.MAX_BATCH_ELEMENTS, size)
    assert utils.default_chunk_size(size, cap=3) <= 3

    def model(params):
        return dnp.stack((params ** 3, params ** 2), axis=0)

    params = dnp.array([1., 2., 3., 4., 5.])
    expected = hessians.exact_diagonal(model)(params)

    # Force a tiny budget so several chunks are needed
    monkeypatch.setattr(utils, "MAX_BATCH_ELEMENTS", 8)
    assert utils.default_chunk_size(5) == 1
    assert dnp.allclose(hessians.exact_diagonal(model)(params), expected)
    assert dnp.allclose(
        hessians.exact_diagonal(model, method="hvp")(params), expected,
    )


def test_exact_diagonal_rejects_invalid_arguments():

    import pytest

    import duvida.numpy as dnp

    from duvida.hessians import exact_diagonal

    f = lambda x: dnp.sum(x ** 2)

    with pytest.raises(ValueError):
        exact_diagonal(f, method="backward")
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            exact_diagonal(f, chunk_size=bad)


def test_exact_diagonal_pytree_vector_output_and_argnums():

    import duvida.numpy as dnp

    from duvida.hessians import exact_diagonal

    def model(x, params, scale=1.):
        return scale * dnp.stack(
            (
                x * dnp.sum(params["a"] ** 3) + params["b"][0] ** 2 * x,
                x * params["b"][1] ** 4,
            ),
            axis=-1,
        )

    x = dnp.array([1., 2., 3.])
    params = {"a": dnp.array([1., 2.]), "b": dnp.array([3., 4.])}

    for method in ("forward", "hvp"):
        observed = exact_diagonal(model, argnums=1, method=method)(
            x, params, scale=2.,
        )
        assert set(observed) == {"a", "b"}
        assert observed["a"].shape == (3, 2, 2)
        assert observed["b"].shape == (3, 2, 2)
        # d2/da_i2 of 2 * x * sum(a^3) = 12 * x * a_i (output 0); output 1 has no a
        assert dnp.allclose(
            observed["a"][:, 0, :], 12. * x[:, None] * params["a"][None, :],
        )
        assert dnp.allclose(observed["a"][:, 1, :], dnp.zeros((3, 2)))
        # d2/db_0^2 of 2 * x * b0^2 = 4x; d2/db_1^2 of 2 * x * b1^4 = 24 x b1^2
        assert dnp.allclose(observed["b"][:, 0, 0], 4. * x)
        assert dnp.allclose(observed["b"][:, 0, 1], dnp.zeros((3,)))
        assert dnp.allclose(observed["b"][:, 1, 1], 24. * x * params["b"][1] ** 2)


def test_exact_diagonal_float32_and_float64_agree():

    import numpy as np

    import duvida.numpy as dnp

    from duvida.hessians import exact_diagonal

    f = lambda p: dnp.stack((p ** 3, p ** 2), axis=0)

    for dtype in (np.float32, np.float64):
        params = dnp.asarray(np.array([1., 2., 3.], dtype=dtype))
        observed = exact_diagonal(f)(params)
        assert observed.dtype == params.dtype
        # output (2, 3) by parameter (3): each output depends on one parameter
        expected = np.stack(
            (np.diag(6. * np.array([1., 2., 3.])), np.diag(2. * np.ones(3))), axis=0,
        )
        assert np.allclose(np.asarray(observed), expected, rtol=1e-5)


def test_directional_finite_difference_is_a_hessian_row_sum():
    """`directional_hessian_finite_difference` perturbs every parameter at once, so it
    estimates H @ 1 (row sums), which equals the diagonal only when parameters
    don't interact. Use `exact_diagonal` or `bekas` for a diagonal that holds with
    interacting parameters."""

    import numpy as np

    import duvida.numpy as dnp

    from duvida.hessians import (
        directional_hessian_finite_difference, 
        exact_diagonal,
    )

    f = lambda p: dnp.sum(p) ** 2   # H = 2 * ones((p, p))
    params = dnp.asarray(np.array([.5, 1., 1.5, 2.], dtype=np.float64))

    assert np.allclose(np.asarray(exact_diagonal(f)(params)), 2.)
    assert np.allclose(
        np.asarray(directional_hessian_finite_difference(f)(params)), 
        2. * 4, 
        rtol=1e-5,
    )

    g = lambda p: dnp.sum(p ** 3)    # diagonal Hessian: row sum == diagonal
    assert np.allclose(
        np.asarray(directional_hessian_finite_difference(g)(params)), 
        np.asarray(exact_diagonal(g)(params)),
        rtol=1e-5,
    )


def test_rough_finite_difference_is_a_deprecated_alias():

    import numpy as np
    import pytest

    import duvida.numpy as dnp

    from duvida.hessians import (
        directional_hessian_finite_difference,
        rough_finite_difference,
    )

    f = lambda p: dnp.sum(p ** 3) + dnp.sum(p) ** 2
    params = dnp.asarray(np.array([.5, 1., 1.5], dtype=np.float64))

    with pytest.warns(DeprecationWarning, match="directional_hessian_finite_difference"):
        old = rough_finite_difference(f)(params)

    new = directional_hessian_finite_difference(f)(params)
    assert np.array_equal(np.asarray(old), np.asarray(new))


def test_row_sum_estimator_is_not_advertised_but_resolves_with_a_warning():

    import numpy as np
    import pytest

    import duvida.numpy as dnp

    from duvida.hessians import get_approximators

    names = get_approximators()
    assert "rough_finite_difference" not in names
    assert "directional_hessian_finite_difference" not in names
    assert {"exact_diagonal", "bekas"} <= set(names)

    f = lambda p: dnp.sum(p ** 3)
    params = dnp.asarray(np.array([.5, 1.], dtype=np.float64))

    with pytest.warns(UserWarning, match="not the Hessian diagonal"):
        fn = get_approximators("directional_hessian_finite_difference")
    assert np.allclose(np.asarray(fn(f)(params)), 6. * np.array([.5, 1.]), rtol=1e-5)

    with pytest.warns(UserWarning, match="not the Hessian diagonal"):
        with pytest.warns(DeprecationWarning):
            get_approximators("rough_finite_difference")(f)(params)

    with pytest.raises(NotImplementedError):
        get_approximators("not_a_method")
