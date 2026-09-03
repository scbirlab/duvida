def test_ravel_pytree_preserves_leading_dimensions():

    import duvida.numpy as dnp

    from duvida.utils import ravel_pytree

    params = {
        "a": dnp.ones((2,)),
        "b": dnp.ones((2, 2)),
    }

    flat, unravel = ravel_pytree(params)

    assert flat.shape == (6,)

    values = dnp.ones((3, 4, 6))

    rebuilt = unravel(values)

    assert rebuilt["a"].shape == (3, 4, 2)
    assert rebuilt["b"].shape == (3, 4, 2, 2)
