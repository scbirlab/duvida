import pytest


def test_chunk_size_validation():

    from duvida.utils import chunked_vmap

    f = lambda z: z * 2

    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError):
            chunked_vmap(f, chunk_size=bad)


def test_env_positive_int(monkeypatch):

    from duvida.utils import _env_positive_int

    monkeypatch.delenv("DUVIDA_TEST_VALUE", raising=False)
    assert _env_positive_int("DUVIDA_TEST_VALUE", 7) == 7
    monkeypatch.setenv("DUVIDA_TEST_VALUE", "12")
    assert _env_positive_int("DUVIDA_TEST_VALUE", 7) == 12
    for bad in ("0", "-4", "abc", "1.5", ""):
        monkeypatch.setenv("DUVIDA_TEST_VALUE", bad)
        with pytest.raises(ValueError):
            _env_positive_int("DUVIDA_TEST_VALUE", 7)


def test_chunked_vmap_matches_vmap_for_pytrees_and_unmapped_args():

    import numpy as np

    import duvida.numpy as dnp

    from duvida.utils import chunked_vmap, vmap

    def f(tree, scale, bias):
        return {"s": dnp.sum(tree["a"]) * scale + bias, "v": tree["b"] * scale}

    tree = {"a": dnp.arange(5 * 3).reshape(5, 3) * 1., "b": dnp.arange(5) * 1.}
    scale, bias = dnp.asarray(2.), dnp.asarray(.5)

    expected = vmap(f, in_axes=(0, None, None))(tree, scale, bias)
    for chunk_size in (None, 1, 2, 5, 9):
        observed = chunked_vmap(
            f, chunk_size=chunk_size, in_axes=(0, None, None),
        )(tree, scale, bias)
        for key in expected:
            assert np.allclose(np.asarray(observed[key]), np.asarray(expected[key]))


def test_chunked_vmap_out_axes_and_nonleading_in_axes():

    import numpy as np

    import duvida.numpy as dnp

    from duvida.utils import chunked_vmap, vmap

    f = lambda row: dnp.stack((row, row ** 2), axis=0)   # (3,) -> (2, 3)
    x = dnp.arange(4 * 3).reshape(3, 4) * 1.              # map over axis 1

    expected = vmap(f, in_axes=1, out_axes=-1)(x)
    for chunk_size in (1, 3, 4):
        observed = chunked_vmap(f, chunk_size=chunk_size, in_axes=1, out_axes=-1)(x)
        assert observed.shape == expected.shape
        assert np.allclose(np.asarray(observed), np.asarray(expected))


def test_jax_chunked_vmap_rejects_unsupported_specs():

    from duvida.config import config

    if config.backend != "jax":
        pytest.skip("explicit rejection is specific to the JAX implementation")

    from duvida.utils import chunked_vmap

    f = lambda a, b: a + b
    with pytest.raises(ValueError):
        chunked_vmap(f, chunk_size=2, in_axes=({"x": 0}, None))  # nested in_axes
    with pytest.raises(ValueError):
        chunked_vmap(f, chunk_size=2, out_axes=(0, 1))
