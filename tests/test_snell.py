import numpy as np
import pytest

from example.refraction import snell


# For any indexes, a ray normal to the surface should not bend.
# We'll try a couple different combinations of indexes....


def test_perpendicular_1():
    actual = snell(0, 2.00, 3.00)
    assert actual == pytest.approx(0)


def test_perpendicular_2():
    actual = snell(0, 3.00, 2.00)
    assert actual == pytest.approx(0)


def test_air_water():
    n_air, n_water = 1.00, 1.33
    actual = snell(np.pi / 4, n_air, n_water)
    expected = 0.5605584137424605
    assert actual == pytest.approx(expected)