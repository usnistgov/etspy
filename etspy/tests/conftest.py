"""Test config file."""

import contextlib

import astra
import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")


@pytest.fixture
def astra_sentinel():
    """
    Hold an unrelated ASTRA object for the duration of a test.

    Tests that count ASTRA objects use this so that a global ``astra.clear()``
    (which would also delete this object) is detected as a change in the count.
    """
    vol = np.ones((4, 4), np.float32)
    sentinel_id = astra.data2d.create("-vol", astra.create_vol_geom(4, 4), vol)
    yield sentinel_id
    with contextlib.suppress(Exception):
        astra.data2d.delete(sentinel_id)
