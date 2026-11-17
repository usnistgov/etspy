"""Test config file."""

import contextlib

import astra
import matplotlib as mpl
import numpy as np
import pytest

from etspy import datasets as ds

from . import load_serialem_multiframe_data

mpl.use("Agg")


# Shared test data, created once per session. Tests must not modify these in
# place: work on a ``deepcopy()`` instead.


@pytest.fixture(scope="session")
def full_stack():
    """Load the experimental needle stack."""
    return ds.get_needle_data()


@pytest.fixture(scope="session")
def short_stack(full_stack):
    """Truncate the needle stack to 5 images."""
    return full_stack.inav[0:5].deepcopy()


@pytest.fixture(scope="session")
def aligned_full_stack(full_stack):
    """Register the needle stack with phase correlation."""
    return full_stack.stack_register("PC")


@pytest.fixture(scope="session")
def aligned_short_stack(short_stack):
    """Register the truncated needle stack with phase correlation."""
    return short_stack.stack_register("PC")


@pytest.fixture(scope="session")
def catalyst_stack():
    """Load the simulated catalyst stack."""
    return ds.get_catalyst_data()


@pytest.fixture(scope="session")
def multiframe_stack():
    """Load the SerialEM multi-frame test stack."""
    return load_serialem_multiframe_data()


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
