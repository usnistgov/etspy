"""Tests for composed alignment transforms."""

import numpy as np
import pytest

from etspy import datasets as ds
from etspy import io
from etspy.base import IDENTITY_TRANSFORM, TomoTransforms
from etspy.tests.phantoms import ncc

MIN_LEGACY_NCC = 0.98
MIN_REAPPLIED_NCC = 0.9999


@pytest.fixture(scope="module")
def short_stack():
    """Five projections of the needle dataset."""
    return ds.get_needle_data().inav[0:5].deepcopy()


def _assert_reapplied(aligned, raw, min_ncc=MIN_REAPPLIED_NCC):
    reapplied = aligned.apply_transforms(raw)
    assert reapplied.data.shape == aligned.data.shape
    assert ncc(reapplied.data, aligned.data) > min_ncc


class TestTransforms:
    """Test that every alignment operation records its geometry."""

    def test_default_is_identity(self, short_stack):
        assert isinstance(short_stack.transforms, TomoTransforms)
        np.testing.assert_array_equal(
            short_stack.transforms.data,
            np.tile(IDENTITY_TRANSFORM, (5, 1)),
        )

    def test_navigation_slicing(self, short_stack):
        shifted = short_stack.trans_stack(xshift=2)
        assert shifted.inav[1:3].transforms.data.shape == (2, 6)

    def test_remove_projections(self, short_stack):
        shifted = short_stack.stack_register("PC")
        removed = shifted.remove_projections([1])
        np.testing.assert_array_equal(
            removed.transforms.data,
            shifted.transforms.data[[0, 2, 3, 4]],
        )

    def test_save_load_round_trip(self, short_stack, tmp_path):
        aligned = short_stack.stack_register("PC").trans_stack(angle=1.5, yshift=2)
        filename = tmp_path / "aligned.hspy"
        aligned.save(filename)
        loaded = io.load(filename)
        np.testing.assert_allclose(loaded.transforms.data, aligned.transforms.data)
        assert loaded.metadata.Tomography.yshift == aligned.metadata.Tomography.yshift

    def test_registration_rotation_and_crop(self, short_stack):
        aligned = short_stack.stack_register("PC", crop=True).trans_stack(
            xshift=3,
            yshift=-2,
            angle=2,
        )
        _assert_reapplied(aligned, short_stack)

    def test_tilt_align(self, short_stack):
        aligned = short_stack.stack_register("PC").tilt_align("MaxImage")
        _assert_reapplied(aligned, short_stack)

    def test_manual_align(self, short_stack):
        aligned = short_stack.manual_align(2, xshift=3, yshift=-2)
        _assert_reapplied(aligned, short_stack)

    def test_filter_taper(self, short_stack):
        padded = short_stack.filter("median", size=1, taper=0.1)
        reapplied = padded.apply_transforms(short_stack)
        interior = (slice(None), slice(30, -30), slice(30, -30))
        np.testing.assert_allclose(
            reapplied.data[interior],
            padded.data[interior],
            rtol=1e-4,
            atol=1e-3 * np.ptp(short_stack.data),
        )

    def test_legacy_file_conversion(self):
        # HAADF_Aligned.hspy was saved before transforms were recorded
        aligned = ds.get_needle_data(aligned=True)
        raw = ds.get_needle_data()
        _assert_reapplied(aligned, raw, MIN_LEGACY_NCC)
