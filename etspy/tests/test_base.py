# ruff: noqa: PLR2004

"""Tests for base functions of ETSpy."""

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import astra
import h5py
import hyperspy.api as hs
import numpy as np
import pytest
from hyperspy._signals.signal2d import Signal2D
from hyperspy.io import load as hs_load
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from etspy import datasets as ds
from etspy.align import TiltCOMAligner, TiltMaxImageAligner
from etspy.base import CommonStack, RecStack, TomoShifts, TomoStack, TomoTilts

from . import load_serialem_multiframe_data

if TYPE_CHECKING:
    from hyperspy.axes import UniformDataAxis as Uda
    from hyperspy.misc.utils import DictionaryTreeBrowser

NUM_AXES_THREE = 3


@pytest.fixture(scope="module")
def short_stack():
    """Load stack from test data and truncate to 5 images."""
    return ds.get_needle_data().inav[0:5]


@pytest.fixture(scope="module")
def full_stack():
    """Load stack from test data."""
    return ds.get_needle_data()


@pytest.fixture(scope="module")
def aligned_short_stack():
    """Create truncated and spatially registered stack from test data."""
    s = ds.get_needle_data().inav[0:5]
    s = s.stack_register("PC")
    return s


@pytest.fixture(scope="module")
def aligned_full_stack():
    """Create full spatially registered stack from test data."""
    s = ds.get_needle_data()
    s = s.stack_register("PC")
    return s


@pytest.fixture(scope="module")
def multiframe_stack():
    """Create multiframe stack."""
    s = load_serialem_multiframe_data()
    return s


def _set_tomo_metadata(s: Signal2D) -> Signal2D:
    tomo_metadata = {
        "cropped": False,
        "tiltaxis": 1,
        "xshift": 2,
        "yshift": 3,
    }
    s.metadata.add_node("Tomography")
    cast(
        "DictionaryTreeBrowser",
        s.metadata.Tomography,
    ).add_dictionary(tomo_metadata)
    return s


class TestCommonStack:
    """Test methods of CommonStack (creation shouldn't be possible)."""

    def test_commonstack(self):
        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                "CommonStack should not be instantiated directly. Use one of "
                "its sub-classes instead (TomoStack or RecStack)",
            ),
        ):
            CommonStack(np.random.rand(10, 100, 100))

    def test_slicing(self, short_stack):
        assert short_stack.data.shape == (5, 256, 256)
        assert short_stack.shifts.data.shape == (5, 2)
        assert short_stack.tilts.data.shape == (5, 1)

    def test_plot(self, short_stack):
        short_stack.plot()
        f = plt.gcf()
        plt.close(f)

    def test_save(self, tmp_path, short_stack):
        fname = tmp_path / "save_test.hspy"
        short_stack.save(fname, file_format="HSPY")
        with h5py.File(fname, "r") as h5:
            data = h5.get("/Experiments/__unnamed__/data")
            assert data.shape == (5, 256, 256)  # type: ignore
            tilts = h5.get("/Experiments/__unnamed__/metadata/Tomography/_sig_tilts")
            assert tilts.get("data").shape == (5, 1)  # type: ignore
            shifts = h5.get("/Experiments/__unnamed__/metadata/Tomography/_sig_shifts")
            assert shifts.get("data").shape == (5, 2)  # type: ignore

    def test_save_raw(self, tmp_path, short_stack):
        os.chdir(tmp_path)
        fname = short_stack.save_raw()
        hs_s = hs_load(fname)
        assert isinstance(hs_s, Signal2D)
        assert isinstance(fname, Path)
        assert fname.exists()

    def test_save_raw_with_fname(self, tmp_path, short_stack):
        output_file = tmp_path / "test.rpl"
        fname = short_stack.save_raw(filename=output_file)
        hs_s = hs_load(fname)
        assert isinstance(hs_s, Signal2D)
        assert isinstance(fname, Path)
        assert fname.exists()
        assert fname == output_file.parent / "test_5x256x256_float32.rpl"

    def test_save_raw_with_fname_str(self, tmp_path, short_stack):
        output_file = tmp_path / "test.rpl"
        output_file_str = str(output_file)
        fname = short_stack.save_raw(filename=output_file_str)
        hs_s = hs_load(fname)
        assert isinstance(hs_s, Signal2D)
        assert isinstance(fname, Path)
        assert fname.exists()
        assert fname == output_file.parent / "test_5x256x256_float32.rpl"


class TestTomoStack:
    """Test creation of a TomoStack."""

    def test_tomostack_create_by_signal(self):
        s = hs.signals.Signal2D(np.random.random([5, 100, 100]))
        stack = TomoStack(s)
        assert isinstance(stack, TomoStack)
        assert hasattr(stack, "tilts")
        assert hasattr(stack, "shifts")
        assert stack.metadata.get_item("Tomography.xshift") == 0
        assert stack.metadata.get_item("Tomography.yshift") == 0

    def test_tomostack_create_by_signal_with_existing_tomometa(self):
        s = hs.signals.Signal2D(np.random.random([5, 100, 100]))
        s = _set_tomo_metadata(s)
        s.metadata.set_item("General.title", "Test title")
        stack = TomoStack(s)
        assert isinstance(stack, TomoStack)
        assert hasattr(stack, "tilts")
        assert hasattr(stack, "shifts")
        assert not stack.metadata.get_item("Tomography.cropped")
        assert stack.metadata.get_item("Tomography.xshift") == 2
        assert stack.metadata.get_item("Tomography.yshift") == 3
        assert stack.metadata.get_item("Tomography.tiltaxis") == 1
        assert stack.metadata.get_item("General.title") == "Test title"

    def test_tomostack_create_by_signal_with_tomometa_dict(self):
        s = hs.signals.Signal2D(np.random.random([5, 100, 100]))
        meta_dict = {
            "General": {"title": "test signal"},
            "Tomography": {
                "cropped": False,
                "tiltaxis": -5,
                "xshift": 10,
                "yshift": 20,
            },
        }
        stack = TomoStack(s, metadata=meta_dict)
        assert isinstance(stack, TomoStack)
        assert hasattr(stack, "tilts")
        assert hasattr(stack, "shifts")
        assert not stack.metadata.get_item("Tomography.cropped")
        assert stack.metadata.get_item("Tomography.xshift") == 10
        assert stack.metadata.get_item("Tomography.yshift") == 20
        assert stack.metadata.get_item("Tomography.tiltaxis") == -5
        assert stack.metadata.get_item("General.title") == "test signal"

    def test_tomostack_create_by_signal_without_tomometa_dict(self):
        s = hs.signals.Signal2D(np.random.random([5, 100, 100]))
        meta_dict = {
            "General": {"title": "test signal"},
        }
        stack = TomoStack(s, metadata=meta_dict)
        assert isinstance(stack, TomoStack)
        assert hasattr(stack, "tilts")
        assert hasattr(stack, "shifts")
        assert not stack.metadata.get_item("Tomography.cropped")
        assert stack.metadata.get_item("Tomography.xshift") == 0
        assert stack.metadata.get_item("Tomography.yshift") == 0
        assert stack.metadata.get_item("Tomography.tiltaxis") == 0
        assert stack.metadata.get_item("General.title") == "test signal"

    def test_tomostack_create_by_signal_original_metadata(self):
        s = cast("Signal2D", hs.signals.Signal2D(np.random.random([5, 100, 100])))
        s.metadata.set_item("General.title", "original_metadata test")
        s.original_metadata.set_item("Level1.level2", "the value")
        stack = TomoStack(s)
        assert stack.metadata.get_item("General.title") == "original_metadata test"
        assert stack.original_metadata.get_item("Level1.level2") == "the value"

    def test_tomostack_create_by_signal_original_metadata_arg(self):
        s = cast("Signal2D", hs.signals.Signal2D(np.random.random([5, 100, 100])))
        s.metadata.set_item("General.title", "original_metadata test")
        s.original_metadata.set_item("Level1.level2", "the value")
        stack = TomoStack(s, original_metadata={"A1": {"B1": "C", "B2": "C2"}})
        assert stack.metadata.get_item("General.title") == "original_metadata test"
        assert not stack.original_metadata.has_item("Level1.level2")
        assert stack.original_metadata.get_item("A1.B1") == "C"
        assert stack.original_metadata.get_item("A1.B2") == "C2"

    def test_tomostack_create_by_signal_axes(self):
        s = cast("Signal2D", hs.signals.Signal2D(np.random.random([5, 100, 100])))
        s.axes_manager[0].name = "Test nav"  # type: ignore
        s.axes_manager[0].units = "Nav units"  # type: ignore
        stack = TomoStack(s)
        assert stack.axes_manager[0].name == "Test nav"  # type: ignore
        assert stack.axes_manager[0].units == "Nav units"  # type: ignore

    def test_tomostack_create_by_signal_axes_list_arg(self):
        s = cast("Signal2D", hs.signals.Signal2D(np.random.random([5, 100, 100])))
        ax_list = [
            {
                "_type": "UniformDataAxis",
                "name": "nav_from_list",
                "units": "test units",
                "navigate": True,
                "size": 5,
                "scale": 1.0,
                "offset": 1,
            },
            {
                "_type": "UniformDataAxis",
                "name": "test x",
                "units": "nm",
                "navigate": False,
                "size": 100,
                "scale": 0.123,
                "offset": 0,
            },
            {
                "_type": "UniformDataAxis",
                "name": "test y",
                "units": "nm",
                "navigate": False,
                "size": 100,
                "scale": 0.321,
                "offset": 0,
            },
        ]
        s.axes_manager[0].name = "Test nav"  # type: ignore
        s.axes_manager[0].units = "Nav units"  # type: ignore
        stack = TomoStack(s, axes=ax_list)
        assert stack.axes_manager[0].name == "nav_from_list"  # type: ignore
        assert stack.axes_manager[0].units == "test units"  # type: ignore
        # signal dimensions are always renamed to just "x" and "y":
        assert stack.axes_manager[-1].name == "y"  # type: ignore
        assert stack.axes_manager[-1].scale == 0.123  # type: ignore

    def test_tomostack_create_by_signal_undefined_axes(self):
        s = cast("Signal2D", hs.signals.Signal2D(np.random.random([5, 100, 100])))
        axes = [
            {"size": 5, "name": "Axis0", "units": ""},
            {"size": 100, "name": "Axis1", "units": ""},
            {"size": 100, "name": "Axis2", "units": ""},
        ]
        stack = TomoStack(s, axes=axes)
        assert stack.axes_manager[0].name == "Projections"  # type: ignore
        assert stack.axes_manager[0].units == "degrees"  # type: ignore

    def test_tomostack_create_by_array_multiframe(self):
        n = np.random.random([5, 2, 110, 120])
        stack = TomoStack(n)
        ax0, ax1, ax2, ax3 = (cast("Uda", stack.axes_manager[i]) for i in range(4))
        assert ax0.name == "Frames"
        assert ax1.name == "Projections"
        assert ax2.name == "x"
        assert ax3.name == "y"
        assert ax0.units == "images"
        assert ax1.units == "degrees"
        assert ax2.units == "pixels"
        assert ax3.units == "pixels"
        assert ax0.size == 2
        assert ax1.size == 5
        assert ax2.size == 120
        assert ax3.size == 110

    def test_tomostack_create_by_array_too_many_dims(self):
        n = np.random.rand(2, 3, 5, 2, 1, 3)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Invalid number of navigation dimensions for a TomoStack (4). "
                "Must be either 0, 1, or 2.",
            ),
        ):
            TomoStack(n)

    def test_deepcopy(self, short_stack):
        s2 = short_stack.deepcopy()
        assert np.all(short_stack.data == s2.data)
        assert np.all(short_stack.tilts.data == s2.tilts.data)
        assert np.all(short_stack.shifts.data == s2.shifts.data)
        assert short_stack is not s2
        assert short_stack.data is not s2.data
        assert short_stack.tilts.data is not s2.tilts.data
        assert short_stack.shifts.data is not s2.shifts.data
        np.testing.assert_equal(
            short_stack.metadata.as_dictionary(),
            s2.metadata.as_dictionary(),
        )

    def test_copy(self, short_stack):
        s2 = short_stack.copy()
        assert np.all(short_stack.data == s2.data)
        assert np.all(short_stack.tilts.data == s2.tilts.data)
        assert np.all(short_stack.shifts.data == s2.shifts.data)
        assert short_stack is not s2
        assert short_stack.data is s2.data
        assert short_stack.tilts.data is s2.tilts.data
        assert short_stack.shifts.data is s2.shifts.data
        np.testing.assert_equal(
            short_stack.metadata.as_dictionary(),
            s2.metadata.as_dictionary(),
        )

    def test_remove_projections(self, short_stack):
        remove_indices = [0, 1, 3]
        tilts_to_remove = [short_stack.tilts.data[i] for i in remove_indices]
        s_new = short_stack.remove_projections(remove_indices)
        # original shape is (5, 256, 256)
        assert s_new.data.shape == (2, 256, 256)
        assert s_new.tilts.data.shape == (2, 1)
        assert s_new.shifts.data.shape == (2, 2)
        for t in tilts_to_remove:
            assert t not in s_new.tilts.data

    def test_remove_projections_none(self, short_stack):
        with pytest.raises(ValueError, match="No projections provided"):
            short_stack.remove_projections(None)

    def test_plot_sinos(self, short_stack):
        short_stack.plot_sinos()
        f = plt.gcf()
        xlim = f.axes[0].get_xlim()
        ylim = f.axes[0].get_ylim()
        assert xlim[0] == pytest.approx(-1.68)
        assert xlim[1] == pytest.approx(858.48)
        assert ylim[0] == pytest.approx(4.5)
        assert ylim[1] == pytest.approx(-0.5)
        assert f.axes[0].title.get_text() == " Signal"
        assert f.axes[0].get_xlabel() == "y axis (nm)"
        assert f.axes[0].get_ylabel() == "Projections axis (degrees)"
        plt.close(f)


class TestProperties:
    """Test tilt and shift properties."""

    def test_tomotilt_constructor_hs_signal(self):
        n = np.random.rand(77, 1)
        sig = hs.signals.Signal1D(n)
        tilt = TomoTilts(sig)
        assert sig.axes_manager.shape == (77, 1)
        assert tilt.axes_manager.shape == (77, 1)
        assert sig.data.shape == (77, 1)
        assert tilt.data.shape == (77, 1)

    def test_tomotilt_constructor_ndarray(self):
        n = np.random.rand(77, 1)
        tilt = TomoTilts(n)
        assert tilt.axes_manager.shape == (77, 1)
        assert tilt.data.shape == (77, 1)

    def test_tomotilt_constructor_hs_signal_bad_dims(self):
        n = np.random.rand(77, 5)
        sig = hs.signals.Signal1D(n)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Tilt values must have a signal shape of (1,), but was (5,)",
            ),
        ):
            TomoTilts(sig)

    def test_tomotilt_constructor_ndarray_bad_dims(self):
        n = np.random.rand(77, 5)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Tilt values must have a signal shape of (1,), but was (5,)",
            ),
        ):
            TomoTilts(n)

    def test_tomoshift_constructor_hs_signal(self):
        n = np.random.rand(77, 2)
        sig = hs.signals.Signal1D(n)
        tilt = TomoShifts(sig)
        assert sig.axes_manager.shape == (77, 2)
        assert tilt.axes_manager.shape == (77, 2)
        assert sig.data.shape == (77, 2)
        assert tilt.data.shape == (77, 2)

    def test_tomoshift_constructor_ndarray(self):
        n = np.random.rand(77, 2)
        tilt = TomoShifts(n)
        assert tilt.axes_manager.shape == (77, 2)
        assert tilt.data.shape == (77, 2)

    def test_tomoshift_constructor_hs_signal_bad_dims(self):
        n = np.random.rand(77, 1)
        sig = hs.signals.Signal1D(n)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shift values must have a signal shape of (2,), but was (1,)",
            ),
        ):
            TomoShifts(sig)

    def test_tomoshift_constructor_ndarray_bad_dims(self):
        n = np.random.rand(77, 1)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shift values must have a signal shape of (2,), but was (1,)",
            ),
        ):
            TomoShifts(n)

    def test_tilt_setter(self, full_stack):
        full_stack_random_tilts = full_stack.deepcopy()
        full_stack_random_tilts.tilts = np.random.rand(77, 1)
        assert (
            full_stack_random_tilts.tilts.metadata.get_item("General.title")
            == "Image tilt values"
        )
        assert full_stack_random_tilts.tilts.axes_manager.shape == (77, 1)
        assert full_stack_random_tilts.tilts.data.shape == (77, 1)

    def test_tilt_setter_1d_array(self, full_stack):
        full_stack_random_tilts = full_stack.deepcopy()
        full_stack_random_tilts.tilts = np.random.rand(
            77,
        )  # should be coerced to (77, 1)

        assert (
            full_stack_random_tilts.tilts.metadata.get_item("General.title")
            == "Image tilt values"
        )
        assert full_stack_random_tilts.tilts.axes_manager.shape == (77, 1)
        assert full_stack_random_tilts.tilts.data.shape == (77, 1)
        assert full_stack_random_tilts.tilts.axes_manager[-1].name == "Tilt values"  # type: ignore
        assert full_stack_random_tilts.tilts.axes_manager[-1].units == "degrees"  # type: ignore

        # check that tilt axes info matches signal
        assert (
            full_stack_random_tilts.tilts.axes_manager[0].name
            == full_stack.axes_manager[0].name
        )  # type: ignore
        assert (
            full_stack_random_tilts.tilts.axes_manager[0].units
            == full_stack.axes_manager[0].units
        )  # type: ignore
        assert (
            full_stack_random_tilts.tilts.axes_manager[0].scale
            == full_stack.axes_manager[0].scale
        )  # type: ignore
        assert (
            full_stack_random_tilts.tilts.axes_manager[0].offset
            == full_stack.axes_manager[0].offset
        )  # type: ignore

    def test_tilt_setter_bad_dims(self, full_stack):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of tilts array must be (77, 1) to match the navigation "
                "size of the stack (was (20, 1))",
            ),
        ):
            full_stack.tilts = np.random.rand(20, 1)

    def test_tilt_setter_tomotilt(self, full_stack):
        full_stack_random_tilts = full_stack.deepcopy()
        n = np.random.rand(77, 1)
        tilts = TomoTilts(n)
        assert tilts.metadata.get_item("General.title") == ""
        full_stack_random_tilts.tilts = tilts  # title should be set since it was empty
        assert (
            full_stack_random_tilts.tilts.metadata.get_item("General.title")
            == "Image tilt values"
        )

    def test_tilt_setter_tomotilt_bad_dims(self, full_stack):
        n = np.random.rand(20, 1)
        tilts = TomoTilts(n)
        assert tilts.metadata.get_item("General.title") == ""
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of tilts array must be (77, 1) to match the navigation size "
                "of the stack (was (20, 1))",
            ),
        ):
            full_stack.tilts = tilts

    def test_shift_setter(self, full_stack):
        full_stack_random_shifts = full_stack.deepcopy()
        full_stack_random_shifts.shifts = np.random.rand(77, 2)
        assert (
            full_stack_random_shifts.shifts.metadata.get_item("General.title")
            == "Image shift values"
        )
        assert full_stack_random_shifts.shifts.axes_manager.shape == (77, 2)
        assert full_stack_random_shifts.shifts.data.shape == (77, 2)
        assert (
            full_stack_random_shifts.shifts.axes_manager[-1].name
            == "Shift values (x/y)"
        )  # type: ignore
        assert full_stack_random_shifts.shifts.axes_manager[-1].units == "pixels"  # type: ignore

        # check that tilt axes info matches signal
        assert (
            full_stack_random_shifts.shifts.axes_manager[0].name
            == full_stack_random_shifts.axes_manager[0].name
        )  # type: ignore
        assert (
            full_stack_random_shifts.shifts.axes_manager[0].units
            == full_stack_random_shifts.axes_manager[0].units
        )  # type: ignore
        assert (
            full_stack_random_shifts.shifts.axes_manager[0].scale
            == full_stack.axes_manager[0].scale
        )  # type: ignore
        assert (
            full_stack.shifts.axes_manager[0].offset
            == full_stack_random_shifts.axes_manager[0].offset
        )  # type: ignore

    def test_shift_setter_bad_dims(self, full_stack):
        full_stack_shift_set = full_stack.deepcopy()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of shifts array must be (77, 2) to match the navigation "
                "size of the stack (was (77,))",
            ),
        ):
            full_stack_shift_set.shifts = np.random.rand(77)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of shifts array must be (77, 2) to match the navigation "
                "size of the stack (was (20, 1, 5))",
            ),
        ):
            full_stack_shift_set.shifts = np.random.rand(20, 1, 5)

    def test_shift_setter_tomoshift(self, full_stack):
        full_stack_shift_set = full_stack.deepcopy()
        n = np.random.rand(77, 2)
        shifts = TomoShifts(n)
        assert shifts.metadata.get_item("General.title") == ""
        full_stack_shift_set.shifts = shifts  # title should be set since it was empty
        assert (
            full_stack_shift_set.shifts.metadata.get_item("General.title")
            == "Image shift values"
        )

    def test_shift_setter_tomoshift_bad_dims(self, full_stack):
        n = np.random.rand(20, 2)
        shifts = TomoShifts(n)
        assert shifts.metadata.get_item("General.title") == ""
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of shifts array must be (77, 2) to match the navigation size "
                "of the stack (was (20, 2))",
            ),
        ):
            full_stack.shifts = shifts

    def test_multiframe_tilt_setter(self, multiframe_stack):
        assert multiframe_stack.axes_manager.shape == (2, 3, 1024, 1024)
        assert multiframe_stack.data.shape == (3, 2, 1024, 1024)
        assert multiframe_stack.tilts.axes_manager.shape == (2, 3, 1)
        assert multiframe_stack.tilts.data.shape == (3, 2, 1)

        multiframe_stack_set_tilts = multiframe_stack.deepcopy()
        n = np.random.rand(3, 2, 1)
        multiframe_stack_set_tilts.tilts = n
        assert multiframe_stack_set_tilts.tilts.axes_manager.shape == (2, 3, 1)
        assert multiframe_stack_set_tilts.tilts.data.shape == (3, 2, 1)
        assert (
            multiframe_stack_set_tilts.tilts.metadata.get_item("General.title")
            == "Image tilt values"
        )

    def test_multiframe_tilt_setter_2d(self, multiframe_stack):
        assert multiframe_stack.axes_manager.shape == (2, 3, 1024, 1024)
        assert multiframe_stack.data.shape == (3, 2, 1024, 1024)
        assert multiframe_stack.tilts.axes_manager.shape == (2, 3, 1)
        assert multiframe_stack.tilts.data.shape == (3, 2, 1)

        multiframe_stack_set_tilts = multiframe_stack.deepcopy()
        n = np.random.rand(3, 2)
        multiframe_stack_set_tilts.tilts = n
        assert multiframe_stack_set_tilts.tilts.axes_manager.shape == (2, 3, 1)
        assert multiframe_stack_set_tilts.tilts.data.shape == (3, 2, 1)
        assert (
            multiframe_stack_set_tilts.tilts.metadata.get_item("General.title")
            == "Image tilt values"
        )

    def test_multiframe_tilt_setter_bad_dims(self, multiframe_stack):
        assert multiframe_stack.axes_manager.shape == (2, 3, 1024, 1024)
        assert multiframe_stack.data.shape == (3, 2, 1024, 1024)
        assert multiframe_stack.tilts.axes_manager.shape == (2, 3, 1)
        assert multiframe_stack.tilts.data.shape == (3, 2, 1)

        multiframe_stack_bad_tilt_shape = multiframe_stack.deepcopy()
        n = np.random.rand(2, 3, 1)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of tilts array must be (3, 2, 1) to match the navigation size "
                "of the stack (was (2, 3, 1))",
            ),
        ):
            multiframe_stack_bad_tilt_shape.tilts = n

        n = np.random.rand(3, 2, 10)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of tilts array must be (3, 2, 1) to match the navigation size "
                "of the stack (was (3, 2, 10))",
            ),
        ):
            multiframe_stack_bad_tilt_shape.tilts = n

    def test_multiframe_shift__setter(self, multiframe_stack):
        assert multiframe_stack.axes_manager.shape == (2, 3, 1024, 1024)
        assert multiframe_stack.data.shape == (3, 2, 1024, 1024)
        assert multiframe_stack.shifts.axes_manager.shape == (2, 3, 2)
        assert multiframe_stack.shifts.data.shape == (3, 2, 2)

        n = np.random.rand(3, 2, 2)
        multiframe_stack.shifts = n
        assert multiframe_stack.shifts.axes_manager.shape == (2, 3, 2)
        assert multiframe_stack.shifts.data.shape == (3, 2, 2)
        assert (
            multiframe_stack.shifts.metadata.get_item("General.title")
            == "Image shift values"
        )

    def test_multiframe_shift__setter_bad_dims(self, multiframe_stack):
        assert multiframe_stack.axes_manager.shape == (2, 3, 1024, 1024)
        assert multiframe_stack.data.shape == (3, 2, 1024, 1024)
        assert multiframe_stack.shifts.axes_manager.shape == (2, 3, 2)
        assert multiframe_stack.shifts.data.shape == (3, 2, 2)

        n = np.random.rand(2, 3, 2)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of shifts array must be (3, 2, 2) to match the navigation size "
                "of the stack (was (2, 3, 2))",
            ),
        ):
            multiframe_stack.shifts = n

        n = np.random.rand(3, 2, 10)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of shifts array must be (3, 2, 2) to match the navigation size "
                "of the stack (was (3, 2, 10))",
            ),
        ):
            multiframe_stack.shifts = n

    def test_property_deleters(self, full_stack):
        # tilts
        full_stack_no_tilts = full_stack.deepcopy()
        assert np.all(full_stack.tilts.data.squeeze() == np.arange(-76, 78, 2))
        del full_stack_no_tilts.tilts
        assert np.all(full_stack_no_tilts.tilts.data.squeeze() == np.zeros((77, 1)))

        # shifts
        full_stack_no_shifts = full_stack.deepcopy()
        full_stack.shifts = np.random.rand(77, 2) + 2  # offset to ensure non-zero
        assert np.all(full_stack.shifts.data != np.zeros((77, 2)))
        del full_stack_no_shifts.shifts
        assert np.all(full_stack_no_shifts.shifts.data == np.zeros((77, 2)))


class TestSlicers:
    """Test inav/isig slicers."""

    def test_tilt_nav_slicer(self, full_stack):
        t = full_stack.inav[:5].tilts
        assert isinstance(t, TomoTilts)
        assert t.data.shape == (5, 1)

    def test_tilt_sig_slicer(self, caplog, full_stack):
        with caplog.at_level(logging.WARNING):
            t = full_stack.tilts.isig[:5]
            assert "TomoTilts does not support 'isig' slicing" in caplog.text
        assert isinstance(t, TomoTilts)
        assert t.data.shape == (77, 1)

    def test_shift_nav_slicer(self, full_stack):
        t = full_stack.shifts.inav[:5]
        assert isinstance(t, TomoShifts)
        assert t.data.shape == (5, 2)

    def test_shift_sig_slicer(self, caplog, full_stack):
        with caplog.at_level(logging.WARNING):
            t = full_stack.shifts.isig[:5]
            # warning should be triggered when slicing the TomoTilts directly
            assert "TomoShifts does not support 'isig' slicing" in caplog.text
        assert isinstance(t, TomoShifts)
        assert t.data.shape == (77, 2)

    def test_tomostack_nav_slicer(self, full_stack):
        t = full_stack.inav[:5]
        assert t.data.shape == (5, 256, 256)
        assert t.tilts.data.shape == (5, 1)
        assert t.shifts.data.shape == (5, 2)
        assert isinstance(t, TomoStack)
        assert isinstance(t.tilts, TomoTilts)
        assert isinstance(t.shifts, TomoShifts)

    def test_tomostack_sig_slicer(self, caplog, full_stack):
        with caplog.at_level(logging.WARNING):
            t = full_stack.isig[:5, :10]
            # warning should not be triggered when slicing the TomoStack
            assert "TomoShifts does not support 'isig' slicing" not in caplog.text
        assert t.data.shape == (77, 10, 5)
        assert t.tilts.data.shape == (77, 1)
        assert t.shifts.data.shape == (77, 2)
        assert isinstance(t, TomoStack)
        assert isinstance(t.tilts, TomoTilts)
        assert isinstance(t.shifts, TomoShifts)

    def test_two_d_tomo_stack_slicing(self, multiframe_stack):
        """Test handling of multi-frame TomoStack with 2 navigation dimensions."""
        assert multiframe_stack.axes_manager.shape == (2, 3, 1024, 1024)
        assert multiframe_stack.data.shape == (3, 2, 1024, 1024)
        ax_0, ax_1, ax_2, ax_3 = cast(
            "list[Uda]",
            [multiframe_stack.axes_manager[i] for i in range(4)],
        )
        assert ax_0.name == "Frames"
        assert ax_0.units == "images"
        assert ax_1.name == "Projections"
        assert ax_1.units == "degrees"
        assert ax_2.name == "x"
        assert ax_2.units == "nm"
        assert ax_3.name == "y"
        assert ax_3.units == "nm"

        # test inav and isig together with ranges
        t = multiframe_stack.inav[:1, :2].isig[:20, :120]
        assert t.axes_manager.shape == (1, 2, 20, 120)
        assert t.data.shape == (2, 1, 120, 20)
        assert t.tilts.axes_manager.shape == (1, 2, 1)
        assert t.shifts.axes_manager.shape == (1, 2, 2)

        # test extracting single projection
        t2 = multiframe_stack.isig[:20, :120].inav[:, 2]
        assert t2.axes_manager.shape == (2, 20, 120)
        assert t2.data.shape == (2, 120, 20)
        assert t2.axes_manager[0].name == "Frames"  # type: ignore
        assert t2.tilts.axes_manager.navigation_shape == (2,)
        assert t2.shifts.axes_manager.navigation_shape == (2,)

        # test extracting single frame
        t3 = multiframe_stack.isig[:20, :120].inav[1, :]
        assert t3.axes_manager.shape == (3, 20, 120)
        assert t3.data.shape == (3, 120, 20)
        assert t3.axes_manager[0].name == "Projections"  # type: ignore
        assert t3.tilts.axes_manager.navigation_shape == (3,)
        assert t3.shifts.axes_manager.navigation_shape == (3,)

        # test extracting single frame and projection
        t4 = multiframe_stack.isig[:20, :120].inav[1, 1]
        assert t4.axes_manager.shape == (20, 120)
        assert t4.data.shape == (120, 20)
        assert t4.axes_manager[0].name == "x"  # type: ignore
        assert t4.axes_manager[1].name == "y"  # type: ignore
        assert t4.tilts.axes_manager.navigation_shape == ()
        assert t4.shifts.axes_manager.navigation_shape == ()
        assert t4.tilts.data[0] == pytest.approx(-0.000488)

    def test_single_pixel_nav_slicer(self, short_stack):
        t = short_stack.inav[3]
        ax = t.axes_manager
        assert t.data.shape == (256, 256)
        assert ax.navigation_shape == ()
        assert ax.shape == (256, 256)

    def test_single_pixel_sig_slicer_x(self, caplog, short_stack):  # type: ignore
        with caplog.at_level(logging.WARNING):
            # warning should be triggered and shape should be (77, 1, 256)
            t = short_stack.isig[5, :]
            assert (
                "Slicing a TomoStack signal axis with a single pixel "
                'is not supported. Returning a single pixel on the "x" '
                "axis instead"
            ) in caplog.text
        assert t.data.shape == (5, 256, 1)
        assert t.axes_manager.shape == (5, 1, 256)
        ax_0, ax_1, ax_2 = cast("list[Uda]", [t.axes_manager[i] for i in range(3)])
        assert ax_0.name == "Projections"
        assert ax_1.name == "x"
        assert ax_2.name == "y"
        assert ax_0.size == 5
        assert ax_1.size == 1
        assert ax_2.size == 256
        assert ax_0.units == "degrees"
        assert ax_1.units == "nm"
        assert ax_2.units == "nm"
        assert ax_1.offset == pytest.approx(ax_1.scale * 5)
        assert ax_2.offset == 0

    def test_single_pixel_sig_slicer_y(self, caplog, short_stack):
        with caplog.at_level(logging.WARNING):
            # warning should be triggered and shape should be (77, 256, 1)
            t = short_stack.isig[:, 128]
            assert (
                "Slicing a TomoStack signal axis with a single pixel "
                'is not supported. Returning a single pixel on the "y" '
                "axis instead"
            ) in caplog.text
        assert t.data.shape == (5, 1, 256)
        assert t.axes_manager.shape == (5, 256, 1)
        ax_0, ax_1, ax_2 = cast("list[Uda]", [t.axes_manager[i] for i in range(3)])
        assert ax_0.name == "Projections"
        assert ax_1.name == "x"
        assert ax_2.name == "y"
        assert ax_0.size == 5
        assert ax_1.size == 256
        assert ax_2.size == 1
        assert ax_0.units == "degrees"
        assert ax_1.units == "nm"
        assert ax_2.units == "nm"
        assert ax_1.offset == 0
        assert ax_2.offset == pytest.approx(ax_2.scale * 128)

    def test_single_pixel_sig_slicer_float(self, caplog, short_stack):
        with caplog.at_level(logging.WARNING):
            # warning should be triggered and shape should be (77, 256, 1)
            t = short_stack.isig[:, 30.4]
            assert (
                "Slicing a TomoStack signal axis with a single pixel "
                'is not supported. Returning a single pixel on the "y" '
                "axis instead"
            ) in caplog.text
        assert t.data.shape == (5, 1, 256)
        assert t.axes_manager.shape == (5, 256, 1)
        ax_0, ax_1, ax_2 = cast("list[Uda]", [t.axes_manager[i] for i in range(3)])
        assert ax_0.name == "Projections"
        assert ax_1.name == "x"
        assert ax_2.name == "y"
        assert ax_0.size == 5
        assert ax_1.size == 256
        assert ax_2.size == 1
        assert ax_0.units == "degrees"
        assert ax_1.units == "nm"
        assert ax_2.units == "nm"
        assert ax_1.offset == 0
        assert ax_2.offset == pytest.approx(30.24)

    # def test_single_pixel_nav_slicer(self):
    #     pass

    def test_recstack_nav_slicer(self, full_stack):
        stack = RecStack(full_stack)
        t = stack.inav[:5]
        assert t.data.shape == (5, 256, 256)
        assert isinstance(t, RecStack)

    def test_recstack_sig_slicer(self, caplog, full_stack):
        stack = RecStack(full_stack)
        with caplog.at_level(logging.WARNING):
            t = stack.isig[:5, :10]
            # warning should not be triggered when slicing the TomoStack
            assert "TomoShifts does not support 'isig' slicing" not in caplog.text
        assert t.data.shape == (77, 10, 5)
        assert isinstance(t, RecStack)


class TestExtractSinogram:
    """Test extract_sinogram method."""

    def test_extract_sinogram(self, full_stack):
        sino = full_stack.extract_sinogram(128)
        ax_0, ax_1 = cast("list[Uda]", [sino.axes_manager[i] for i in range(2)])
        assert sino.axes_manager.shape == (256, 77)
        assert sino.metadata.get_item("Signal.signal_type") == ""
        assert ax_0.name == "y"
        assert ax_1.name == "Projections"
        assert sino.metadata.get_item("General.title") == "Sinogram at column 128"

    def test_extract_sinogram_float(self, full_stack):
        sino = full_stack.extract_sinogram(106.32)
        ax_0, ax_1 = cast("list[Uda]", [sino.axes_manager[i] for i in range(2)])
        assert sino.axes_manager.shape == (256, 77)
        assert sino.metadata.get_item("Signal.signal_type") == ""
        assert ax_0.name == "y"
        assert ax_1.name == "Projections"
        assert sino.metadata.get_item("General.title") == "Sinogram at x = 106.32 nm"

    def test_extract_sinogram_bad_argument_type(self, full_stack):
        with pytest.raises(
            TypeError,
            match=re.escape(
                '"column" argument must be either a float or an integer '
                "(was <class 'str'>)",
            ),
        ):
            full_stack.extract_sinogram("bad_val")  # type: ignore

    def test_extract_sinogram_exception_handling(self, full_stack):
        # test that on exception, logger is still enabled
        assert logging.getLogger("etspy.base").disabled is False
        with pytest.raises(IndexError):
            # using too large of a value should trigger an error
            full_stack.extract_sinogram(column=10000)
        assert logging.getLogger("etspy.base").disabled is False


class TestFiltering:
    """Test filtering of TomoStack data."""

    def test_correlation_check(self, short_stack):
        fig = short_stack.test_correlation()
        assert isinstance(fig, Figure)
        assert len(fig.axes) == NUM_AXES_THREE

    def test_image_filter_median(self, short_stack):
        filt = short_stack.filter(method="median")
        assert (
            filt.axes_manager.navigation_shape
            == short_stack.axes_manager.navigation_shape
        )
        assert filt.axes_manager.signal_shape == short_stack.axes_manager.signal_shape

    def test_image_filter_sobel(self, short_stack):
        filt = short_stack.filter(method="sobel")
        assert (
            filt.axes_manager.navigation_shape
            == short_stack.axes_manager.navigation_shape
        )
        assert filt.axes_manager.signal_shape == short_stack.axes_manager.signal_shape

    def test_image_filter_both(self, short_stack):
        filt = short_stack.filter(method="both")
        assert (
            filt.axes_manager.navigation_shape
            == short_stack.axes_manager.navigation_shape
        )
        assert filt.axes_manager.signal_shape == short_stack.axes_manager.signal_shape

    def test_image_filter_bpf(self, short_stack):
        filt = short_stack.filter(method="bpf")
        assert (
            filt.axes_manager.navigation_shape
            == short_stack.axes_manager.navigation_shape
        )
        assert filt.axes_manager.signal_shape == short_stack.axes_manager.signal_shape

    def test_image_filter_wrong_name(self, short_stack):
        bad_name = "WRONG"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid filter method "{bad_name}". '
                'Must be one of ["median", "bpf", "both", or "sobel"]',
            ),
        ):
            short_stack.filter(method="WRONG")  # type: ignore


class TestOperations:
    """Test various operations of a TomoStack."""

    def test_stack_normalize(self, short_stack):
        norm = cast("TomoStack", short_stack.normalize())
        assert (
            norm.axes_manager.navigation_shape
            == short_stack.axes_manager.navigation_shape
        )
        assert norm.axes_manager.signal_shape == short_stack.axes_manager.signal_shape
        assert norm.data.min() == 0.0

    def test_stack_invert(self):
        im = np.zeros([10, 100, 100])
        im[:, 40:60, 40:60] = 10
        stack = TomoStack(im)
        invert = cast("TomoStack", stack.invert())
        hist, _ = np.histogram(stack.data)
        hist_inv, _ = np.histogram(invert.data)
        assert hist[0] > hist_inv[0]

    def test_stack_stats(self, capsys, short_stack):
        short_stack.stats()

        # capture output stream to test print statements
        captured = capsys.readouterr()
        out = captured.out.split("\n")

        assert out[0] == f"Mean: {short_stack.data.mean():.1f}"
        assert out[1] == f"Std: {short_stack.data.std():.2f}"
        assert out[2] == f"Max: {short_stack.data.max():.1f}"
        assert out[3] == f"Min: {short_stack.data.min():.1f}"

    def test_set_tilts(self, full_stack):
        start, increment = -50, 5
        full_stack.set_tilts(start, increment)
        ax = cast("Uda", full_stack.axes_manager[0])
        assert ax.name == "Projections"
        assert ax.scale == increment
        assert ax.units == "degrees"
        assert ax.offset == start
        assert (
            ax.axis.all()
            == np.arange(
                start,
                full_stack.data.shape[0] * increment + start,
                increment,
            ).all()
        )

    def test_set_tilts_no_metadata(self, full_stack):
        stack = full_stack.deepcopy()
        del stack.metadata.Tomography  # pyright: ignore[reportAttributeAccessIssue]
        start, increment = -50, 5
        stack.set_tilts(start, increment)
        ax = cast("Uda", stack.axes_manager[0])
        assert ax.name == "Projections"
        assert ax.scale == increment
        assert ax.units == "degrees"
        assert ax.offset == start
        assert (
            ax.axis.all()
            == np.arange(
                start,
                stack.data.shape[0] * increment + start,
                increment,
            ).all()
        )


class TestTestAlign:
    """Test test alignment of a TomoStack."""

    def test_test_align_no_slices(self, aligned_short_stack):
        fig = aligned_short_stack.test_align()
        assert len(fig.axes) == NUM_AXES_THREE

    def test_test_align_with_angle(self, aligned_short_stack):
        fig = aligned_short_stack.test_align(tilt_rotation=3.0)
        assert len(fig.axes) == NUM_AXES_THREE

    def test_test_align_with_xshift(self, aligned_short_stack):
        fig = aligned_short_stack.test_align(tilt_shift=3.0)
        assert len(fig.axes) == NUM_AXES_THREE

    def test_test_align_with_thickness(self, aligned_short_stack):
        fig = aligned_short_stack.test_align(thickness=200)
        assert len(fig.axes) == NUM_AXES_THREE

    def test_test_align_no_cuda(self, aligned_short_stack):
        fig = aligned_short_stack.test_align(thickness=200, cuda=False)
        assert len(fig.axes) == NUM_AXES_THREE

    @patch("astra.use_cuda", new=lambda: False)
    def test_test_align_cuda_none_mock_false(self, aligned_short_stack):
        fig = aligned_short_stack.test_align(thickness=200, cuda=None)
        assert len(fig.axes) == NUM_AXES_THREE

    @patch("matplotlib.get_backend", new=lambda: "widget")
    def test_test_align_mock_mpl_backend_none(self, aligned_short_stack):
        fig = aligned_short_stack.test_align(thickness=200)
        assert len(fig.axes) == NUM_AXES_THREE
        assert np.all(fig.get_size_inches() == np.array([12, 4]))

    @patch("matplotlib.get_backend", new=lambda: "ipympl")
    def test_test_align_mock_mpl_backend_ipympl(self, aligned_short_stack):
        fig = aligned_short_stack.test_align(thickness=200)
        assert len(fig.axes) == NUM_AXES_THREE
        assert np.all(fig.get_size_inches() == np.array([7, 3]))

    @patch("matplotlib.get_backend", new=lambda: "nbagg")
    def test_test_align_mock_mpl_backend_nbagg(self, aligned_short_stack):
        fig = aligned_short_stack.test_align(thickness=200)
        assert len(fig.axes) == NUM_AXES_THREE
        assert np.all(fig.get_size_inches() == np.array([8, 4]))


class TestAlignOther:
    """Test alignment of another TomoStack from an existing one."""

    def test_align_other_no_shifts(self, short_stack):
        stack2 = short_stack.deepcopy()
        with pytest.raises(
            ValueError,
            match="No transformations have been applied to this stack",
        ):
            short_stack.align_other(stack2)

    def test_align_other_with_shifts(self, aligned_short_stack):
        stack2 = aligned_short_stack.deepcopy()
        stack3 = aligned_short_stack.align_other(stack2)
        assert isinstance(stack3, TomoStack)
        assert (
            aligned_short_stack.metadata.Tomography.xshift
            == stack2.metadata.Tomography.xshift  # type: ignore
        )
        assert (
            stack3.metadata.Tomography.xshift == 2 * stack2.metadata.Tomography.xshift  # type: ignore
        )


class TestStackRegister:
    """Test StackReg alignment of a TomoStack."""

    def test_stack_register_unknown_method(self, short_stack):
        bad_method = "UNKNOWN"
        with pytest.raises(
            TypeError,
            match=re.escape(
                f'Invalid registration method "{bad_method}". '
                'Must be one of ["StackReg", "PC", "COM", or "COM-CL"].',
            ),
        ):
            short_stack.stack_register(bad_method)  # type: ignore

    def test_stack_register_pc(self, short_stack):
        reg = short_stack.stack_register("PC")
        assert isinstance(reg, TomoStack)

    def test_stack_register_com(self, short_stack):
        reg = short_stack.stack_register("COM")
        assert isinstance(reg, TomoStack)

    def test_stack_register_stackreg(self, short_stack):
        reg = short_stack.stack_register("COM-CL")
        assert isinstance(reg, TomoStack)

    def test_stack_register_with_crop(self, short_stack):
        reg = short_stack.stack_register("PC", crop=True)
        assert isinstance(reg, TomoStack)
        assert np.sum(reg.data.shape) < np.sum(short_stack.data.shape)


class TestErrorPlots:
    """Test error plots for TomoStack."""

    def test_sirt_error(self, aligned_full_stack):
        rec_stack, error = aligned_full_stack.recon_error(
            128,
            iterations=2,
            constrain=True,
            cuda=False,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (
            aligned_full_stack.data.shape[1],
            aligned_full_stack.data.shape[1],
        )

    def test_sirt_error_no_slice(self, aligned_full_stack):
        rec_stack, error = aligned_full_stack.recon_error(
            None,
            iterations=2,
            constrain=True,
            cuda=False,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (
            aligned_full_stack.data.shape[1],
            aligned_full_stack.data.shape[1],
        )

    def test_recon_error_no_tilts(self, aligned_full_stack):
        stack_no_tilts = aligned_full_stack.deepcopy()
        del stack_no_tilts.tilts
        with pytest.raises(ValueError, match="Tilt angles not defined"):
            stack_no_tilts.recon_error(None, iterations=2, constrain=True, cuda=False)

    def test_sirt_error_no_cuda(self, aligned_full_stack):
        rec_stack, error = aligned_full_stack.recon_error(
            128,
            iterations=50,
            constrain=True,
            cuda=None,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (
            aligned_full_stack.data.shape[1],
            aligned_full_stack.data.shape[1],
        )

    @patch("astra.use_cuda", new=lambda: False)
    def test_recon_error_astra_detect_use_cuda_false(self, aligned_full_stack):
        rec_stack, error = aligned_full_stack.recon_error(
            128,
            iterations=2,
            constrain=True,
            cuda=None,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (
            aligned_full_stack.data.shape[1],
            aligned_full_stack.data.shape[1],
        )

    def test_sart_error(self, aligned_full_stack):
        rec_stack, error = aligned_full_stack.recon_error(
            128,
            algorithm="SART",
            iterations=2,
            constrain=True,
            cuda=False,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (
            aligned_full_stack.data.shape[1],
            aligned_full_stack.data.shape[1],
        )

    def test_sart_error_no_slice(self, aligned_full_stack):
        rec_stack, error = aligned_full_stack.recon_error(
            None,
            algorithm="SART",
            iterations=2,
            constrain=True,
            cuda=False,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (
            aligned_full_stack.data.shape[1],
            aligned_full_stack.data.shape[1],
        )

    def test_sart_error_no_cuda(self, aligned_full_stack):
        rec_stack, error = aligned_full_stack.recon_error(
            128,
            algorithm="SART",
            iterations=2,
            constrain=True,
            cuda=None,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (
            aligned_full_stack.data.shape[1],
            aligned_full_stack.data.shape[1],
        )


class TestTiltAlign:
    """Test tilt alignment of a TomoStack."""

    def test_tilt_align_com_axis_zero(self, aligned_full_stack):
        com_tilt_aligner = TiltCOMAligner(
            aligned_full_stack,
            slices=np.array([64, 100, 114]),
        )
        ali = com_tilt_aligner.align_tilt_axis()
        assert isinstance(ali, TomoStack)

    def test_tilt_align_maximage(self, aligned_short_stack):
        maximage_tilt_aligner = TiltMaxImageAligner(
            aligned_short_stack,
        )
        ali = maximage_tilt_aligner.align_tilt_axis()
        assert isinstance(ali, TomoStack)

    def test_tilt_align_unknown_method(self, aligned_short_stack):
        bad_method = "UNKNOWN"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid alignment method "{bad_method}". '
                'Must be one of ["CoM" or "MaxImage"]',
            ),
        ):
            aligned_short_stack.tilt_align(bad_method)  # pyright: ignore[reportArgumentType]


class TestTransStack:
    """Test translation of a TomoStack."""

    def test_test_trans_stack_linear(self, aligned_full_stack):
        shifted = aligned_full_stack.trans_stack(1, 1, 1, "linear")
        assert isinstance(shifted, TomoStack)

    def test_test_trans_stack_nearest(self, aligned_full_stack):
        shifted = aligned_full_stack.trans_stack(1, 1, 1, "nearest")
        assert isinstance(shifted, TomoStack)

    def test_test_trans_stack_cubic(self, aligned_full_stack):
        shifted = aligned_full_stack.trans_stack(1, 1, 1, "cubic")
        assert isinstance(shifted, TomoStack)

    def test_test_trans_stack_unknown(self, aligned_full_stack):
        bad_method = "UNKNOWN"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid interpolation method "{bad_method}". Must be one of '
                '["linear", "cubic", "nearest", or "none"].',
            ),
        ):
            aligned_full_stack.trans_stack(
                1,
                1,
                1,
                bad_method,  # pyright: ignore[reportArgumentType]
            )


class TestReconstruct:
    """Test reconstruction of a TomoStack."""

    def test_cuda_detect(self, aligned_short_stack):
        rec = aligned_short_stack.reconstruct("FBP", cuda=None)
        assert isinstance(rec, RecStack)

    @patch("astra.use_cuda", new=lambda: False)
    def test_astra_detect_use_cuda_false(self, aligned_short_stack):
        rec = aligned_short_stack.reconstruct("FBP", cuda=None)
        assert isinstance(rec, RecStack)

    def test_reconstruct_dart_no_gray_levels(self, aligned_short_stack):
        with pytest.raises(ValueError, match="gray_levels must be provided for DART"):
            aligned_short_stack.reconstruct("DART", gray_levels=None)

    def test_reconstruct_dart_gray_levels_bad_type(self, aligned_short_stack):
        with pytest.raises(
            ValueError,
            match=re.escape("Unknown type (<class 'str'>) for gray_levels"),
        ):
            aligned_short_stack.reconstruct("DART", gray_levels="bad_type")  # type: ignore

    def test_reconstruct_dart_dart_iterations_none(self, caplog, aligned_short_stack):
        sino = aligned_short_stack.inav[0:1].deepcopy()
        gray_levels = [
            0.0,
            sino.data.max() / 2,
            sino.data.max(),
        ]
        sino.reconstruct(
            "DART",
            dart_iterations=None,
            gray_levels=gray_levels,
        )
        assert "Using default number of DART iterations (5)" in caplog.text


class TestManualAlign:
    """Test manual alignment of a TomoStack."""

    def test_manual_align_positive_x(self, short_stack):
        shifted = short_stack.manual_align(2, xshift=10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_x(self, short_stack):
        shifted = short_stack.manual_align(2, xshift=-10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_positive_y(self, short_stack):
        shifted = short_stack.manual_align(2, yshift=10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_y(self, short_stack):
        shifted = short_stack.manual_align(2, yshift=-10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_y_positive_x(self, short_stack):
        shifted = short_stack.manual_align(2, yshift=-10, xshift=10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_x_positive_y(self, short_stack):
        shifted = short_stack.manual_align(2, yshift=10, xshift=-10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_y_negative_x(self, short_stack):
        shifted = short_stack.manual_align(2, yshift=-10, xshift=-10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_positive_y_positive_x(self, short_stack):
        shifted = short_stack.manual_align(2, yshift=10, xshift=10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_no_shifts(self, short_stack):
        shifted = short_stack.manual_align(2)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_with_display(self, short_stack):
        shifted = short_stack.manual_align(2, yshift=10, xshift=-10, display=True)
        assert isinstance(shifted, TomoStack)


class TestRecStack:
    """Test creating RecStacks."""

    def test_rec_stack_init(self):
        rec = RecStack(np.random.rand(10, 11, 12))
        assert isinstance(rec, RecStack)
        assert rec.metadata.get_item("Signal.signal_type") == "RecStack"
        assert rec.axes_manager.shape == (10, 12, 11)

    def test_rec_stack_init_bad_dims(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "A RecStack must have a singular (or no) navigation axis. "
                "Navigation shape was: (8, 10)",
            ),
        ):
            RecStack(np.random.rand(10, 8, 11, 12))


class TestRecStackPlotSlices:
    """Test plotting slices of a RecStack."""

    def test_plot_slices(self):
        rec = RecStack(np.zeros([10, 10, 10]))
        fig = rec.plot_slices(return_fig=True)
        assert isinstance(fig, Figure)

    @patch("matplotlib.get_backend", new=lambda: "widget")
    def test_plot_slices_widget_backend(self):
        rec = RecStack(np.zeros([10, 10, 10]))
        fig = rec.plot_slices(return_fig=True)
        assert isinstance(fig, Figure)

    @patch("matplotlib.get_backend", new=lambda: "ipympl")
    def test_plot_slices_ipympl_backend(self):
        rec = RecStack(np.zeros([10, 10, 10]))
        fig = rec.plot_slices(return_fig=True)
        assert isinstance(fig, Figure)

    @patch("matplotlib.get_backend", new=lambda: "nbagg")
    def test_plot_slices_nbagg_backend(self):
        rec = RecStack(np.zeros([10, 10, 10]))
        fig = rec.plot_slices(return_fig=True)
        assert isinstance(fig, Figure)


@pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
class TestRecStackForwardProject:
    """Test forward projection of RecStacks."""

    def test_rec_stack_forward_project_3d(self):
        rec = RecStack(np.random.rand(10, 100, 100))
        tilts = np.linspace(0, 180, 90)
        sino = rec.forward_project(tilts, cuda=False)
        assert isinstance(sino, TomoStack)
        assert isinstance(sino.tilts, TomoTilts)

    def test_rec_stack_forward_project_2d(self):
        rec = RecStack(np.random.rand(100, 100))
        tilts = np.linspace(0, 180, 90)
        sino = rec.forward_project(tilts, cuda=False)
        assert isinstance(sino, TomoStack)
        assert isinstance(sino.tilts, TomoTilts)

    def test_rec_stack_forward_project_3d_cuda(self):
        rec = RecStack(np.random.rand(10, 100, 100))
        tilts = np.linspace(0, 180, 90)
        sino = rec.forward_project(tilts, cuda=True)
        assert isinstance(sino, TomoStack)
        assert isinstance(sino.tilts, TomoTilts)

    def test_rec_stack_forward_project_2d_cuda(self):
        rec = RecStack(np.random.rand(100, 100))
        tilts = np.linspace(0, 180, 90)
        sino = rec.forward_project(tilts, cuda=True)
        assert isinstance(sino, TomoStack)
        assert isinstance(sino.tilts, TomoTilts)
