# ruff: noqa: PLR2004

"""Tests for base functions of ETSpy."""

import logging
import os
import re
from pathlib import Path
from typing import cast
from unittest.mock import patch

import h5py
import hyperspy.api as hs
import numpy as np
import pytest
from hyperspy._signals.signal2d import Signal2D
from hyperspy.axes import UniformDataAxis as Uda
from hyperspy.io import load as hs_load
from hyperspy.misc.utils import DictionaryTreeBrowser
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from etspy import datasets as ds
from etspy.base import CommonStack, RecStack, TomoShifts, TomoStack, TomoTilts

from . import load_serialem_multiframe_data

NUM_AXES_THREE = 3


def _set_tomo_metadata(s: Signal2D) -> Signal2D:
    tomo_metadata = {
        "cropped": False,
        "tiltaxis": 1,
        "xshift": 2,
        "yshift": 3,
    }
    s.metadata.add_node("Tomography")
    cast(
        DictionaryTreeBrowser,
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

    def test_slicing(self):
        s = ds.get_needle_data()
        s2 = s.inav[:5]
        assert s2.data.shape == (5, 256, 256)
        assert s2.shifts.data.shape == (5, 2)
        assert s2.tilts.data.shape == (5, 1)

    def test_plot(self):
        s = ds.get_needle_data()
        s.plot()
        f = plt.gcf()
        plt.close(f)

    def test_save(self, tmp_path):
        s = ds.get_needle_data()
        fname = tmp_path / "save_test.hdf5"
        s.save(fname, file_format="HSPY")
        with h5py.File(fname, "r") as h5:
            data = h5.get("/Experiments/__unnamed__/data")
            assert data.shape == (77, 256, 256)  # type: ignore
            tilts = h5.get("/Experiments/__unnamed__/metadata/Tomography/_sig_tilts")
            assert tilts.get("data").shape == (77, 1)  # type: ignore
            shifts = h5.get("/Experiments/__unnamed__/metadata/Tomography/_sig_shifts")
            assert shifts.get("data").shape == (77, 2)  # type: ignore

    @pytest.mark.parametrize("axis", [("XY", 77), ("XZ", 256), ("YZ", 256)])
    def test_save_movie(self, tmp_path, axis):
        s = ds.get_needle_data(aligned=True)
        f = tmp_path / f"output_movie_{axis[0]}.avi"
        s.save_movie(start=0, stop=axis[1], outfile=f, axis=axis[0])

    def test_save_movie_invalid_axis(self):
        s = ds.get_needle_data(aligned=True)
        with pytest.raises(
            ValueError,
            match=re.escape(
                'Invalid axis "not valid". Must be one of ["XY", "YZ", or "XZ"].',
            ),
        ):
            s.save_movie(start=0, stop=100, outfile="", axis="not valid")  # type: ignore

    def test_save_raw(self, tmp_path):
        os.chdir(tmp_path)
        s = ds.get_needle_data()
        fname = s.save_raw()
        hs_s = hs_load(fname)
        assert isinstance(hs_s, Signal2D)
        assert isinstance(fname, Path)
        assert fname.exists()

    def test_save_raw_with_fname(self, tmp_path):
        s = ds.get_needle_data()
        output_file = tmp_path / "test.rpl"
        fname = s.save_raw(filename=output_file)
        hs_s = hs_load(fname)
        assert isinstance(hs_s, Signal2D)
        assert isinstance(fname, Path)
        assert fname.exists()
        assert fname == output_file.parent / "test_77x256x256_float32.rpl"

    def test_save_raw_with_fname_str(self, tmp_path):
        s = ds.get_needle_data()
        output_file = tmp_path / "test.rpl"
        output_file_str = str(output_file)
        fname = s.save_raw(filename=output_file_str)
        hs_s = hs_load(fname)
        assert isinstance(hs_s, Signal2D)
        assert isinstance(fname, Path)
        assert fname.exists()
        assert fname == output_file.parent / "test_77x256x256_float32.rpl"

    def test_print_stats(self, capsys):
        ds.get_needle_data().stats()
        captured = capsys.readouterr()
        assert captured.out == "Mean: 4259.6\nStd: 11485.53\nMax: 64233.0\nMin: 0.0\n\n"


class TestTomoStack:
    """Test creation of a TomoStack."""

    def test_tomostack_create_by_signal(self):
        s = hs.signals.Signal2D(np.random.random([10, 100, 100]))
        stack = TomoStack(s)
        assert isinstance(stack, TomoStack)
        assert hasattr(stack, "tilts")
        assert hasattr(stack, "shifts")
        assert stack.metadata.get_item("Tomography.xshift") == 0
        assert stack.metadata.get_item("Tomography.yshift") == 0

    def test_tomostack_create_by_signal_with_existing_tomometa(self):
        s = hs.signals.Signal2D(np.random.random([10, 100, 100]))
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
        s = hs.signals.Signal2D(np.random.random([10, 100, 100]))
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
        s = hs.signals.Signal2D(np.random.random([10, 100, 100]))
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
        s = cast(Signal2D, hs.signals.Signal2D(np.random.random([10, 100, 100])))
        s.metadata.set_item("General.title", "original_metadata test")
        s.original_metadata.set_item("Level1.level2", "the value")
        stack = TomoStack(s)
        assert stack.metadata.get_item("General.title") == "original_metadata test"
        assert stack.original_metadata.get_item("Level1.level2") == "the value"

    def test_tomostack_create_by_signal_original_metadata_arg(self):
        s = cast(Signal2D, hs.signals.Signal2D(np.random.random([10, 100, 100])))
        s.metadata.set_item("General.title", "original_metadata test")
        s.original_metadata.set_item("Level1.level2", "the value")
        stack = TomoStack(s, original_metadata={"A1": {"B1": "C", "B2": "C2"}})
        assert stack.metadata.get_item("General.title") == "original_metadata test"
        assert not stack.original_metadata.has_item("Level1.level2")
        assert stack.original_metadata.get_item("A1.B1") == "C"
        assert stack.original_metadata.get_item("A1.B2") == "C2"

    def test_tomostack_create_by_signal_axes(self):
        s = cast(Signal2D, hs.signals.Signal2D(np.random.random([10, 100, 100])))
        s.axes_manager[0].name = "Test nav"  # type: ignore
        s.axes_manager[0].units = "Nav units"  # type: ignore
        stack = TomoStack(s)
        assert stack.axes_manager[0].name == "Test nav"  # type: ignore
        assert stack.axes_manager[0].units == "Nav units"  # type: ignore

    def test_tomostack_create_by_signal_axes_list_arg(self):
        s = cast(Signal2D, hs.signals.Signal2D(np.random.random([10, 100, 100])))
        ax_list = [
            {
                "_type": "UniformDataAxis",
                "name": "nav_from_list",
                "units": "test units",
                "navigate": True,
                "size": 10,
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
        s = cast(Signal2D, hs.signals.Signal2D(np.random.random([10, 100, 100])))
        axes = [
            {"size": 10, "name": "Axis0", "units": ""},
            {"size": 100, "name": "Axis1", "units": ""},
            {"size": 100, "name": "Axis2", "units": ""},
        ]
        stack = TomoStack(s, axes=axes)
        assert stack.axes_manager[0].name == "Projections"  # type: ignore
        assert stack.axes_manager[0].units == "degrees"  # type: ignore

    def test_tomostack_create_by_array_multiframe(self):
        n = np.random.random([20, 5, 110, 120])
        stack = TomoStack(n)
        ax0, ax1, ax2, ax3 = (cast(Uda, stack.axes_manager[i]) for i in range(4))
        assert ax0.name == "Frames"
        assert ax1.name == "Projections"
        assert ax2.name == "x"
        assert ax3.name == "y"
        assert ax0.units == "images"
        assert ax1.units == "degrees"
        assert ax2.units == "pixels"
        assert ax3.units == "pixels"
        assert ax0.size == 5
        assert ax1.size == 20
        assert ax2.size == 120
        assert ax3.size == 110

    def test_tomostack_create_by_array_too_many_dims(self):
        n = np.random.rand(2, 3, 5, 2, 1, 3)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Invalid number of navigation dimensions for a TomoStack (4). "
                "Must be either 0, 1, or 2."),
        ):
            TomoStack(n)

    def test_deepcopy(self):
        s = ds.get_needle_data()
        s2 = s.deepcopy()
        assert np.all(s.data == s2.data)
        assert np.all(s.tilts.data == s2.tilts.data)
        assert np.all(s.shifts.data == s2.shifts.data)
        assert s is not s2
        assert s.data is not s2.data
        assert s.tilts.data is not s2.tilts.data
        assert s.shifts.data is not s2.shifts.data
        np.testing.assert_equal(
            s.metadata.as_dictionary(),
            s2.metadata.as_dictionary(),
        )

    def test_copy(self):
        s = ds.get_needle_data()
        s2 = s.copy()
        assert np.all(s.data == s2.data)
        assert np.all(s.tilts.data == s2.tilts.data)
        assert np.all(s.shifts.data == s2.shifts.data)
        assert s is not s2
        assert s.data is s2.data
        assert s.tilts.data is s2.tilts.data
        assert s.shifts.data is s2.shifts.data
        np.testing.assert_equal(
            s.metadata.as_dictionary(),
            s2.metadata.as_dictionary(),
        )

    def test_remove_projections(self):
        s = ds.get_needle_data(aligned=True)
        s_new = s.remove_projections([0, 5, 10, 15, 20])
        # original shape is (77, 256, 256)
        assert s_new.data.shape == (72, 256, 256)
        assert s_new.tilts.data.shape == (72, 1)
        assert s_new.shifts.data.shape == (72, 2)
        for t in [-76, -66, -56, -46, -36]:
            assert t not in s_new.tilts.data

    def test_remove_projections_none(self):
        s = ds.get_needle_data(aligned=True)
        with pytest.raises(ValueError, match="No projections provided"):
            s.remove_projections(None)

    def test_plot_sinos(self):
        s = ds.get_needle_data(aligned=True)
        s.plot_sinos()
        f = plt.gcf()
        xlim = f.axes[0].get_xlim()
        ylim = f.axes[0].get_ylim()
        assert xlim[0] == pytest.approx(-1.68)
        assert xlim[1] == pytest.approx(858.48)
        assert ylim[0] == pytest.approx(77)
        assert ylim[1] == pytest.approx(-77)
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

    def test_tilt_setter(self):
        s = ds.get_needle_data()
        s.tilts = np.random.rand(77, 1)
        assert s.tilts.metadata.get_item("General.title") == "Image tilt values"
        assert s.tilts.axes_manager.shape == (77, 1)
        assert s.tilts.data.shape == (77, 1)

    def test_tilt_setter_1d_array(self):
        s = ds.get_needle_data()
        s.tilts = np.random.rand(77)  # should be coerced to (77, 1)

        assert s.tilts.metadata.get_item("General.title") == "Image tilt values"
        assert s.tilts.axes_manager.shape == (77, 1)
        assert s.tilts.data.shape == (77, 1)
        assert s.tilts.axes_manager[-1].name == "Tilt values"  # type: ignore
        assert s.tilts.axes_manager[-1].units == "degrees"  # type: ignore

        # check that tilt axes info matches signal
        assert s.tilts.axes_manager[0].name == s.axes_manager[0].name  # type: ignore
        assert s.tilts.axes_manager[0].units == s.axes_manager[0].units  # type: ignore
        assert s.tilts.axes_manager[0].scale == s.axes_manager[0].scale  # type: ignore
        assert s.tilts.axes_manager[0].offset == s.axes_manager[0].offset  # type: ignore

    def test_tilt_setter_bad_dims(self):
        s = ds.get_needle_data()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of tilts array must be (77, 1) to match the navigation "
                "size of the stack (was (20, 1))",
            ),
        ):
            s.tilts = np.random.rand(20, 1)

    def test_tilt_setter_tomotilt(self):
        s = ds.get_needle_data()
        n = np.random.rand(77, 1)
        tilts = TomoTilts(n)
        assert tilts.metadata.get_item("General.title") == ""
        s.tilts = tilts  # title should be set since it was empty
        assert s.tilts.metadata.get_item("General.title") == "Image tilt values"

    def test_tilt_setter_tomotilt_bad_dims(self):
        s = ds.get_needle_data()
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
            s.tilts = tilts

    def test_shift_setter(self):
        s = ds.get_needle_data()
        s.shifts = np.random.rand(77, 2)
        assert s.shifts.metadata.get_item("General.title") == "Image shift values"
        assert s.shifts.axes_manager.shape == (77, 2)
        assert s.shifts.data.shape == (77, 2)
        assert s.shifts.axes_manager[-1].name == "Shift values (x/y)"  # type: ignore
        assert s.shifts.axes_manager[-1].units == "pixels"  # type: ignore

        # check that tilt axes info matches signal
        assert s.shifts.axes_manager[0].name == s.axes_manager[0].name  # type: ignore
        assert s.shifts.axes_manager[0].units == s.axes_manager[0].units  # type: ignore
        assert s.shifts.axes_manager[0].scale == s.axes_manager[0].scale  # type: ignore
        assert s.shifts.axes_manager[0].offset == s.axes_manager[0].offset  # type: ignore

    def test_shift_setter_bad_dims(self):
        s = ds.get_needle_data()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of shifts array must be (77, 2) to match the navigation "
                "size of the stack (was (77,))",
            ),
        ):
            s.shifts = np.random.rand(77)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of shifts array must be (77, 2) to match the navigation "
                "size of the stack (was (20, 1, 5))",
            ),
        ):
            s.shifts = np.random.rand(20, 1, 5)

    def test_shift_setter_tomoshift(self):
        s = ds.get_needle_data()
        n = np.random.rand(77, 2)
        shifts = TomoShifts(n)
        assert shifts.metadata.get_item("General.title") == ""
        s.shifts = shifts  # title should be set since it was empty
        assert s.shifts.metadata.get_item("General.title") == "Image shift values"

    def test_shift_setter_tomoshift_bad_dims(self):
        s = ds.get_needle_data()
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
            s.shifts = shifts

    def test_multiframe_tilt_setter(self):
        s = load_serialem_multiframe_data()
        assert s.axes_manager.shape == (2, 3, 1024, 1024)
        assert s.data.shape == (3, 2, 1024, 1024)
        assert s.tilts.axes_manager.shape == (2, 3, 1)
        assert s.tilts.data.shape == (3, 2, 1)

        n = np.random.rand(3, 2, 1)
        s.tilts = n
        assert s.tilts.axes_manager.shape == (2, 3, 1)
        assert s.tilts.data.shape == (3, 2, 1)
        assert s.tilts.metadata.get_item("General.title") == "Image tilt values"

    def test_multiframe_tilt_setter_2d(self):
        s = load_serialem_multiframe_data()
        assert s.axes_manager.shape == (2, 3, 1024, 1024)
        assert s.data.shape == (3, 2, 1024, 1024)
        assert s.tilts.axes_manager.shape == (2, 3, 1)
        assert s.tilts.data.shape == (3, 2, 1)

        n = np.random.rand(3, 2)
        s.tilts = n
        assert s.tilts.axes_manager.shape == (2, 3, 1)
        assert s.tilts.data.shape == (3, 2, 1)
        assert s.tilts.metadata.get_item("General.title") == "Image tilt values"

    def test_multiframe_tilt_setter_bad_dims(self):
        s = load_serialem_multiframe_data()
        assert s.axes_manager.shape == (2, 3, 1024, 1024)
        assert s.data.shape == (3, 2, 1024, 1024)
        assert s.tilts.axes_manager.shape == (2, 3, 1)
        assert s.tilts.data.shape == (3, 2, 1)

        n = np.random.rand(2, 3, 1)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of tilts array must be (3, 2, 1) to match the navigation size "
                "of the stack (was (2, 3, 1))",
            ),
        ):
            s.tilts = n

        n = np.random.rand(3, 2, 10)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of tilts array must be (3, 2, 1) to match the navigation size "
                "of the stack (was (3, 2, 10))",
            ),
        ):
            s.tilts = n

    def test_multiframe_shift__setter(self):
        s = load_serialem_multiframe_data()
        assert s.axes_manager.shape == (2, 3, 1024, 1024)
        assert s.data.shape == (3, 2, 1024, 1024)
        assert s.shifts.axes_manager.shape == (2, 3, 2)
        assert s.shifts.data.shape == (3, 2, 2)

        n = np.random.rand(3, 2, 2)
        s.shifts = n
        assert s.shifts.axes_manager.shape == (2, 3, 2)
        assert s.shifts.data.shape == (3, 2, 2)
        assert s.shifts.metadata.get_item("General.title") == "Image shift values"

    def test_multiframe_shift__setter_bad_dims(self):
        s = load_serialem_multiframe_data()
        assert s.axes_manager.shape == (2, 3, 1024, 1024)
        assert s.data.shape == (3, 2, 1024, 1024)
        assert s.shifts.axes_manager.shape == (2, 3, 2)
        assert s.shifts.data.shape == (3, 2, 2)

        n = np.random.rand(2, 3, 2)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of shifts array must be (3, 2, 2) to match the navigation size "
                "of the stack (was (2, 3, 2))",
            ),
        ):
            s.shifts = n

        n = np.random.rand(3, 2, 10)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Shape of shifts array must be (3, 2, 2) to match the navigation size "
                "of the stack (was (3, 2, 10))",
            ),
        ):
            s.shifts = n

    def test_property_deleters(self):
        s = ds.get_needle_data()

        # tilts
        assert np.all(s.tilts.data.squeeze() == np.arange(-76, 78, 2))
        del s.tilts
        assert np.all(s.tilts.data.squeeze() == np.zeros((77, 1)))

        # shifts
        s.shifts = np.random.rand(77, 2) + 2  # offset to ensure non-zero
        assert np.all(s.shifts.data != np.zeros((77, 2)))
        del s.shifts
        assert np.all(s.shifts.data == np.zeros((77, 2)))


class TestSlicers:
    """Test inav/isig slicers."""

    def test_tilt_nav_slicer(self):
        stack = ds.get_needle_data()
        t = stack.tilts.inav[:5]
        assert isinstance(t, TomoTilts)
        assert t.data.shape == (5, 1)

    def test_tilt_sig_slicer(self, caplog):
        stack = ds.get_needle_data()
        with caplog.at_level(logging.WARNING):
            t = stack.tilts.isig[:5]
            assert "TomoTilts does not support 'isig' slicing" in caplog.text
        assert isinstance(t, TomoTilts)
        assert t.data.shape == (77, 1)

    def test_shift_nav_slicer(self):
        stack = ds.get_needle_data()
        t = stack.shifts.inav[:5]
        assert isinstance(t, TomoShifts)
        assert t.data.shape == (5, 2)

    def test_shift_sig_slicer(self, caplog):
        stack = ds.get_needle_data()
        with caplog.at_level(logging.WARNING):
            t = stack.shifts.isig[:5]
            # warning should be triggered when slicing the TomoTilts directly
            assert "TomoShifts does not support 'isig' slicing" in caplog.text
        assert isinstance(t, TomoShifts)
        assert t.data.shape == (77, 2)

    def test_tomostack_nav_slicer(self):
        stack = ds.get_needle_data()
        t = stack.inav[:5]
        assert t.data.shape == (5, 256, 256)
        assert t.tilts.data.shape == (5, 1)
        assert t.shifts.data.shape == (5, 2)
        assert isinstance(t, TomoStack)
        assert isinstance(t.tilts, TomoTilts)
        assert isinstance(t.shifts, TomoShifts)

    def test_tomostack_sig_slicer(self, caplog):
        stack = ds.get_needle_data()
        with caplog.at_level(logging.WARNING):
            t = stack.isig[:5, :10]
            # warning should not be triggered when slicing the TomoStack
            assert "TomoShifts does not support 'isig' slicing" not in caplog.text
        assert t.data.shape == (77, 10, 5)
        assert t.tilts.data.shape == (77, 1)
        assert t.shifts.data.shape == (77, 2)
        assert isinstance(t, TomoStack)
        assert isinstance(t.tilts, TomoTilts)
        assert isinstance(t.shifts, TomoShifts)

    def test_two_d_tomo_stack_slicing(self):
        """Test handling of multi-frame TomoStack with 2 navigation dimensions."""
        s = load_serialem_multiframe_data()
        assert s.axes_manager.shape == (2, 3, 1024, 1024)
        assert s.data.shape == (3, 2, 1024, 1024)
        ax_0, ax_1, ax_2, ax_3 = cast(list[Uda], [s.axes_manager[i] for i in range(4)])
        assert ax_0.name == "Frames"
        assert ax_0.units == "images"
        assert ax_1.name == "Projections"
        assert ax_1.units == "degrees"
        assert ax_2.name == "x"
        assert ax_2.units == "nm"
        assert ax_3.name == "y"
        assert ax_3.units == "nm"

        # test inav and isig together with ranges
        t = s.inav[:1, :2].isig[:20, :120]
        assert t.axes_manager.shape == (1, 2, 20, 120)
        assert t.data.shape == (2, 1, 120, 20)
        assert t.tilts.axes_manager.shape == (1, 2, 1)
        assert t.shifts.axes_manager.shape == (1, 2, 2)

        # test extracting single projection
        t2 = s.isig[:20, :120].inav[:, 2]
        assert t2.axes_manager.shape == (2, 20, 120)
        assert t2.data.shape == (2, 120, 20)
        assert t2.axes_manager[0].name == "Frames"  # type: ignore
        assert t2.tilts.axes_manager.navigation_shape == (2,)
        assert t2.shifts.axes_manager.navigation_shape == (2,)

        # test extracting single frame
        t3 = s.isig[:20, :120].inav[1, :]
        assert t3.axes_manager.shape == (3, 20, 120)
        assert t3.data.shape == (3, 120, 20)
        assert t3.axes_manager[0].name == "Projections"  # type: ignore
        assert t3.tilts.axes_manager.navigation_shape == (3,)
        assert t3.shifts.axes_manager.navigation_shape == (3,)

        # test extracting single frame and projection
        t4 = s.isig[:20, :120].inav[1, 1]
        assert t4.axes_manager.shape == (20, 120)
        assert t4.data.shape == (120, 20)
        assert t4.axes_manager[0].name == "x"  # type: ignore
        assert t4.axes_manager[1].name == "y"  # type: ignore
        assert t4.tilts.axes_manager.navigation_shape == ()
        assert t4.shifts.axes_manager.navigation_shape == ()
        assert t4.tilts.data[0] == pytest.approx(-0.000488)

    def test_single_pixel_nav_slicer(self):
        stack = ds.get_needle_data()
        t = stack.inav[30]
        ax = t.axes_manager
        assert t.data.shape == (256, 256)
        assert ax.navigation_shape == ()
        assert ax.shape == (256, 256)

    def test_single_pixel_sig_slicer_x(self, caplog): # type: ignore
        stack = ds.get_needle_data()
        with caplog.at_level(logging.WARNING):
            # warning should be triggered and shape should be (77, 1, 256)
            t = stack.isig[5, :]
            assert (
                "Slicing a TomoStack signal axis with a single pixel "
                'is not supported. Returning a single pixel on the "x" '
                "axis instead"
            ) in caplog.text
        assert t.data.shape == (77, 256, 1)
        assert t.axes_manager.shape == (77, 1, 256)
        ax_0, ax_1, ax_2 = cast(list[Uda], [t.axes_manager[i] for i in range(3)])
        assert ax_0.name == "Projections"
        assert ax_1.name == "x"
        assert ax_2.name == "y"
        assert ax_0.size == 77
        assert ax_1.size == 1
        assert ax_2.size == 256
        assert ax_0.units == "degrees"
        assert ax_1.units == "nm"
        assert ax_2.units == "nm"
        assert ax_1.offset == pytest.approx(ax_1.scale * 5)
        assert ax_2.offset == 0

    def test_single_pixel_sig_slicer_y(self, caplog):
        stack = ds.get_needle_data()
        with caplog.at_level(logging.WARNING):
            # warning should be triggered and shape should be (77, 256, 1)
            t = stack.isig[:, 128]
            assert (
                "Slicing a TomoStack signal axis with a single pixel "
                'is not supported. Returning a single pixel on the "y" '
                "axis instead"
            ) in caplog.text
        assert t.data.shape == (77, 1, 256)
        assert t.axes_manager.shape == (77, 256, 1)
        ax_0, ax_1, ax_2 = cast(list[Uda], [t.axes_manager[i] for i in range(3)])
        assert ax_0.name == "Projections"
        assert ax_1.name == "x"
        assert ax_2.name == "y"
        assert ax_0.size == 77
        assert ax_1.size == 256
        assert ax_2.size == 1
        assert ax_0.units == "degrees"
        assert ax_1.units == "nm"
        assert ax_2.units == "nm"
        assert ax_1.offset == 0
        assert ax_2.offset == pytest.approx(ax_2.scale * 128)

    def test_single_pixel_sig_slicer_float(self, caplog):
        stack = ds.get_needle_data()
        with caplog.at_level(logging.WARNING):
            # warning should be triggered and shape should be (77, 256, 1)
            t = stack.isig[:, 30.4]
            assert (
                "Slicing a TomoStack signal axis with a single pixel "
                'is not supported. Returning a single pixel on the "y" '
                "axis instead"
            ) in caplog.text
        assert t.data.shape == (77, 1, 256)
        assert t.axes_manager.shape == (77, 256, 1)
        ax_0, ax_1, ax_2 = cast(list[Uda], [t.axes_manager[i] for i in range(3)])
        assert ax_0.name == "Projections"
        assert ax_1.name == "x"
        assert ax_2.name == "y"
        assert ax_0.size == 77
        assert ax_1.size == 256
        assert ax_2.size == 1
        assert ax_0.units == "degrees"
        assert ax_1.units == "nm"
        assert ax_2.units == "nm"
        assert ax_1.offset == 0
        assert ax_2.offset == pytest.approx(30.24)

    # def test_single_pixel_nav_slicer(self):
    #     pass

    def test_recstack_nav_slicer(self):
        stack = RecStack(ds.get_needle_data())
        t = stack.inav[:5]
        assert t.data.shape == (5, 256, 256)
        assert isinstance(t, RecStack)

    def test_recstack_sig_slicer(self, caplog):
        stack = RecStack(ds.get_needle_data())
        with caplog.at_level(logging.WARNING):
            t = stack.isig[:5, :10]
            # warning should not be triggered when slicing the TomoStack
            assert "TomoShifts does not support 'isig' slicing" not in caplog.text
        assert t.data.shape == (77, 10, 5)
        assert isinstance(t, RecStack)

class TestExtractSinogram:
    """Test extract_sinogram method."""

    def test_extract_sinogram(self):
        stack = ds.get_catalyst_data()
        sino = stack.extract_sinogram(300)
        ax_0, ax_1 = cast(list[Uda], [sino.axes_manager[i] for i in range(2)])
        assert sino.axes_manager.shape == (600, 90)
        assert sino.metadata.get_item("Signal.signal_type") == ""
        assert ax_0.name == "y"
        assert ax_1.name == "Projections"
        assert sino.metadata.get_item("General.title") == "Sinogram at column 300"

    def test_extract_sinogram_float(self):
        stack = ds.get_catalyst_data()
        sino = stack.extract_sinogram(106.32)
        ax_0, ax_1 = cast(list[Uda], [sino.axes_manager[i] for i in range(2)])
        assert sino.axes_manager.shape == (600, 90)
        assert sino.metadata.get_item("Signal.signal_type") == ""
        assert ax_0.name == "y"
        assert ax_1.name == "Projections"
        assert sino.metadata.get_item("General.title") == "Sinogram at x = 106.32 nm"

    def test_extract_sinogram_bad_argument_type(self):
        stack = ds.get_catalyst_data()
        with pytest.raises(
            TypeError,
            match=re.escape(
                '"column" argument must be either a float or an integer '
                "(was <class 'str'>)",
            ),
        ):
            stack.extract_sinogram("bad_val") # type: ignore


    def test_extract_sinogram_exception_handling(self):
        # test that on exception, logger is still enabled
        stack = ds.get_catalyst_data()
        assert logging.getLogger("etspy.base").disabled is False
        with pytest.raises(IndexError):
            # using too large of a value should trigger an error
            stack.extract_sinogram(column=10000)
        assert logging.getLogger("etspy.base").disabled is False

class TestFiltering:
    """Test filtering of TomoStack data."""

    def test_correlation_check(self):
        stack = ds.get_needle_data()
        fig = stack.test_correlation()
        assert isinstance(fig, Figure)
        assert len(fig.axes) == NUM_AXES_THREE

    def test_image_filter_median(self):
        stack = ds.get_needle_data()
        filt = stack.inav[0:10].filter(method="median")
        assert (
            filt.axes_manager.navigation_shape
            == stack.inav[0:10].axes_manager.navigation_shape
        )
        assert (
            filt.axes_manager.signal_shape == stack.inav[0:10].axes_manager.signal_shape
        )

    def test_image_filter_sobel(self):
        stack = ds.get_needle_data()
        filt = stack.inav[0:10].filter(method="sobel")
        assert (
            filt.axes_manager.navigation_shape
            == stack.inav[0:10].axes_manager.navigation_shape
        )
        assert (
            filt.axes_manager.signal_shape == stack.inav[0:10].axes_manager.signal_shape
        )

    def test_image_filter_both(self):
        stack = ds.get_needle_data()
        filt = stack.inav[0:10].filter(method="both")
        assert (
            filt.axes_manager.navigation_shape
            == stack.inav[0:10].axes_manager.navigation_shape
        )
        assert (
            filt.axes_manager.signal_shape == stack.inav[0:10].axes_manager.signal_shape
        )

    def test_image_filter_bpf(self):
        stack = ds.get_needle_data()
        filt = stack.inav[0:10].filter(method="bpf")
        assert (
            filt.axes_manager.navigation_shape
            == stack.inav[0:10].axes_manager.navigation_shape
        )
        assert (
            filt.axes_manager.signal_shape == stack.inav[0:10].axes_manager.signal_shape
        )

    def test_image_filter_wrong_name(self):
        stack = ds.get_needle_data()
        bad_name = "WRONG"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid filter method "{bad_name}". '
                'Must be one of ["median", "bpf", "both", or "sobel"]',
            ),
        ):
            stack.inav[0:10].filter(method="WRONG")  # type: ignore


class TestOperations:
    """Test various operations of a TomoStack."""

    def test_stack_normalize(self):
        stack = ds.get_needle_data()
        norm = cast(TomoStack, stack.normalize())
        assert norm.axes_manager.navigation_shape == stack.axes_manager.navigation_shape
        assert norm.axes_manager.signal_shape == stack.axes_manager.signal_shape
        assert norm.data.min() == 0.0

    def test_stack_invert(self):
        im = np.zeros([10, 100, 100])
        im[:, 40:60, 40:60] = 10
        stack = TomoStack(im)
        invert = cast(TomoStack, stack.invert())
        hist, bins = np.histogram(stack.data)
        hist_inv, bins_inv = np.histogram(invert.data)
        assert hist[0] > hist_inv[0]

    def test_stack_stats(self, capsys):
        stack = ds.get_needle_data()
        stack.stats()

        # capture output stream to test print statements
        captured = capsys.readouterr()
        out = captured.out.split("\n")

        assert out[0] == f"Mean: {stack.data.mean():.1f}"
        assert out[1] == f"Std: {stack.data.std():.2f}"
        assert out[2] == f"Max: {stack.data.max():.1f}"
        assert out[3] == f"Min: {stack.data.min():.1f}"

    def test_set_tilts(self):
        stack = ds.get_needle_data()
        start, increment = -50, 5
        stack.set_tilts(start, increment)
        ax = cast(Uda, stack.axes_manager[0])
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

    def test_set_tilts_no_metadata(self):
        stack = ds.get_needle_data()
        del stack.metadata.Tomography  # pyright: ignore[reportAttributeAccessIssue]
        start, increment = -50, 5
        stack.set_tilts(start, increment)
        ax = cast(Uda, stack.axes_manager[0])
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

    def test_test_align_no_slices(self):
        stack = ds.get_needle_data(aligned=True)
        fig = stack.test_align()
        assert len(fig.axes) == NUM_AXES_THREE

    def test_test_align_with_angle(self):
        stack = ds.get_needle_data(aligned=True)
        fig = stack.test_align(tilt_rotation=3.0)
        assert len(fig.axes) == NUM_AXES_THREE

    def test_test_align_with_xshift(self):
        stack = ds.get_needle_data(aligned=True)
        fig = stack.test_align(tilt_shift=3.0)
        assert len(fig.axes) == NUM_AXES_THREE

    def test_test_align_with_thickness(self):
        stack = ds.get_needle_data(aligned=True)
        fig = stack.test_align(thickness=200)
        assert len(fig.axes) == NUM_AXES_THREE

    def test_test_align_no_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        fig = stack.test_align(thickness=200, cuda=False)
        assert len(fig.axes) == NUM_AXES_THREE

    @patch("astra.use_cuda", new=lambda: False)
    def test_test_align_cuda_none_mock_false(self):
        stack = ds.get_needle_data(aligned=True)
        fig = stack.test_align(thickness=200, cuda=None)
        assert len(fig.axes) == NUM_AXES_THREE

    @patch("matplotlib.get_backend", new=lambda: "widget")
    def test_test_align_mock_mpl_backend_none(self):
        stack = ds.get_needle_data(aligned=True)
        fig = stack.test_align(thickness=200)
        assert len(fig.axes) == NUM_AXES_THREE
        assert np.all(fig.get_size_inches() == np.array([12, 4]))

    @patch("matplotlib.get_backend", new=lambda: "ipympl")
    def test_test_align_mock_mpl_backend_ipympl(self):
        stack = ds.get_needle_data(aligned=True)
        fig = stack.test_align(thickness=200)
        assert len(fig.axes) == NUM_AXES_THREE
        assert np.all(fig.get_size_inches() == np.array([7, 3]))

    @patch("matplotlib.get_backend", new=lambda: "nbagg")
    def test_test_align_mock_mpl_backend_nbagg(self):
        stack = ds.get_needle_data(aligned=True)
        fig = stack.test_align(thickness=200)
        assert len(fig.axes) == NUM_AXES_THREE
        assert np.all(fig.get_size_inches() == np.array([8, 4]))


class TestAlignOther:
    """Test alignment of another TomoStack from an existing one."""

    def test_align_other_no_shifts(self):
        stack = ds.get_needle_data(aligned=False)
        stack2 = stack.deepcopy()
        with pytest.raises(
            ValueError,
            match="No transformations have been applied to this stack",
        ):
            stack.align_other(stack2)

    def test_align_other_with_shifts(self):
        stack = ds.get_needle_data(aligned=True)
        stack2 = stack.deepcopy()
        stack3 = stack.align_other(stack2)
        assert isinstance(stack3, TomoStack)
        assert (
            stack.metadata.Tomography.xshift == stack2.metadata.Tomography.xshift  # type: ignore
        )
        assert (
            stack3.metadata.Tomography.xshift == 2 * stack2.metadata.Tomography.xshift  # type: ignore
        )


class TestStackRegister:
    """Test StackReg alignment of a TomoStack."""

    def test_stack_register_unknown_method(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        bad_method = "UNKNOWN"
        with pytest.raises(
            TypeError,
            match=re.escape(
                f'Invalid registration method "{bad_method}". '
                'Must be one of ["StackReg", "PC", "COM", or "COM-CL"].',
            ),
        ):
            stack.stack_register(bad_method)  # type: ignore

    def test_stack_register_pc(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        reg = stack.stack_register("PC")
        assert isinstance(reg, TomoStack)

    def test_stack_register_com(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        reg = stack.stack_register("COM")
        assert isinstance(reg, TomoStack)

    def test_stack_register_stackreg(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        reg = stack.stack_register("COM-CL")
        assert isinstance(reg, TomoStack)

    def test_stack_register_with_crop(self):
        stack = ds.get_needle_data(aligned=False).inav[0:5]
        reg = stack.stack_register("PC", crop=True)
        assert isinstance(reg, TomoStack)
        assert np.sum(reg.data.shape) < np.sum(stack.data.shape)


class TestErrorPlots:
    """Test error plots for TomoStack."""

    def test_sirt_error(self):
        stack = ds.get_needle_data(aligned=True)
        rec_stack, error = stack.recon_error(
            128,
            iterations=2,
            constrain=True,
            cuda=False,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (stack.data.shape[1], stack.data.shape[1])

    def test_sirt_error_no_slice(self):
        stack = ds.get_needle_data(aligned=True)
        rec_stack, error = stack.recon_error(
            None,
            iterations=2,
            constrain=True,
            cuda=False,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (stack.data.shape[1], stack.data.shape[1])

    def test_recon_error_no_tilts(self):
        stack = ds.get_needle_data(aligned=True)
        del stack.tilts
        with pytest.raises(ValueError, match="Tilt angles not defined"):
            stack.recon_error(None, iterations=2, constrain=True, cuda=False)

    def test_sirt_error_no_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        rec_stack, error = stack.recon_error(
            128,
            iterations=50,
            constrain=True,
            cuda=None,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (stack.data.shape[1], stack.data.shape[1])

    @patch("astra.use_cuda", new=lambda: False)
    def test_recon_error_astra_detect_use_cuda_false(self):
        stack = ds.get_needle_data(aligned=True)
        rec_stack, error = stack.recon_error(
            128,
            iterations=50,
            constrain=True,
            cuda=None,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (stack.data.shape[1], stack.data.shape[1])

    def test_sart_error(self):
        stack = ds.get_needle_data(aligned=True)
        rec_stack, error = stack.recon_error(
            128,
            algorithm="SART",
            iterations=2,
            constrain=True,
            cuda=False,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (stack.data.shape[1], stack.data.shape[1])

    def test_sart_error_no_slice(self):
        stack = ds.get_needle_data(aligned=True)
        rec_stack, error = stack.recon_error(
            None,
            algorithm="SART",
            iterations=2,
            constrain=True,
            cuda=False,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (stack.data.shape[1], stack.data.shape[1])

    def test_sart_error_no_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        rec_stack, error = stack.recon_error(
            128,
            algorithm="SART",
            iterations=50,
            constrain=True,
            cuda=None,
        )
        assert error.data.shape[0] == rec_stack.data.shape[0]
        assert rec_stack.data.shape[1:] == (stack.data.shape[1], stack.data.shape[1])


class TestTiltAlign:
    """Test tilt alignment of a TomoStack."""

    def test_tilt_align_com_axis_zero(self):
        stack = ds.get_needle_data(aligned=True)
        ali = stack.tilt_align("CoM", slices=np.array([64, 100, 114]))
        assert isinstance(ali, TomoStack)

    def test_tilt_align_maximage(self):
        stack = ds.get_needle_data(aligned=True)
        stack = stack.inav[0:10]
        ali = stack.tilt_align("MaxImage")
        assert isinstance(ali, TomoStack)

    def test_tilt_align_unknown_method(self):
        stack = ds.get_needle_data(aligned=True)
        bad_method = "UNKNOWN"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid alignment method "{bad_method}". '
                'Must be one of ["CoM" or "MaxImage"]',
            ),
        ):
            stack.tilt_align(bad_method)  # pyright: ignore[reportArgumentType]


class TestTransStack:
    """Test translation of a TomoStack."""

    def test_test_trans_stack_linear(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.trans_stack(1, 1, 1, "linear")
        assert isinstance(shifted, TomoStack)

    def test_test_trans_stack_nearest(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.trans_stack(1, 1, 1, "nearest")
        assert isinstance(shifted, TomoStack)

    def test_test_trans_stack_cubic(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.trans_stack(1, 1, 1, "cubic")
        assert isinstance(shifted, TomoStack)

    def test_test_trans_stack_unknown(self):
        stack = ds.get_needle_data(aligned=True)
        bad_method = "UNKNOWN"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid interpolation method "{bad_method}". Must be one of '
                '["linear", "cubic", "nearest", or "none"].',
            ),
        ):
            stack.trans_stack(
                1,
                1,
                1,
                bad_method,  # pyright: ignore[reportArgumentType]
            )


class TestReconstruct:
    """Test reconstruction of a TomoStack."""

    def test_cuda_detect(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[:, 120:121].deepcopy()
        rec = slices.reconstruct("FBP", cuda=None)
        assert isinstance(rec, RecStack)

    @patch("astra.use_cuda", new=lambda: False)
    def test_astra_detect_use_cuda_false(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[:, 120:121].deepcopy()
        rec = slices.reconstruct("FBP", cuda=None)
        assert isinstance(rec, RecStack)

    def test_reconstruct_dart_no_gray_levels(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[:, 120:121].deepcopy()
        with pytest.raises(ValueError, match="gray_levels must be provided for DART"):
            slices.reconstruct("DART", gray_levels=None)

    def test_reconstruct_dart_gray_levels_bad_type(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[:, 120:121].deepcopy()
        with pytest.raises(
            ValueError,
            match=re.escape("Unknown type (<class 'str'>) for gray_levels"),
        ):
            slices.reconstruct("DART", gray_levels="bad_type")  # type: ignore

    def test_reconstruct_dart_dart_iterations_none(self, caplog):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[:, 120:121].deepcopy()
        gray_levels = [0.0, slices.data.max() / 2, slices.data.max()]
        slices.reconstruct("DART", dart_iterations=None, gray_levels=gray_levels)
        assert "Using default number of DART iterations (5)" in caplog.text


class TestManualAlign:
    """Test manual alignment of a TomoStack."""

    def test_manual_align_positive_x(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, xshift=10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_x(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, xshift=-10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_positive_y(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, yshift=10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_y(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, yshift=-10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_y_positive_x(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, yshift=-10, xshift=10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_x_positive_y(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, yshift=10, xshift=-10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_negative_y_negative_x(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, yshift=-10, xshift=-10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_positive_y_positive_x(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128, yshift=10, xshift=10)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_no_shifts(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(128)
        assert isinstance(shifted, TomoStack)

    def test_manual_align_with_display(self):
        stack = ds.get_needle_data(aligned=True)
        shifted = stack.manual_align(64, display=True)
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
        fig = rec.plot_slices()
        assert isinstance(fig, Figure)

    @patch("matplotlib.get_backend", new=lambda: "widget")
    def test_plot_slices_widget_backend(self):
        rec = RecStack(np.zeros([10, 10, 10]))
        fig = rec.plot_slices()
        assert isinstance(fig, Figure)
        assert np.all(fig.get_size_inches() == np.array([12, 4]))

    @patch("matplotlib.get_backend", new=lambda: "ipympl")
    def test_plot_slices_ipympl_backend(self):
        rec = RecStack(np.zeros([10, 10, 10]))
        fig = rec.plot_slices()
        assert isinstance(fig, Figure)
        assert np.all(fig.get_size_inches() == np.array([7, 3]))

    @patch("matplotlib.get_backend", new=lambda: "nbagg")
    def test_plot_slices_nbagg_backend(self):
        rec = RecStack(np.zeros([10, 10, 10]))
        fig = rec.plot_slices()
        assert isinstance(fig, Figure)
        assert np.all(fig.get_size_inches() == np.array([8, 4]))
