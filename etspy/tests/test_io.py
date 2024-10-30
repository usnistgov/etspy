"""Tests for the IO functionality of ETSpy."""

from typing import List, cast
from unittest.mock import patch

import h5py
import hyperspy.api as hs
import numpy as np
import pytest
from h5py import Dataset
from hyperspy.axes import UniformDataAxis as Uda
from hyperspy.io import load as hs_load

import etspy.api as etspy
from etspy.api import etspy_path
from etspy.base import MismatchedTiltError, TomoShifts, TomoStack, TomoTilts

from . import hspy_mrc_reader_check, load_serialem_multiframe_data

try:
    hspy_mrc_reader_check()
except TypeError:
    hspy_mrc_broken = True
else:
    hspy_mrc_broken = False

def _type_error_mock(_, reader):  # noqa: ARG001
    msg = "Mocked type error for broken MRC reader"
    raise TypeError(msg)

class TestLoadMRC:
    """Test loading an MRC file."""

    def test_load_mrc(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.mrc"
        stack_orig = hs_load(filename)
        stack = etspy.io.load(filename)
        ax_list = cast(List[Uda], stack.axes_manager)

        assert stack.axes_manager.signal_shape == (256, 256)
        assert stack.axes_manager.navigation_shape == (77,)
        assert stack.metadata.has_item("Tomography")
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)
        assert stack.tilts.data.shape[0] == stack.data.shape[0]
        assert stack.tilts.data[0] == -76  # noqa: PLR2004
        assert isinstance(stack, TomoStack)
        assert ax_list[1].scale == stack_orig.axes_manager[1].scale
        assert ax_list[1].units == stack_orig.axes_manager[1].units

    @patch("etspy.io.hs_load", new=_type_error_mock)
    def test_load_mrc_broken_hyperspy_mrc(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.mrc"
        with pytest.raises(RuntimeError, match="Unable to read MRC with Hyperspy"):
            etspy.io.load(filename)

    def test_load_ali(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.ali"
        stack_orig = hs_load(filename)
        stack = etspy.io.load(filename)
        ax_list = cast(List[Uda], stack.axes_manager)

        assert stack.axes_manager.signal_shape == (256, 256)
        assert stack.axes_manager.navigation_shape == (77,)
        assert stack.metadata.has_item("Tomography")
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)
        assert stack.tilts.data.shape[0] == stack.data.shape[0]
        assert stack.tilts.data[0][0] == -76  # noqa: PLR2004
        assert isinstance(stack, TomoStack)
        assert ax_list[1].scale == stack_orig.axes_manager[1].scale
        assert ax_list[1].units == stack_orig.axes_manager[1].units

    def test_load_mrc_with_rawtlt(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.mrc"
        stack = etspy.io.load(filename)
        del stack.original_metadata.fei_header # pyright: ignore[reportAttributeAccessIssue]
        del stack.original_metadata.std_header # pyright: ignore[reportAttributeAccessIssue]
        tilts = etspy.io.get_mrc_tilts(stack, filename)
        assert isinstance(tilts, np.ndarray)
        assert tilts.shape[0] == stack.data.shape[0]
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)
        assert np.all(stack.tilts.data.squeeze() == tilts)

    def test_load_mrc_with_rawtlt_str_filename(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.mrc"
        stack = etspy.io.load(str(filename))
        del stack.original_metadata.fei_header # pyright: ignore[reportAttributeAccessIssue]
        del stack.original_metadata.std_header # pyright: ignore[reportAttributeAccessIssue]
        tilts = etspy.io.get_mrc_tilts(stack, str(filename))
        assert isinstance(tilts, np.ndarray)
        assert tilts.shape[0] == stack.data.shape[0]
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)
        assert np.all(stack.tilts.data.squeeze() == tilts)

    def test_load_mrc_with_bad_rawtlt(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.mrc"
        stack = etspy.io.load(filename)
        del stack.original_metadata.fei_header # pyright: ignore[reportAttributeAccessIssue]
        del stack.original_metadata.std_header # pyright: ignore[reportAttributeAccessIssue]
        stack.data = np.append(
            stack.data,
            np.zeros([1, stack.data.shape[1], stack.data.shape[2]]),
            axis=0,
        )
        with pytest.raises(
            ValueError,
            match="Number of tilts in .rawtlt file inconsistent with data shape",
        ):
            etspy.io.get_mrc_tilts(stack, filename)


class TestHspy:
    """Test loading a HyperSpy signal."""

    def test_convert_signal2d(self):
        tilts = np.arange(-10, 10, 2)
        data = hs.signals.Signal2D(np.zeros([10, 100, 100]))
        stack = TomoStack(data, tilts=tilts)
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (10,)
        assert stack.metadata.has_item("Tomography")
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)
        assert stack.tilts.data.shape[0] == stack.data.shape[0]
        assert stack.tilts.data[0][0] == -10  # noqa: PLR2004
        assert isinstance(stack, TomoStack)

    def test_load_hspy_hdf5(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF_Aligned.hdf5"
        stack_orig = hs_load(filename, reader="HSPY")
        stack = etspy.io.load(filename)
        ax_list = cast(List[Uda], stack.axes_manager)
        with h5py.File(filename, "r") as h5:
            h5_data = h5.get("/Experiments/__unnamed__/data")
            h5_data = cast(Dataset, h5_data)  # cast for type-checking
            h5_shape = h5_data.shape
        assert stack.data.shape[1:] == h5_shape[1:]
        assert stack.data.shape[0] == h5_shape[0]
        assert stack.metadata.has_item("Tomography")
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)
        assert stack.tilts.axes_manager.shape == (77, 1)
        assert stack.shifts.axes_manager.shape == (77, 2)
        assert stack.tilts.data[0][0] == -76  # noqa: PLR2004
        assert isinstance(stack, TomoStack)
        assert ax_list[1].scale == stack_orig.axes_manager[1].scale
        assert ax_list[1].units == stack_orig.axes_manager[1].units


class TestNumpy:
    """Test creating a TomoStack from NumPy arrays."""

    def test_numpy_to_stack_no_tilts(self):
        stack = TomoStack(np.random.random([50, 100, 100]), tilts=None)
        assert isinstance(stack, TomoStack)
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (50,)
        assert stack.metadata.has_item("Tomography")
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)
        assert stack.tilts.data[0][0] == 0
        assert stack.tilts.data.shape[0] == stack.data.shape[0]

    def test_numpy_to_stack_with_bad_tilts(self):
        tilts = np.arange(-50, 50, 2)
        data = np.random.random([25, 100, 100])
        with pytest.raises(MismatchedTiltError):
            TomoStack(data, tilts=tilts)

    def test_numpy_to_stack_with_tilts(self):
        tilts = np.arange(-50, 50, 2)
        stack = TomoStack(np.random.random([50, 100, 100]), tilts=tilts)
        assert isinstance(stack, etspy.TomoStack)
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (50,)
        assert stack.metadata.has_item("Tomography")
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)
        assert np.all(stack.tilts.data.squeeze() == np.arange(-50, 50, 2))
        assert stack.tilts.data.shape[0] == stack.data.shape[0]

class TestSignal:
    """Test creating stacks from HyperSpy signals."""

    def test_signal_to_stack(self):
        signal = hs.signals.Signal2D(np.random.random([50, 100, 100]))
        stack = TomoStack(signal)
        ax_list = cast(List[Uda], stack.axes_manager)
        assert ax_list[0].name == "Projections"
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (50,)
        assert stack.metadata.has_item("Tomography")
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)
        assert np.all(stack.tilts.data == 0)
        assert isinstance(stack, TomoStack)
        assert ax_list[1].scale == signal.axes_manager[1].scale
        assert ax_list[1].units == signal.axes_manager[1].units

    def test_signal_to_stack_bad_tilts(self):
        signal = hs.signals.Signal2D(np.random.random([50, 100, 100]))
        tilts = np.zeros(20)
        with pytest.raises(MismatchedTiltError):
            TomoStack(signal, tilts)


class TestDM:
    """Test loading DigitalMicrograph data."""

    def test_load_single_dm(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.dm3"
        signal = hs_load(filename)
        stack = etspy.load(filename)
        ax_list = cast(List[Uda], stack.axes_manager)
        assert ax_list[0].name == "Projections"
        assert ax_list[0].units == "degrees"
        assert stack.axes_manager.signal_shape == (64, 64)
        assert stack.axes_manager.navigation_shape == (91,)
        assert stack.metadata.has_item("Tomography")
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)
        assert stack.tilts.data[0][0] == -90  # noqa: PLR2004
        assert isinstance(stack, TomoStack)
        assert ax_list[1].scale == signal.axes_manager[1].scale
        assert ax_list[1].units == signal.axes_manager[1].units
        assert ax_list[2].scale == signal.axes_manager[2].scale
        assert ax_list[2].units == signal.axes_manager[2].units

    def test_load_dm_series(self):
        dirname = etspy_path / "tests" / "test_data" / "DM_Series_Test"
        files = list(dirname.glob("*.dm3"))
        signal = hs_load(files, stack=True)
        stack = etspy.load(files)
        ax_list = cast(List[Uda], stack.axes_manager)
        assert ax_list[0].name == "Projections"
        assert ax_list[0].units == "degrees"
        assert ax_list[0].scale == pytest.approx(3, abs=0.1)
        assert stack.axes_manager.signal_shape == (128, 128)
        assert stack.axes_manager.navigation_shape == (3,)
        assert stack.metadata.has_item("Tomography")
        assert isinstance(stack.tilts, TomoTilts)
        assert stack.tilts.data[0][0] == pytest.approx(-72, abs=0.01)
        assert stack.tilts.data[1][0] == pytest.approx(-69, abs=0.01)
        assert stack.tilts.data[2][0] == pytest.approx(-66, abs=0.01)
        assert isinstance(stack.shifts, TomoShifts)
        assert np.all(stack.shifts.data == 0)
        assert isinstance(stack, TomoStack)
        assert ax_list[1].scale == signal.axes_manager[1].scale
        assert ax_list[1].units == signal.axes_manager[1].units
        assert ax_list[2].scale == signal.axes_manager[2].scale
        assert ax_list[2].units == signal.axes_manager[2].units


@pytest.mark.skipif(hspy_mrc_broken is True, reason="Hyperspy MRC reader broken")
class TestSerialEM:
    """Test loading SerialEM data."""

    def test_load_serialem_multiframe_series(self):
        stack = load_serialem_multiframe_data()
        assert stack.axes_manager.signal_shape == (1024, 1024)
        assert stack.axes_manager.navigation_shape == (2, 3)
        assert stack.data.shape == (3, 2, 1024, 1024)

        ax_dict = stack.axes_manager.as_dictionary()
        assert ax_dict["axis-0"]["name"] == "Projections"
        assert ax_dict["axis-0"]["units"] == "degrees"
        assert ax_dict["axis-0"]["navigate"]
        assert ax_dict["axis-0"]["size"] == 3  # noqa: PLR2004
        assert ax_dict["axis-1"]["name"] == "Frames"
        assert ax_dict["axis-1"]["units"] == "images"
        assert ax_dict["axis-1"]["navigate"]
        assert ax_dict["axis-1"]["size"] == 2  # noqa: PLR2004
        assert ax_dict["axis-2"]["name"] == "y"
        assert ax_dict["axis-2"]["units"] == "nm"
        assert not ax_dict["axis-2"]["navigate"]
        assert ax_dict["axis-2"]["size"] == 1024  # noqa: PLR2004
        assert ax_dict["axis-3"]["name"] == "x"
        assert ax_dict["axis-3"]["units"] == "nm"
        assert not ax_dict["axis-3"]["navigate"]
        assert ax_dict["axis-3"]["size"] == 1024  # noqa: PLR2004

        assert stack.metadata.has_item("Tomography")
        assert isinstance(stack.tilts, TomoTilts)
        assert stack.tilts.data.shape == (3, 2, 1)
        assert stack.tilts.axes_manager.navigation_shape == (2, 3)
        assert np.allclose(
            stack.tilts.data,
            np.array([
                [[-5.00151e+00],[-5.00151e+00]],
                [[-4.88000e-04],[-4.88000e-04]],
                [[ 5.00054e+00],[ 5.00054e+00]],
            ]),
        )
        assert isinstance(stack.shifts, TomoShifts)
        assert stack.shifts.data.shape == (3, 2, 2)
        assert np.all(stack.shifts.data == 0)
        assert isinstance(stack, TomoStack)

    def test_load_serialem_multiframe_str_fnames(self):
        dirname = etspy_path / "tests" / "test_data" / "SerialEM_Multiframe_Test"
        files = list(dirname.glob("*.mrc"))
        etspy.load([str(f) for f in files])

    def test_load_serialem_multiframe_str_fnames_with_mdocs(self):
        dirname = etspy_path / "tests" / "test_data" / "SerialEM_Multiframe_Test"
        files = list(dirname.glob("*.mrc"))
        mdoc_files = [i.with_suffix(".mdoc") for i in files]
        etspy.load([str(f) for f in files], mdocs=mdoc_files)

    def test_load_serialem_multiframe_str_fnames_with_mdocs_str(self):
        dirname = etspy_path / "tests" / "test_data" / "SerialEM_Multiframe_Test"
        files = list(dirname.glob("*.mrc"))
        mdoc_files = [str(i.with_suffix(".mdoc")) for i in files]
        etspy.load([str(f) for f in files], mdocs=mdoc_files)

    def test_load_serialem_multiframe_add_mrc_to_mdoc_names(self):
        dirname = etspy_path / "tests" / "test_data" / "SerialEM_Multiframe_Test"
        files = list(dirname.glob("*.mrc"))
        mdoc_files = [str(i.with_suffix(".mrc.mdoc")) for i in files]
        etspy.load([str(f) for f in files], mdocs=mdoc_files)

    def test_load_serialem_multi_tilt_single_file(self):
        dirname = etspy_path / "tests" / "test_data" / "SerialEM_MultiTilt_Test"
        stack = etspy.load(dirname / "test_000.mrc")
        assert stack.axes_manager.signal_shape == (1024, 1024)
        assert stack.axes_manager.navigation_shape == (2,)

        ax_dict = stack.axes_manager.as_dictionary()
        assert ax_dict["axis-0"]["name"] == "Projections"
        assert ax_dict["axis-0"]["units"] == "degrees"
        assert ax_dict["axis-0"]["navigate"]
        assert ax_dict["axis-0"]["size"] == 2  # noqa: PLR2004
        assert ax_dict["axis-1"]["name"] == "y"
        assert ax_dict["axis-1"]["units"] == "nm"
        assert not ax_dict["axis-1"]["navigate"]
        assert ax_dict["axis-1"]["size"] == 1024  # noqa: PLR2004
        assert ax_dict["axis-2"]["name"] == "x"
        assert ax_dict["axis-2"]["units"] == "nm"
        assert not ax_dict["axis-2"]["navigate"]
        assert ax_dict["axis-2"]["size"] == 1024  # noqa: PLR2004

        assert stack.metadata.has_item("Tomography")
        # TODO(jat): update tilts when Andy updates new file
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)
        assert stack.tilts.data[0][0] == 0.0
        assert isinstance(stack, TomoStack)

    def test_load_serial_em_explicit(self):
        dirname = etspy_path / "tests" / "test_data" / "SerialEM_Multiframe_Test"
        mrcfile = next(dirname.glob("*.mrc"))
        mdocfile = mrcfile.with_suffix(".mdoc")
        stack = etspy.io.load_serialem(mrcfile, mdocfile)
        assert isinstance(stack, TomoStack)
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)

    def test_load_serial_em_explicit_str_filename(self):
        dirname = etspy_path / "tests" / "test_data" / "SerialEM_Multiframe_Test"
        mrcfile = next(dirname.glob("*.mrc"))
        mdocfile = mrcfile.with_suffix(".mdoc")
        stack = etspy.io.load_serialem(str(mrcfile), str(mdocfile))
        assert isinstance(stack, TomoStack)
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)



class TestUnknown:
    """Tests to ensure unknown filename types generate the expected errors."""

    def test_load_unknown_string(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.NONE"
        with pytest.raises(TypeError):
            etspy.load(filename)

    def test_load_unknown_list(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.NONE"
        files = [filename, filename]
        with pytest.raises(TypeError):
            etspy.load(files)  # pyright: ignore[reportArgumentType]

    def test_load_unknown_type(self):
        filename = np.zeros(10)
        with pytest.raises(TypeError):
            etspy.load(filename)  # pyright: ignore[reportArgumentType]


class TestMRCHeader:
    """Test reading an MRC file header."""

    def test_mrc_header_parser(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.mrc"
        header = etspy.io.parse_mrc_header(filename)
        expected = {
            "nx": 256,
            "nextra": 131072,
        }
        assert isinstance(header, dict)
        assert header["nx"] == expected["nx"]
        assert header["nextra"] == expected["nextra"]

    def test_mrc_header_parser_str_filename(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.mrc"
        header = etspy.io.parse_mrc_header(str(filename))
        expected = {
            "nx": 256,
            "nextra": 131072,
        }
        assert isinstance(header, dict)
        assert header["nx"] == expected["nx"]
        assert header["nextra"] == expected["nextra"]
