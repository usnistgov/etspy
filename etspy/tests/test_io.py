"""Tests for the IO functionality of ETSpy."""

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import h5py
import hyperspy.api as hs
import numpy as np
import pytest
from h5py import Dataset
from hyperspy.io import load as hs_load

import etspy.api as etspy
from etspy.api import etspy_path
from etspy.base import MismatchedTiltError, TomoShifts, TomoStack, TomoTilts

from . import hspy_mrc_reader_check, load_serialem_multiframe_data

if TYPE_CHECKING:
    from hyperspy.axes import UniformDataAxis as Uda

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
        ax_list = cast("list[Uda]", stack.axes_manager)

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
        ax_list = cast("list[Uda]", stack.axes_manager)

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
        del stack.original_metadata.fei_header  # pyright: ignore[reportAttributeAccessIssue]
        del stack.original_metadata.std_header  # pyright: ignore[reportAttributeAccessIssue]
        tilts = etspy.io.get_mrc_tilts(stack, filename)
        assert isinstance(tilts, np.ndarray)
        assert tilts.shape[0] == stack.data.shape[0]
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)
        assert np.all(stack.tilts.data.squeeze() == tilts)

    def test_load_mrc_with_rawtlt_str_filename(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.mrc"
        stack = etspy.io.load(str(filename))
        del stack.original_metadata.fei_header  # pyright: ignore[reportAttributeAccessIssue]
        del stack.original_metadata.std_header  # pyright: ignore[reportAttributeAccessIssue]
        tilts = etspy.io.get_mrc_tilts(stack, str(filename))
        assert isinstance(tilts, np.ndarray)
        assert tilts.shape[0] == stack.data.shape[0]
        assert isinstance(stack.tilts, TomoTilts)
        assert isinstance(stack.shifts, TomoShifts)
        assert np.all(stack.tilts.data.squeeze() == tilts)

    def test_load_mrc_with_bad_rawtlt(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.mrc"
        stack = etspy.io.load(filename)
        del stack.original_metadata.fei_header  # pyright: ignore[reportAttributeAccessIssue]
        del stack.original_metadata.std_header  # pyright: ignore[reportAttributeAccessIssue]
        stack.data = np.append(
            stack.data,
            np.zeros([1, stack.data.shape[1], stack.data.shape[2]]),
            axis=0,
        )
        with pytest.raises(
            ValueError,
            match=r"Number of tilts in .rawtlt file inconsistent with data shape",
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
        filename = etspy_path / "tests" / "test_data" / "HAADF_Aligned.hspy"
        stack_orig = hs_load(filename, file_format="HSPY")
        stack = etspy.io.load(filename)
        ax_list = cast("list[Uda]", stack.axes_manager)
        with h5py.File(filename, "r") as h5:
            h5_data = h5.get("/Experiments/__unnamed__/data")
            h5_data = cast("Dataset", h5_data)  # cast for type-checking
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
        ax_list = cast("list[Uda]", stack.axes_manager)
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

    @pytest.mark.parametrize(
        ("mintilt", "maxtilt", "step"),
        [(-30, 30, 1.2), (-35, 35, 0.7), (-75, 75, 0.3), (-72, -66, 3), (-60, 60, 2)],
    )
    def test_get_dm_tilts_float_step(self, mintilt, maxtilt, step):
        signal = hs.signals.Signal2D(np.zeros((2, 4, 4)))
        signal.original_metadata.add_dictionary(
            {
                "ImageList": {
                    "TagGroup0": {
                        "ImageTags": {
                            "Tomography": {
                                "Tomography_setup": {
                                    "Tilt_angles": {
                                        "Maximum_tilt_angle_deg": maxtilt,
                                        "Minimum_tilt_angle_deg": mintilt,
                                        "Tilt_angle_step_deg": step,
                                    },
                                },
                            },
                        },
                    },
                },
            },
        )
        tilts = etspy.io.get_dm_tilts(signal)
        ntilts = round((maxtilt - mintilt) / step) + 1
        assert tilts.shape == (ntilts,)
        np.testing.assert_allclose(tilts[0], mintilt)
        np.testing.assert_allclose(tilts[-1], maxtilt)

    def test_load_single_dm(self):
        filename = etspy_path / "tests" / "test_data" / "HAADF.dm3"
        signal = hs_load(filename)
        stack = etspy.load(filename)
        ax_list = cast("list[Uda]", stack.axes_manager)
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
        ax_list = cast("list[Uda]", stack.axes_manager)
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
            np.array(
                [
                    [[-5.00151e00], [-5.00151e00]],
                    [[-4.88000e-04], [-4.88000e-04]],
                    [[5.00054e00], [5.00054e00]],
                ],
            ),
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


def write_test_mrc(
    path,
    data,
    ext_type="",
    nint=0,
    nreal=0,
    ext_header=b"",
):
    """Write a minimal MRC file (mode 2, float32) for testing header parsing."""
    nz, ny, nx = data.shape
    words = np.zeros(256, np.int32)  # the 1024-byte header as 4-byte words
    floats = words.view(np.float32)
    words[0:3] = nx, ny, nz
    words[3] = 2  # mode 2 = float32
    words[7:10] = nx, ny, nz  # mx, my, mz
    floats[10:13] = 2.0 * nx, 2.0 * ny, 2.0 * nz  # cell size (2 A/pixel)
    floats[13:16] = 90.0  # cell angles
    words[16:19] = 1, 2, 3  # mapc, mapr, maps
    floats[19:22] = data.min(), data.max(), data.mean()
    words[23] = len(ext_header)
    header = bytearray(words.tobytes())
    header[104:108] = ext_type.encode().ljust(4, b"\0")
    header[108:112] = np.int32(20140 if ext_type == "SERI" else 0).tobytes()
    header[128:130] = np.int16(nint).tobytes()
    header[130:132] = np.int16(nreal).tobytes()
    header[208:212] = b"MAP "
    header[212:216] = bytes([0x44, 0x44, 0, 0])  # little-endian machine stamp
    path.write_bytes(bytes(header) + ext_header + data.astype(np.float32).tobytes())


@pytest.mark.skipif(hspy_mrc_broken is True, reason="Hyperspy MRC reader broken")
class TestMRCTilts:
    """Test reading tilts from MRC headers and .rawtlt files."""

    tilts = np.array([-40.0, -20.5, 0.0, 20.25, 40.0])
    data = np.random.default_rng(0).random((5, 16, 16), np.float32)

    def test_serialem_tilts_non_default_layout(self, tmp_path):
        # tilt angle + magnification only (nreal = 1 | 8), so 4 bytes per section
        sections = np.zeros((5, 2), np.int16)
        sections[:, 0] = np.round(self.tilts * 100)
        sections[:, 1] = 1150
        filename = tmp_path / "seri.mrc"
        write_test_mrc(filename, self.data, "SERI", 4, 9, sections.tobytes())
        stack = etspy.load(filename)
        np.testing.assert_allclose(stack.tilts.data.squeeze(), self.tilts)

    def test_serialem_tilts_extra_sections(self, tmp_path):
        # SerialEM may allocate extended header entries for more sections than
        # are in the file; only the first nz tilts should be used
        sections = np.zeros((8, 7), np.int16)
        sections[:5, 0] = np.round(self.tilts * 100)
        sections[5:, 0] = 9999
        filename = tmp_path / "seri.mrc"
        write_test_mrc(filename, self.data, "SERI", 14, 61, sections.tobytes())
        stack = etspy.load(filename)
        np.testing.assert_allclose(stack.tilts.data.squeeze(), self.tilts)

    def test_serialem_without_tilts_uses_rawtlt(self, tmp_path):
        # magnification only (nreal = 8): no tilt angles in the extended header
        sections = np.full((5, 1), 1150, np.int16)
        filename = tmp_path / "seri.mrc"
        write_test_mrc(filename, self.data, "SERI", 2, 8, sections.tobytes())
        np.savetxt(filename.with_suffix(".rawtlt"), self.tilts)
        stack = etspy.load(filename)
        np.testing.assert_allclose(stack.tilts.data.squeeze(), self.tilts)

    def test_imod_mrc_uses_rawtlt(self, tmp_path):
        # an MRC file without FEI or SerialEM extended headers (e.g. from IMOD)
        filename = tmp_path / "imod.mrc"
        write_test_mrc(filename, self.data)
        np.savetxt(filename.with_suffix(".rawtlt"), self.tilts)
        stack = etspy.load(filename)
        np.testing.assert_allclose(stack.tilts.data.squeeze(), self.tilts)

    def test_mrc_header_types(self, tmp_path):
        filename = tmp_path / "imod.mrc"
        write_test_mrc(filename, self.data)
        header = etspy.io.parse_mrc_header(filename)
        assert (header["nx"], header["ny"], header["nz"]) == (16, 16, 5)
        assert (header["xlen"], header["ylen"], header["zlen"]) == (32.0, 32.0, 10.0)
        assert header["amin"] == pytest.approx(self.data.min())
        assert header["amax"] == pytest.approx(self.data.max())
        assert header["amean"] == pytest.approx(self.data.mean())

    def test_serialem_test_data_header(self):
        dirname = etspy_path / "tests" / "test_data" / "SerialEM_MultiTilt_Test"
        filename = dirname / "test_000.mrc"
        header = etspy.io.parse_mrc_header(filename)
        assert header["ext_type"] == "SERI"
        assert (header["nint"], header["nreal"]) == (14, 61)


class TestParseMdoc:
    """Test parsing of SerialEM .mdoc files."""

    @staticmethod
    def _write_mdoc(path, n_titles=0):
        lines = [
            "PixelSpacing = 7.88",
            "Voltage = 300",
            "ImageFile = test.mrc",
            "ImageSize = 1024 512",
            "DataMode = 1",
            "",
            *[f"[T = title line {i} Defocus = 99]" for i in range(n_titles)],
        ]
        for i, tilt in enumerate([-30.0, 0.0, 30.0]):
            lines += [
                "",
                f"[ZValue = {i}]",
                f"TiltAngle = {tilt}",
                "Magnification = 115000",
                "SpotSize = 7",
                f"Defocus = {-1.5 - i}",
                "TargetDefocus = -9",
                "ExposureTime = 0.5",
            ]
        path.write_text("\n".join(lines) + "\n")

    def test_parse_mdoc_values(self, tmp_path):
        filename = tmp_path / "test.mdoc"
        self._write_mdoc(filename)
        meta, tilts = etspy.io.parse_mdoc(filename)
        assert meta == {
            "PixelSpacing": 7.88,
            "Voltage": 300.0,
            "ImageFile": "test.mrc",
            "ImageSize": [1024, 512],
            "DataMode": 1.0,
            "Magnification": 115000.0,
            "ExposureTime": 0.5,
            "SpotSize": 7.0,
            "Defocus": -1.5,  # from the first section, not TargetDefocus
        }
        np.testing.assert_allclose(tilts, [-30.0, 0.0, 30.0])

    def test_parse_mdoc_series(self, tmp_path):
        filename = tmp_path / "test.mdoc"
        self._write_mdoc(filename)
        _, tilt = etspy.io.parse_mdoc(filename, series=True)
        assert tilt == -30.0  # noqa: PLR2004

    def test_parse_mdoc_long_header(self, tmp_path):
        # values after the first 35 lines are still found
        filename = tmp_path / "test.mdoc"
        self._write_mdoc(filename, n_titles=40)
        meta, tilts = etspy.io.parse_mdoc(filename)
        assert meta["Magnification"] == 115000.0  # noqa: PLR2004
        assert meta["Defocus"] == -1.5  # noqa: PLR2004
        np.testing.assert_allclose(tilts, [-30.0, 0.0, 30.0])

    def test_parse_mdoc_short_file(self, tmp_path):
        filename = tmp_path / "short.mdoc"
        filename.write_text("PixelSpacing = 2\n\n[ZValue = 0]\nTiltAngle = 5\n")
        meta, tilts = etspy.io.parse_mdoc(filename)
        assert meta == {"PixelSpacing": 2.0}
        np.testing.assert_allclose(tilts, [5.0])
