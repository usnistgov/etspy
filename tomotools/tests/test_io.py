import etspy.api as etspy
from etspy.base import TomoStack
import hyperspy.api as hs
import os
import numpy as np
import pytest
import h5py
import glob
from hyperspy.signals import Signal2D

etspy_path = os.path.dirname(etspy.__file__)


def hspy_mrc_reader_check():
    dirname = os.path.join(etspy_path, "tests",
                           "test_data", "SerialEM_Multiframe_Test")
    files = glob.glob(dirname + "/*.mrc")
    file = files[0]
    s = hs.load(file)
    return s


try:
    hspy_mrc_reader_check()
except TypeError:
    hspy_mrc_broken = True
else:
    hspy_mrc_broken = False


class TestLoadMRC:
    def test_load_mrc(self):
        filename = os.path.join(etspy_path, "tests",
                                "test_data", "HAADF.mrc")
        stack_orig = hs.load(filename)
        stack = etspy.io.load(filename)

        assert stack.axes_manager.signal_shape == (256, 256)
        assert stack.axes_manager.navigation_shape == (77,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts.shape[0] == stack.data.shape[0]
        assert stack.metadata.Tomography.tilts[0] == -76
        assert type(stack) is TomoStack
        assert stack.axes_manager[1].scale == stack_orig.axes_manager[1].scale
        assert stack.axes_manager[1].units == stack_orig.axes_manager[1].units

    def test_load_ali(self):
        filename = os.path.join(etspy_path, "tests",
                                "test_data", "HAADF.ali")
        stack_orig = hs.load(filename)
        stack = etspy.io.load(filename)

        assert stack.axes_manager.signal_shape == (256, 256)
        assert stack.axes_manager.navigation_shape == (77,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts.shape[0] == stack.data.shape[0]
        assert stack.metadata.Tomography.tilts[0] == -76
        assert type(stack) is TomoStack
        assert stack.axes_manager[1].scale == stack_orig.axes_manager[1].scale
        assert stack.axes_manager[1].units == stack_orig.axes_manager[1].units

    def test_load_mrc_with_rawtlt(self):
        filename = os.path.join(etspy_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = etspy.io.load(filename)
        del stack.original_metadata.fei_header
        del stack.original_metadata.std_header
        tilts = etspy.io.get_mrc_tilts(stack, filename)
        assert type(tilts) is np.ndarray
        assert tilts.shape[0] == stack.data.shape[0]

    def test_load_mrc_with_bad_rawtlt(self):
        filename = os.path.join(etspy_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = etspy.io.load(filename)
        del stack.original_metadata.fei_header
        del stack.original_metadata.std_header
        stack.data = np.append(stack.data, np.zeros([1, stack.data.shape[1], stack.data.shape[2]]), axis=0)
        with pytest.raises(ValueError):
            etspy.io.get_mrc_tilts(stack, filename)


class TestHspy:
    def test_convert_signal2d(self):
        tilts = np.arange(-10, 10, 2)
        data = hs.signals.Signal2D(np.zeros([10, 100, 100]))
        stack = etspy.io.create_stack(data, tilts=tilts)
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (10,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts.shape[0] == stack.data.shape[0]
        assert stack.metadata.Tomography.tilts[0] == -10
        assert type(stack) is TomoStack

    def test_load_hspy_hdf5(self):
        filename = os.path.join(etspy_path, "tests",
                                "test_data", "HAADF_Aligned.hdf5")
        stack_orig = hs.load(filename, reader='HSPY')
        stack = etspy.io.load(filename)
        with h5py.File(filename, 'r') as h5:
            h5_shape = h5['Experiments']['__unnamed__']['data'].shape
        assert stack.data.shape[1:] == h5_shape[1:]
        assert stack.data.shape[0] == h5_shape[0]
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts[0] == -76
        assert type(stack) is TomoStack
        assert stack.axes_manager[1].scale == stack_orig.axes_manager[1].scale
        assert stack.axes_manager[1].units == stack_orig.axes_manager[1].units


class TestNumpy:
    def test_numpy_to_stack_no_tilts(self):
        stack = etspy.io.create_stack(
            np.random.random([50, 100, 100]), tilts=None)
        assert type(stack) is etspy.TomoStack
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (50,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts[0] == 0
        assert stack.metadata.Tomography.tilts.shape[0] == stack.data.shape[0]
        assert type(stack) is TomoStack

    def test_numpy_to_stack_with_bad_tilts(self):
        tilts = np.arange(-50, 50, 2)
        data = np.random.random([25, 100, 100])
        with pytest.raises(ValueError):
            etspy.io.create_stack(data, tilts=tilts)

    def test_numpy_to_stack_with_tilts(self):
        tilts = np.arange(-50, 50, 2)
        stack = etspy.io.create_stack(
            np.random.random([50, 100, 100]), tilts=tilts)
        assert type(stack) is etspy.TomoStack
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (50,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts[0] == -50
        assert stack.metadata.Tomography.tilts.shape[0] == stack.data.shape[0]
        assert type(stack) is TomoStack


class TestSignal:
    def test_signal_to_stack(self):
        signal = hs.signals.Signal2D(np.random.random([50, 100, 100]))
        stack = etspy.io.create_stack(signal)
        assert stack.axes_manager[0].name == 'Tilt'
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (50,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts[0] == 0
        assert type(stack) is TomoStack
        assert stack.axes_manager[1].scale == signal.axes_manager[1].scale
        assert stack.axes_manager[1].units == signal.axes_manager[1].units

    def test_signal_to_stack_bad_tilts(self):
        signal = hs.signals.Signal2D(np.random.random([50, 100, 100]))
        tilts = np.zeros(20)
        with pytest.raises(ValueError):
            etspy.io.create_stack(signal, tilts)


class TestDM:
    def test_load_single_dm(self):
        filename = os.path.join(etspy_path, "tests",
                                "test_data", "HAADF.dm3")
        signal = hs.load(filename)
        stack = etspy.load(filename)
        assert stack.axes_manager[0].name == 'Tilt'
        assert stack.axes_manager.signal_shape == (64, 64)
        assert stack.axes_manager.navigation_shape == (91,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts[0] < 0
        assert type(stack) is TomoStack
        assert stack.axes_manager[1].scale == signal.axes_manager[1].scale
        assert stack.axes_manager[1].units == signal.axes_manager[1].units
        assert stack.axes_manager[2].scale == signal.axes_manager[2].scale
        assert stack.axes_manager[2].units == signal.axes_manager[2].units

    def test_load_dm_series(self):
        dirname = os.path.join(etspy_path, "tests",
                               "test_data", "DM_Series_Test")
        files = glob.glob(dirname + "/*.dm3")
        signal = hs.load(files, stack=True)
        stack = etspy.load(files)
        assert stack.axes_manager[0].name == 'Tilt'
        assert stack.axes_manager.signal_shape == (128, 128)
        assert stack.axes_manager.navigation_shape == (3,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts[0] < 0
        assert type(stack) is TomoStack
        assert stack.axes_manager[1].scale == signal.axes_manager[1].scale
        assert stack.axes_manager[1].units == signal.axes_manager[1].units
        assert stack.axes_manager[2].scale == signal.axes_manager[2].scale
        assert stack.axes_manager[2].units == signal.axes_manager[2].units


@pytest.mark.skipif(hspy_mrc_broken is True, reason="Hyperspy MRC reader broken")
class TestSerialEM:
    def test_load_serialem_series(self):
        dirname = os.path.join(etspy_path, "tests",
                               "test_data", "SerialEM_Multiframe_Test")
        files = glob.glob(dirname + "/*.mrc")
        stack = etspy.load(files)
        assert stack.axes_manager.signal_shape == (1024, 1024)
        assert stack.axes_manager.navigation_shape == (2, 3)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts[0] < 0
        assert type(stack) is TomoStack

    def test_load_serialem(self):
        dirname = os.path.join(etspy_path, "tests",
                               "test_data", "SerialEM_Multiframe_Test")
        # file = glob.glob(dirname + "/*.mrc")[0]
        stack = etspy.load(dirname + "/test_000.mrc")
        assert stack.axes_manager.signal_shape == (1024, 1024)
        assert stack.axes_manager.navigation_shape == (2,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts[0] == 0.0
        assert type(stack) is TomoStack

    def test_load_serial_em_explicit(self):
        dirname = os.path.join(etspy_path, "tests",
                               "test_data", "SerialEM_Multiframe_Test")
        mrcfile = glob.glob(dirname + "/*.mrc")[0]
        mdocfile = mrcfile[:-3] + "mdoc"
        stack = etspy.io.load_serialem(mrcfile, mdocfile)
        assert type(stack) is Signal2D


class TestUnknown:
    def test_load_unknown_string(self):
        filename = os.path.join(etspy_path, "tests",
                                "test_data", "HAADF.NONE")
        with pytest.raises(TypeError):
            etspy.load(filename)

    def test_load_unknown_list(self):
        filename = os.path.join(etspy_path, "tests",
                                "test_data", "HAADF.NONE")
        files = [filename, filename]
        with pytest.raises(TypeError):
            etspy.load(files)

    def test_load_unknown_type(self):
        filename = np.zeros(10)
        with pytest.raises(TypeError):
            etspy.load(filename)


class TestMRCHeader:
    def test_mrc_header_parser(self):
        filename = os.path.join(etspy_path, "tests",
                                "test_data", "HAADF.mrc")
        header = etspy.io.parse_mrc_header(filename)
        print(header)
        assert type(header) is dict
        assert header['nx'] == 256
        assert header['nextra'] == 131072
