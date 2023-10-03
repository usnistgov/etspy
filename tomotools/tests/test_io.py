import tomotools.api as tomotools
from tomotools.base import TomoStack
import hyperspy.api as hs
import os
import numpy as np
import pytest
import h5py
import glob

tomotools_path = os.path.dirname(tomotools.__file__)


class TestHspy:

    def test_convert_signal2d(self):
        tilts = np.arange(-10, 10, 2)
        data = hs.signals.Signal2D(np.zeros([10, 100, 100]))
        stack = tomotools.io.convert_to_tomo_stack(data, tilts=tilts)
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (10,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts.shape[0] == stack.data.shape[0]
        assert stack.metadata.Tomography.tilts[0] == -10
        assert type(stack) is TomoStack

    def test_load_hspy_mrc(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack_orig = hs.load(filename)
        stack = tomotools.io.load_hspy(filename)

        assert stack.axes_manager.signal_shape == (256, 256)
        assert stack.axes_manager.navigation_shape == (77,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts.shape[0] == stack.data.shape[0]
        assert stack.metadata.Tomography.tilts[0] == -76
        assert type(stack) is TomoStack
        assert stack.axes_manager[1].scale == stack_orig.axes_manager[1].scale
        assert stack.axes_manager[1].units == stack_orig.axes_manager[1].units

    def test_load_hspy_ali(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.ali")
        stack_orig = hs.load(filename)
        stack = tomotools.io.load(filename)

        assert stack.axes_manager.signal_shape == (256, 256)
        assert stack.axes_manager.navigation_shape == (77,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts.shape[0] == stack.data.shape[0]
        assert stack.metadata.Tomography.tilts[0] == -76
        assert type(stack) is TomoStack
        assert stack.axes_manager[1].scale == stack_orig.axes_manager[1].scale
        assert stack.axes_manager[1].units == stack_orig.axes_manager[1].units

    def test_load_hspy_hdf5(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF_Aligned.hdf5")
        stack_orig = hs.load(filename)
        stack = tomotools.io.load_hspy(filename)
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
        stack = tomotools.io.convert_to_tomo_stack(
            np.random.random([50, 100, 100]), tilts=None)
        assert type(stack) is tomotools.TomoStack
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
            tomotools.io.convert_to_tomo_stack(data, tilts=tilts)

    def test_numpy_to_stack_with_bad_data(self):
        tilts = np.arange(-50, 0, 2)
        data = hs.signals.Signal1D(np.zeros([25, 100]))
        with pytest.raises(TypeError):
            tomotools.io.convert_to_tomo_stack(data, tilts=tilts)

    def test_numpy_to_stack_with_tilts(self):
        tilts = np.arange(-50, 50, 2)
        stack = tomotools.io.convert_to_tomo_stack(
            np.random.random([50, 100, 100]), tilts=tilts)
        assert type(stack) is tomotools.TomoStack
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (50,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts[0] == -50
        assert stack.metadata.Tomography.tilts.shape[0] == stack.data.shape[0]
        assert type(stack) is TomoStack

    def test_numpy_to_stack_with_tilt_signal(self):
        tilts = hs.signals.Signal1D(np.arange(-50, 50, 2))
        stack = tomotools.io.convert_to_tomo_stack(
            np.random.random([50, 100, 100]), tilts=tilts)
        assert type(stack) is tomotools.TomoStack
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
        stack = tomotools.io.convert_to_tomo_stack(signal)
        assert stack.axes_manager[0].name == 'Tilt'
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (50,)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts[0] == 0
        assert type(stack) is TomoStack
        assert stack.axes_manager[1].scale == signal.axes_manager[1].scale
        assert stack.axes_manager[1].units == signal.axes_manager[1].units


class TestDM:
    def test_load_single_dm(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.dm3")
        signal = hs.load(filename)
        stack = tomotools.load(filename)
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
        dirname = os.path.join(tomotools_path, "tests",
                               "test_data", "DM_Series_Test")
        files = glob.glob(dirname + "/*.dm3")
        signal = hs.load(files, stack=True)
        stack = tomotools.load(files)
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


class TestSerialEM:
    def test_serialem_series(self):
        dirname = os.path.join(tomotools_path, "tests",
                               "test_data", "SerialEM_Multiframe_Test")
        files = glob.glob(dirname + "/*.mrc")
        stack = tomotools.load(files)
        print(stack.axes_manager)
        assert stack.axes_manager[0].name == 'multiframe'
        assert stack.axes_manager.signal_shape == (1024, 1024)
        assert stack.axes_manager.navigation_shape == (2, 3)
        assert stack.metadata.has_item('Tomography')
        assert type(stack.metadata.Tomography.tilts) is np.ndarray
        assert stack.metadata.Tomography.tilts[0] < 0
        assert type(stack) is hs.signals.Signal2D


class TestUnknown:
    def test_load_unknown(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.NONE")
        with pytest.raises(ValueError):
            tomotools.load(filename)


class TestMRCHeader:
    def test_mrc_header_parser(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        header = tomotools.io.parse_mrc_header(filename)
        print(header)
        assert type(header) is dict
        assert header['nx'] == 256
        assert header['nextra'] == 131072
