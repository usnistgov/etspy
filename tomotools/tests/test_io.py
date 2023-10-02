import tomotools.api as tomotools
from tomotools.base import TomoStack
import hyperspy.api as hs
import os
import numpy as np
import pytest
import h5py

tomotools_path = os.path.dirname(tomotools.__file__)


class TestHspy:

    def test_load_hspy_mrc(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        stack = tomotools.io.loadhspy(filename)
        assert type(stack) is TomoStack
        assert stack.axes_manager.signal_shape == (256, 256)
        assert stack.axes_manager.navigation_shape == (77,)
        assert stack.__repr__()[0:10] == '<TomoStack'

    def test_load_hspy_hdf5(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF_Aligned.hdf5")
        stack = tomotools.io.loadhspy(filename)
        with h5py.File(filename, 'r') as h5:
            h5_shape = h5['Experiments']['__unnamed__']['data'].shape
        assert type(stack) is TomoStack
        assert stack.data.shape[1:] == h5_shape[1:]
        assert stack.data.shape[0] == h5_shape[0]
        assert stack.__repr__()[0:10] == '<TomoStack'


class TestNumpy:
    def test_numpy_to_stack(self):
        stack = tomotools.io.convert_to_tomo_stack(
            np.random.random([50, 100, 100]), tilts=np.arange(0, 50))
        assert type(stack) is tomotools.TomoStack
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (50,)


class TestSignal:
    def test_signal_to_stack(self):
        signal = hs.signals.Signal2D(np.random.random([50, 100, 100]))
        stack = tomotools.io.convert_to_tomo_stack(signal)
        assert stack.axes_manager[0].name == 'Tilt'
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (50,)

    def test_signal_to_stack_with_tiltsignal(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        tilt_signal = tomotools.load(filename)
        tilt_signal.set_tilts(-76, 2)
        signal = hs.signals.Signal2D(np.random.random([77, 100, 100]))
        stack = tomotools.io.convert_to_tomo_stack(signal,
                                                   tilt_signal=tilt_signal)
        assert stack.axes_manager[0].name == 'Tilt'
        assert stack.axes_manager.signal_shape == (100, 100)
        assert stack.axes_manager.navigation_shape == (77,)
        assert stack.axes_manager[0].axis.all() == \
            tilt_signal.axes_manager[0].axis.all()

    def test_signal_to_stack_with_rawtlt_file(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        signal = hs.load(filename)
        signal.metadata.General.original_filename = filename
        del signal.metadata.Acquisition_instrument
        stack = tomotools.io.convert_to_tomo_stack(signal)
        assert stack.axes_manager[0].name == 'Tilt'
        assert stack.axes_manager.signal_shape == (256, 256)
        assert stack.axes_manager.navigation_shape == (77,)


class TestDM:
    def test_load_dm(self):
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.dm3")
        stack = tomotools.load(filename)
        assert stack.axes_manager[0].name == 'Tilt'
        assert stack.axes_manager[0].offset == -90.0
        assert stack.axes_manager.signal_shape == (64, 64)
        assert stack.axes_manager.navigation_shape == (91,)


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
