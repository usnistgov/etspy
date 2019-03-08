import tomotools
import os

my_path = os.path.dirname(__file__)


class TestAlignStackRegister:

    def test_recon_fbp(self):
        filename = os.path.join(my_path, "test_data", "HAADF_Aligned.hdf5")
        stack = tomotools.load(filename)
        slices = stack.isig[:, 120:131].deepcopy()
        rec = slices.reconstruct('FBP')
        assert type(stack) is tomotools.base.TomoStack
        assert type(rec) is tomotools.base.TomoStack
        assert rec.axes_manager.navigation_shape[0] == \
            slices.axes_manager.signal_shape[1]

    def test_recon_sirt(self):
        filename = os.path.join(my_path, "test_data", "HAADF_Aligned.hdf5")
        stack = tomotools.load(filename)
        slices = stack.isig[:, 120:131].deepcopy()
        rec = slices.reconstruct('SIRT', constrain=True,
                                 iterations=2, thresh=0)
        assert type(stack) is tomotools.base.TomoStack
        assert type(rec) is tomotools.base.TomoStack
        assert rec.axes_manager.navigation_shape[0] == \
            slices.axes_manager.signal_shape[1]
