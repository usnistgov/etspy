import tomotools
import os
from tomotools.io import load
from tomotools import recon
import numpy

tomotools_path = os.path.dirname(tomotools.__file__)


class TestReconstruction:

    def test_recon_fbp(self):
        filename = os.path.join(tomotools_path, "tests", "test_data",
                                "HAADF_Aligned.hdf5")
        stack = load(filename)
        slices = stack.isig[:, 120:121].deepcopy()
        rec = slices.reconstruct('FBP')
        assert type(stack) is tomotools.base.TomoStack
        assert type(rec) is tomotools.base.TomoStack
        assert rec.axes_manager.navigation_shape[0] == \
            slices.axes_manager.signal_shape[1]

    def test_recon_sirt(self):
        filename = os.path.join(tomotools_path, "tests", "test_data",
                                "HAADF_Aligned.hdf5")
        stack = load(filename)
        slices = stack.isig[:, 120:121].deepcopy()
        rec = slices.reconstruct('SIRT',
                                 constrain=True,
                                 iterations=2,
                                 thresh=0)
        assert type(stack) is tomotools.base.TomoStack
        assert type(rec) is tomotools.base.TomoStack
        assert rec.axes_manager.navigation_shape[0] == \
            slices.axes_manager.signal_shape[1]


class TestAstraReconstruction:

    def test_astra_sirt_3d_data(self):
        filename = os.path.join(tomotools_path, "tests", "test_data",
                                "HAADF_Aligned.hdf5")
        stack = load(filename)
        angles = stack.axes_manager[0].axis
        slices = stack.isig[:, 120:121].deepcopy().data
        rec = recon.astra_sirt(slices, angles,
                               thickness=None, iterations=2,
                               constrain=True, thresh=0, cuda=False)
        assert rec.shape == (1, slices.shape[2], slices.shape[2])
        assert rec.shape[0] == slices.shape[1]
        assert type(rec) is numpy.ndarray

    def test_astra_sirt_2d_data(self):
        filename = os.path.join(tomotools_path, "tests", "test_data",
                                "HAADF_Aligned.hdf5")
        stack = load(filename)
        angles = stack.axes_manager[0].axis
        slices = stack.isig[:, 120].deepcopy().data
        rec = recon.astra_sirt(slices, angles,
                               thickness=None, iterations=2,
                               constrain=True, thresh=0, cuda=False)
        assert rec.shape == (1, slices.shape[1], slices.shape[1])
        assert rec.shape[0] == 1
        assert type(rec) is numpy.ndarray

    def test_astra_project_3d_data(self):
        filename = os.path.join(tomotools_path, "tests", "test_data",
                                "HAADF_Aligned.hdf5")
        stack = load(filename)
        angles = stack.axes_manager[0].axis
        slices = stack.isig[:, 120:121].deepcopy().data
        rec = recon.astra_sirt(slices, angles,
                               thickness=None, iterations=1,
                               constrain=True, thresh=0, cuda=False)
        sino = recon.astra_project(rec, angles)
        assert sino.shape == (len(angles), rec.shape[0], rec.shape[2])
        assert sino.shape[1] == slices.shape[1]
        assert type(sino) is numpy.ndarray

    def test_astra_project_2d_data(self):
        filename = os.path.join(tomotools_path, "tests", "test_data",
                                "HAADF_Aligned.hdf5")
        stack = load(filename)
        angles = stack.axes_manager[0].axis
        slices = stack.isig[:, 120].deepcopy().data
        rec = recon.astra_sirt(slices, angles,
                               thickness=None, iterations=1,
                               constrain=True, thresh=0, cuda=False)
        sino = recon.astra_project(rec, angles)
        assert sino.shape == (len(angles), rec.shape[0], rec.shape[2])
        assert sino.shape[1] == 1
        assert type(sino) is numpy.ndarray
