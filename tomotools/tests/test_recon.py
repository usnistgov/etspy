import tomotools
from tomotools import recon
from tomotools import datasets as ds
import numpy
import pytest


class TestReconstruction:

    def test_recon_no_tilts(self):
        stack = ds.get_needle_data(True)
        stack.metadata.Tomography.tilts = None
        slices = stack.isig[120:121, :].deepcopy()
        with pytest.raises(ValueError):
            slices.reconstruct('FBP')

    def test_recon_unknown_algorithm(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        with pytest.raises(ValueError):
            slices.reconstruct('UNKNOWN')

    def test_fbp_unknown_cuda(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        with pytest.raises(Exception):
            slices.reconstruct('FBP', cuda='UNKNOWN')

    def test_sirt_unknown_cuda(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        with pytest.raises(Exception):
            slices.reconstruct('SIRT', cuda='UNKNOWN')

    def test_recon_fbp_cpu(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = slices.reconstruct('FBP', cuda=False)
        assert type(stack) is tomotools.base.TomoStack
        assert type(rec) is tomotools.base.TomoStack
        assert rec.data.shape[2] == slices.data.shape[1]

    def test_recon_sirt_cpu(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = slices.reconstruct('SIRT',
                                 constrain=True,
                                 iterations=2,
                                 thresh=0,
                                 cuda=False)
        assert type(stack) is tomotools.base.TomoStack
        assert type(rec) is tomotools.base.TomoStack
        assert rec.data.shape[2] == slices.data.shape[1]


class TestReconRun:
    def test_run_fbp_no_cuda(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = recon.run(slices, 'FBP', cuda=False)
        assert rec.data.shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert rec.data.shape[0] == slices.data.shape[2]
        assert type(rec) is numpy.ndarray

    def test_run_sirt_no_cuda(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = recon.run(slices, 'SIRT', iterations=2, cuda=False)
        assert rec.data.shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert rec.data.shape[0] == slices.data.shape[2]
        assert type(rec) is numpy.ndarray


class TestAstraFBPCPU:

    def test_astra_fbp_3d_data(self):
        stack = ds.get_needle_data(True)
        angles = stack.axes_manager[0].axis
        slices = stack.isig[120:121, :].deepcopy().data
        rec = recon.astra_fbp(slices, angles, cuda=False)
        assert rec.shape == (1, slices.shape[1], slices.shape[1])
        assert rec.shape[0] == slices.shape[2]
        assert type(rec) is numpy.ndarray

    def test_astra_fbp_2d_data(self):
        stack = ds.get_needle_data(True)
        angles = stack.axes_manager[0].axis
        slices = stack.isig[120, :].deepcopy().data
        rec = recon.astra_fbp(slices, angles, cuda=False)
        assert rec.shape == (1, slices.shape[1], slices.shape[1])
        assert rec.shape[0] == 1
        assert type(rec) is numpy.ndarray


class TestAstraSIRTCPU:

    def test_astra_sirt_3d_data(self):
        stack = ds.get_needle_data(True)
        angles = stack.axes_manager[0].axis
        slices = stack.isig[120:121, :].deepcopy().data
        rec = recon.astra_sirt(slices, angles,
                               thickness=None, iterations=2,
                               constrain=True, thresh=0, cuda=False)
        assert rec.shape == (1, slices.shape[1], slices.shape[1])
        assert rec.shape[0] == slices.shape[2]
        assert type(rec) is numpy.ndarray

    def test_astra_sirt_2d_data(self):
        stack = ds.get_needle_data(True)
        angles = stack.axes_manager[0].axis
        slices = stack.isig[120, :].deepcopy().data
        rec = recon.astra_sirt(slices, angles,
                               thickness=None, iterations=2,
                               constrain=True, thresh=0, cuda=False)
        assert rec.shape == (1, slices.shape[1], slices.shape[1])
        assert rec.shape[0] == 1
        assert type(rec) is numpy.ndarray

    def test_astra_project_3d_data(self):
        stack = ds.get_needle_data(True)
        angles = stack.axes_manager[0].axis
        slices = stack.isig[120:121, :].deepcopy().data
        rec = recon.astra_sirt(slices, angles,
                               thickness=None, iterations=1,
                               constrain=True, thresh=0, cuda=False)
        sino = recon.astra_project(rec, angles)
        assert sino.shape == (len(angles), rec.shape[0], rec.shape[2])
        assert sino.shape[2] == slices.shape[1]
        assert type(sino) is numpy.ndarray

    def test_astra_project_2d_data(self):
        stack = ds.get_needle_data(True)
        angles = stack.axes_manager[0].axis
        slices = stack.isig[120, :].deepcopy().data
        rec = recon.astra_sirt(slices, angles,
                               thickness=None, iterations=1,
                               constrain=True, thresh=0, cuda=False)
        sino = recon.astra_project(rec, angles)
        assert sino.shape == (len(angles), rec.shape[0], rec.shape[2])
        assert sino.shape[1] == 1
        assert type(sino) is numpy.ndarray


class TestAstraSIRTError:
    def test_astra_sirt_error_cpu(self):
        stack = ds.get_needle_data(True)
        angles = stack.axes_manager[0].axis
        slices = stack.isig[120:121, :].deepcopy()
        rec_stack, error = recon.astra_sirt_error(slices, angles, iterations=10,
                                                  constrain=True, thresh=0, cuda=False)
        assert type(error) is numpy.ndarray
