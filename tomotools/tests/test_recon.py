import tomotools
from tomotools import recon
from tomotools import datasets as ds
import numpy
import pytest


class TestReconstruction:

    def test_recon_single_slice(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120, :]
        rec = recon.run(slices, 'FBP', cuda=False)
        assert type(stack) is tomotools.base.TomoStack
        assert type(rec) is numpy.ndarray
        assert rec.data.shape[2] == slices.data.shape[1]

    def test_recon_unknown_algorithm(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        with pytest.raises(ValueError):
            slices.reconstruct('UNKNOWN')

    def test_recon_fbp_cpu(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = slices.reconstruct('FBP', cuda=False)
        assert type(stack) is tomotools.base.TomoStack
        assert type(rec) is tomotools.base.RecStack
        assert rec.data.shape[2] == slices.data.shape[1]

    def test_recon_fbp_cpu_multicore(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:122, :].deepcopy()
        rec = slices.reconstruct('FBP', cuda=False)
        assert type(stack) is tomotools.base.TomoStack
        assert type(rec) is tomotools.base.RecStack
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
        assert type(rec) is tomotools.base.RecStack
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


class TestAstraSIRTError:
    def test_astra_sirt_error_cpu(self):
        stack = ds.get_needle_data(True)
        [ntilts, ny, nx] = stack.data.shape
        angles = stack.axes_manager[0].axis
        sino = stack.isig[120, :].data
        rec_stack, error = recon.astra_sirt_error(sino, angles, iterations=2,
                                                  constrain=True, thresh=0, cuda=False)
        assert type(error) is numpy.ndarray
        assert rec_stack.shape == (2, ny, ny)
