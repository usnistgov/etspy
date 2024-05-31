import tomotools.datasets as ds
import matplotlib
import tomotools
import numpy
from tomotools import recon
from tomotools.base import TomoStack
import astra
import pytest


@pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
class TestAlignCUDA:
    def test_test_align_cuda(self):
        stack = ds.get_needle_data(True)
        stack.test_align(thickness=200, cuda=True)
        fig = matplotlib.pylab.gcf()
        assert len(fig.axes) == 3


@pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
class TestReconCUDA:
    def test_recon_fbp_gpu(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = slices.reconstruct('FBP', cuda=True)
        assert type(stack) is tomotools.base.TomoStack
        assert type(rec) is tomotools.base.RecStack
        assert rec.data.shape[2] == slices.data.shape[1]

    def test_recon_sirt_gpu(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = slices.reconstruct('SIRT',
                                 constrain=True,
                                 iterations=2,
                                 thresh=0,
                                 cuda=True)
        assert type(stack) is tomotools.base.TomoStack
        assert type(rec) is tomotools.base.RecStack
        assert rec.data.shape[2] == slices.data.shape[1]

    def test_recon_dart_gpu(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        gray_levels = [0., slices.data.max() / 2, slices.data.max()]
        rec = slices.reconstruct('DART',
                                 constrain=True,
                                 iterations=2,
                                 thresh=0,
                                 cuda=True,
                                 gray_levels=gray_levels,
                                 dart_iterations=1)
        assert type(stack) is tomotools.base.TomoStack
        assert type(rec) is tomotools.base.RecStack
        assert rec.data.shape[2] == slices.data.shape[1]


@pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
class TestAstraSIRTGPU:
    def test_astra_sirt_error_gpu(self):
        stack = ds.get_needle_data(True)
        [ntilts, ny, nx] = stack.data.shape
        angles = stack.metadata.Tomography.tilts
        sino = stack.isig[120, :].data
        rec_stack, error = recon.astra_sirt_error(sino, angles, iterations=2,
                                                  constrain=True, thresh=0, cuda=True)
        assert type(error) is numpy.ndarray
        assert rec_stack.shape == (2, ny, ny)


@pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
class TestReconRunCUDA:
    def test_run_fbp_cuda(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = recon.run(slices, 'FBP', cuda=True)
        assert rec.data.shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert rec.data.shape[0] == slices.data.shape[2]
        assert type(rec) is numpy.ndarray

    def test_run_sirt_cuda(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = recon.run(slices, 'SIRT', niterations=2, cuda=True)
        assert rec.data.shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert rec.data.shape[0] == slices.data.shape[2]
        assert type(rec) is numpy.ndarray

    def test_run_dart_cuda(self):
        stack = ds.get_needle_data(True)
        slices = stack.isig[120:121, :].deepcopy()
        gray_levels = [0., slices.data.max() / 2, slices.data.max()]
        rec = recon.run(slices, 'DART', niterations=2, cuda=False, gray_levels=gray_levels, dart_iterations=1)
        assert rec.data.shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert rec.data.shape[0] == slices.data.shape[2]
        assert type(rec) is numpy.ndarray


@pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
class TestStackRegisterCUDA:
    def test_register_pc_cuda(self):
        stack = ds.get_needle_data()
        stack.metadata.Tomography.shifts = \
            stack.metadata.Tomography.shifts[0:20]
        reg = stack.inav[0:20].stack_register('PC', cuda=True)
        assert type(reg) is TomoStack
        assert reg.axes_manager.signal_shape == \
            stack.inav[0:20].axes_manager.signal_shape
        assert reg.axes_manager.navigation_shape == \
            stack.inav[0:20].axes_manager.navigation_shape
