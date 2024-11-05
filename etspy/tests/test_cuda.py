"""Tests for the CUDA-enabled functionality of ETSpy."""

import re
from typing import Tuple, cast

import astra
import matplotlib.pyplot as plt
import numpy as np
import pytest

import etspy.datasets as ds
from etspy import recon
from etspy.base import RecStack, TomoStack

NUM_FIG_AXES = 3

#TODO(jat) test cuda functionality and coverage
@pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
class TestAlignCUDA:
    """Test alignment of a TomoStack using CUDA."""

    def test_test_align_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        stack.test_align(thickness=200, cuda=True)
        fig = plt.gcf()
        assert len(fig.axes) == NUM_FIG_AXES
        plt.close(fig)


@pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
class TestReconCUDA:
    """Test reconstruction of a TomoStack using CUDA."""

    def test_recon_fbp_gpu(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = slices.reconstruct("FBP", cuda=True)
        assert isinstance(stack, TomoStack)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == slices.data.shape[1]

    def test_recon_sirt_gpu(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = slices.reconstruct(
            "SIRT",
            constrain=True,
            iterations=2,
            thresh=0,
            cuda=True,
        )
        assert isinstance(stack, TomoStack)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == slices.data.shape[1]

    def test_recon_sart_gpu(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = slices.reconstruct(
            "SART",
            constrain=True,
            iterations=2,
            thresh=0,
            cuda=True,
        )
        assert isinstance(stack, TomoStack)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == slices.data.shape[1]

    def test_recon_dart_gpu(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        gray_levels = [0.0, slices.data.max() / 2, slices.data.max()]
        rec = slices.reconstruct(
            "DART",
            constrain=True,
            iterations=2,
            thresh=0,
            cuda=True,
            gray_levels=gray_levels,
            dart_iterations=1,
        )
        assert isinstance(stack, TomoStack)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == slices.data.shape[1]


@pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
class TestAstraSIRTGPU:
    """Test SIRT TomoStack reconstruction using Astra toolbox."""

    def test_astra_sirt_error_gpu(self):
        stack = ds.get_needle_data(aligned=True)
        _, ny, _ = stack.data.shape
        angles = stack.tilts.data.squeeze()
        stack = stack.isig[120:121, :].squeeze()
        rec_stack, error = recon.astra_error(
            stack.data,
            angles,
            iterations=2,
            constrain=True,
            thresh=0,
            cuda=True,
        )
        assert isinstance(error, np.ndarray)
        assert rec_stack.shape == (2, ny, ny)

    def test_astra_sirt_error_gpu_bad_dims(self):
        stack = ds.get_needle_data(aligned=True)
        _, ny, _ = stack.data.shape
        angles = stack.tilts.data.squeeze()
        stack = stack.isig[120:121, :]
        with pytest.raises(ValueError, match=re.escape(
            "Sinogram must be two-dimensional (ntilts, y). "
            "Provided shape was (77, 256, 1).",
        )):
            recon.astra_error(
                stack.data,
                angles,
                iterations=2,
                constrain=True,
                thresh=0,
                cuda=True,
            )

@pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
class TestReconRunCUDA:
    """Test reconstruction of TomoStack using CUDA features."""

    def test_run_fbp_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        tilts = slices.tilts.data.squeeze()
        rec = recon.run(slices.data, tilts, "FBP", cuda=True)
        data_shape = cast(Tuple[int, int, int], rec.data.shape)
        assert data_shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert data_shape[0] == slices.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_sirt_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        tilts = slices.tilts.data.squeeze()
        rec = recon.run(slices.data, tilts, "SIRT", niterations=2, cuda=True)
        data_shape = cast(Tuple[int, int, int], rec.data.shape)
        assert data_shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert data_shape[0] == slices.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_sart_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        tilts = slices.tilts.data.squeeze()
        rec = recon.run(slices.data, tilts, "SART", niterations=2, cuda=True)
        data_shape = cast(Tuple[int, int, int], rec.data.shape)
        assert data_shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert data_shape[0] == slices.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_dart_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        gray_levels = [0.0, slices.data.max() / 2, slices.data.max()]
        tilts = slices.tilts.data.squeeze()
        rec = recon.run(
            slices.data,
            tilts,
            "DART",
            niterations=2,
            cuda=False,
            gray_levels=gray_levels,
            dart_iterations=1,
        )
        data_shape = cast(Tuple[int, int, int], rec.data.shape)
        assert data_shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert data_shape[0] == slices.data.shape[2]
        assert isinstance(rec, np.ndarray)


@pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
class TestStackRegisterCUDA:
    """Test StackReg alignment of a TomoStack using CUDA."""

    def test_register_pc_cuda(self):
        stack = ds.get_needle_data(aligned=False)
        reg = stack.inav[0:20].stack_register("PC", cuda=True)
        assert isinstance(reg, TomoStack)
        assert (
            reg.axes_manager.signal_shape == stack.inav[0:20].axes_manager.signal_shape
        )
        assert (
            reg.axes_manager.navigation_shape
            == stack.inav[0:20].axes_manager.navigation_shape
        )
