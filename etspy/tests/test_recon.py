"""Test the reconstruction module of ETSpy."""

from typing import Tuple, cast

import numpy as np
import pytest

from etspy import datasets as ds
from etspy import recon
from etspy.base import RecStack, TomoStack


class TestReconstruction:
    """Test TomoStack reconstruction methods."""

    def test_recon_no_tilts(self):
        stack = ds.get_needle_data(aligned=True)
        stack.metadata.Tomography.tilts = None
        slices = stack.isig[120:121, :].deepcopy()
        with pytest.raises(TypeError):
            slices.reconstruct("FBP")

    def test_recon_single_slice(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120, :]
        rec = recon.run(slices, "FBP", cuda=False)
        assert isinstance(stack, TomoStack)
        assert isinstance(rec, np.ndarray)
        data_shape = rec.data.shape
        data_shape = cast(Tuple[int, int, int], data_shape)  # cast for type checking
        assert data_shape[2] == slices.data.shape[1]

    def test_recon_unknown_algorithm(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        bad_method = "UNKNOWN"
        with pytest.raises(
            ValueError,
            match=f"Unknown reconstruction algorithm: '{bad_method}'",
        ):
            slices.reconstruct(bad_method)

    def test_recon_fbp_cpu(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = slices.reconstruct("FBP", cuda=False)
        assert isinstance(stack, TomoStack)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == slices.data.shape[1]

    def test_recon_fbp_cpu_multicore(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:122, :].deepcopy()
        rec = slices.reconstruct("FBP", cuda=False)
        assert isinstance(stack, TomoStack)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == slices.data.shape[1]

    def test_recon_sirt_cpu(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = slices.reconstruct(
            "SIRT",
            constrain=True,
            iterations=2,
            thresh=0,
            cuda=False,
        )
        assert isinstance(stack, TomoStack)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == slices.data.shape[1]

    def test_recon_sart_cpu(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = slices.reconstruct(
            "SART",
            constrain=True,
            iterations=2,
            thresh=0,
            cuda=False,
        )
        assert isinstance(stack, TomoStack)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == slices.data.shape[1]

    def test_recon_dart_cpu(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        gray_levels = [0.0, slices.data.max() / 2, slices.data.max()]
        rec = slices.reconstruct(
            "DART",
            iterations=2,
            cuda=False,
            gray_levels=gray_levels,
            dart_iterations=1,
            ncores=1,
        )
        assert isinstance(stack, TomoStack)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == slices.data.shape[1]

    def test_recon_dart_cpu_multicore(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:122, :].deepcopy()
        gray_levels = [0.0, slices.data.max() / 2, slices.data.max()]
        rec = slices.reconstruct(
            "DART",
            iterations=2,
            cuda=False,
            gray_levels=gray_levels,
            dart_iterations=1,
            ncores=1,
        )
        assert isinstance(stack, TomoStack)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == slices.data.shape[1]


class TestReconRun:
    """Test the reconstruction "run" method."""

    def test_run_fbp_no_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = recon.run(slices, "FBP", cuda=False)
        data_shape = cast(Tuple[int, int, int], rec.data.shape)
        assert data_shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert data_shape[0] == slices.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_sirt_no_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = recon.run(slices, "SIRT", niterations=2, cuda=False)
        data_shape = cast(Tuple[int, int, int], rec.data.shape)
        assert data_shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert data_shape[0] == slices.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_sart_no_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        rec = recon.run(slices, "SART", niterations=2, cuda=False)
        data_shape = cast(Tuple[int, int, int], rec.data.shape)
        assert data_shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert data_shape[0] == slices.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_dart_no_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        gray_levels = [0.0, slices.data.max() / 2, slices.data.max()]
        rec = recon.run(
            slices,
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


class TestAstraError:
    """Test Astra toolbox errors."""

    def test_astra_sirt_error_cpu(self):
        stack = ds.get_needle_data(aligned=True)
        [ntilts, ny, nx] = stack.data.shape
        angles = stack.metadata.Tomography.tilts
        sino = stack.isig[120, :].data
        rec_stack, error = recon.astra_error(
            sino,
            angles,
            iterations=2,
            constrain=True,
            thresh=0,
            cuda=False,
        )
        assert isinstance(error, np.ndarray)
        assert rec_stack.shape == (2, ny, ny)

    def test_astra_sart_error_cpu(self):
        stack = ds.get_needle_data(aligned=True)
        [ntilts, ny, nx] = stack.data.shape
        angles = stack.metadata.Tomography.tilts
        sino = stack.isig[120, :].data
        rec_stack, error = recon.astra_error(
            sino,
            angles,
            method="SART",
            iterations=2,
            constrain=True,
            thresh=0,
            cuda=False,
        )
        assert isinstance(error, np.ndarray)
        assert rec_stack.shape == (2, ny, ny)
