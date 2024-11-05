"""Test the reconstruction module of ETSpy."""

import re
from typing import Tuple, cast

import astra
import numpy as np
import pytest

from etspy import datasets as ds
from etspy import recon
from etspy.base import RecStack, TomoStack


class TestReconstruction:
    """Test TomoStack reconstruction methods."""

    def test_recon_no_tilts(self):
        stack = ds.get_needle_data(aligned=True)
        del stack.tilts
        slices = stack.isig[120:121, :].deepcopy()
        with pytest.raises(
            ValueError,
            match=r"Tilts are not defined in stack.tilts \(values were all zeros\). "
                  r"Please set tilt values before alignment.",
        ):
            slices.reconstruct("FBP")

    def test_recon_single_slice(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :]
        tilts = stack.tilts.data.squeeze()
        rec = recon.run(slices.data, tilts, "FBP", cuda=False)
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
            match=re.escape(
                f'Invalid reconstruction algorithm "{bad_method}". Must be one of '
                '["FBP", "SIRT", "SART", or "DART"]',
            ),
        ):
            slices.reconstruct(bad_method)  # type: ignore

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
        tilts = slices.tilts.data.squeeze()
        rec = recon.run(slices.data, tilts, "FBP", cuda=False)
        data_shape = cast(Tuple[int, int, int], rec.data.shape)
        assert data_shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert data_shape[0] == slices.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_fbp_no_cuda_2d_array(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        tilts = slices.tilts.data.squeeze()
        rec = recon.run(slices.data.squeeze(), tilts, "FBP", cuda=False)
        data_shape = cast(Tuple[int, int, int], rec.data.shape)
        assert data_shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert data_shape[0] == slices.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_sirt_no_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        tilts = slices.tilts.data.squeeze()
        rec = recon.run(slices.data, tilts, "SIRT", niterations=2, cuda=False)
        data_shape = cast(Tuple[int, int, int], rec.data.shape)
        assert data_shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert data_shape[0] == slices.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_sart_no_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        tilts = slices.tilts.data.squeeze()
        rec = recon.run(slices.data, tilts, "SART", niterations=2, cuda=False)
        data_shape = cast(Tuple[int, int, int], rec.data.shape)
        assert data_shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert data_shape[0] == slices.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_dart_no_cuda(self):
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
    def test_run_dart_no_gray_levels_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        tilts = slices.tilts.data.squeeze()
        with pytest.raises(ValueError, match="gray_levels must be provided for DART"):
            recon.run(
                slices.data,
                tilts,
                "DART",
                niterations=2,
                cuda=True,
                gray_levels=None,
                dart_iterations=1,
            )

    def test_run_dart_no_gray_levels_no_cuda(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        tilts = slices.tilts.data.squeeze()
        with pytest.raises(ValueError, match="gray_levels must be provided for DART"):
            recon.run(
                slices.data,
                tilts,
                "DART",
                niterations=2,
                cuda=False,
                gray_levels=None,
                dart_iterations=1,
            )

    def test_run_dart_no_cuda_no_progress(self):
        stack = ds.get_needle_data(aligned=True)
        slices = stack.isig[120:121, :].deepcopy()
        tilts = slices.tilts.data.squeeze()
        rec = recon.run(
            slices.data,
            tilts,
            "DART",
            niterations=2,
            cuda=False,
            gray_levels=[0.0, slices.data.max() / 2, slices.data.max()],
            dart_iterations=1,
            show_progressbar=False,
        )
        data_shape = cast(Tuple[int, int, int], rec.data.shape)
        assert data_shape == (1, slices.data.shape[1], slices.data.shape[1])
        assert data_shape[0] == slices.data.shape[2]
        assert isinstance(rec, np.ndarray)

class TestAstraError:
    """Test Astra toolbox errors."""

    def test_astra_sirt_error_cpu(self):
        stack = ds.get_needle_data(aligned=True)
        _, ny, _ = stack.data.shape
        angles = stack.tilts.data.squeeze()
        sino = stack.isig[120:121, :].data.squeeze()
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
        _, ny, _ = stack.data.shape
        angles = stack.tilts.data.squeeze()
        sino = stack.isig[120:121, :].data.squeeze()
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

    def test_astra_error_cpu_bad_dims(self):
        stack = ds.get_needle_data(aligned=True)
        angles = stack.tilts.data.squeeze()
        sino = np.random.rand(3, 5, 10)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Sinogram must be two-dimensional (ntilts, y). "
                "Provided shape was (3, 5, 10).",
                ),
        ):
            recon.astra_error(
                sino,
                angles,
                method="SART",
                iterations=2,
                constrain=True,
                thresh=0,
                cuda=False,
            )

    def test_astra_error_cpu_nangles_mismatch(self):
        stack = ds.get_needle_data(aligned=True)
        angles = stack.tilts.data.squeeze()
        sino = np.random.rand(3, 10)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Number of angles must match size of the first dimension of the "
                "sinogram. [len(angles) was 77; sinogram.shape was (3, 10)] "
                "(77 != 3)",
                ),
        ):
            recon.astra_error(
                sino,
                angles,
                method="SART",
                iterations=2,
                constrain=True,
                thresh=0,
                cuda=False,
            )
