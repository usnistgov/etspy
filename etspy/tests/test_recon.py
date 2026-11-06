"""Test the reconstruction module of ETSpy."""

import logging
import re
from typing import cast
from unittest.mock import patch

import astra
import numpy as np
import pytest

from etspy import datasets as ds
from etspy import projmatch, recon
from etspy import simulation as sim
from etspy.base import RecStack, TomoStack

from . import count_astra_objects


@pytest.fixture(scope="module")
def aligned_full_stack():
    """Create full spatially registered stack from test data."""
    s = ds.get_needle_data(aligned=True)
    return s


@pytest.fixture(scope="module")
def aligned_sinogram(aligned_full_stack):
    """Extract single sinogram from aligned full stack."""
    s = aligned_full_stack.isig[120:121, :].deepcopy()
    return s


class TestReconstruction:
    """Test TomoStack reconstruction methods."""

    def test_recon_no_tilts(self, aligned_sinogram):
        aligned_sinogram_no_tilts = aligned_sinogram.deepcopy()
        del aligned_sinogram_no_tilts.tilts
        with pytest.raises(
            ValueError,
            match=r"Tilts are not defined in stack.tilts \(values were all zeros\). "
            r"Please set tilt values before alignment.",
        ):
            aligned_sinogram_no_tilts.reconstruct("FBP")

    def test_recon_single_slice(self, aligned_full_stack, aligned_sinogram):
        tilts = aligned_full_stack.tilts.data.squeeze()
        rec = recon.run(aligned_sinogram.data, tilts, "FBP", cuda=False)
        assert isinstance(aligned_full_stack, TomoStack)
        assert isinstance(rec, np.ndarray)
        data_shape = rec.data.shape
        data_shape = cast("tuple[int, int, int]", data_shape)  # cast for type checking
        assert data_shape[2] == aligned_sinogram.data.shape[1]

    def test_recon_unknown_algorithm(self, aligned_sinogram):
        bad_method = "UNKNOWN"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Invalid reconstruction algorithm "{bad_method}". Must be one of '
                '["FBP", "SIRT", "SART", or "DART"]',
            ),
        ):
            aligned_sinogram.reconstruct(bad_method)  # type: ignore

    def test_recon_fbp_cpu(self, aligned_sinogram):
        rec = aligned_sinogram.reconstruct("FBP", cuda=False)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == aligned_sinogram.data.shape[1]

    def test_recon_fbp_cpu_multicore(self, aligned_sinogram):
        rec = aligned_sinogram.reconstruct("FBP", cuda=False)
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == aligned_sinogram.data.shape[1]

    def test_recon_sirt_cpu(self, aligned_sinogram):
        rec = aligned_sinogram.reconstruct(
            "SIRT",
            constrain=True,
            iterations=2,
            thresh=0,
            cuda=False,
        )
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == aligned_sinogram.data.shape[1]

    def test_recon_sart_cpu(self, aligned_sinogram):
        rec = aligned_sinogram.reconstruct(
            "SART",
            constrain=True,
            iterations=2,
            thresh=0,
            cuda=False,
        )
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == aligned_sinogram.data.shape[1]

    def test_recon_dart_cpu(self, aligned_sinogram):
        gray_levels = [
            0.0,
            aligned_sinogram.data.max() / 2,
            aligned_sinogram.data.max(),
        ]
        rec = aligned_sinogram.reconstruct(
            "DART",
            iterations=2,
            cuda=False,
            gray_levels=gray_levels,
            dart_iterations=1,
            ncores=1,
        )
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == aligned_sinogram.data.shape[1]

    def test_recon_dart_cpu_multicore(self, aligned_sinogram):
        gray_levels = [
            0.0,
            aligned_sinogram.data.max() / 2,
            aligned_sinogram.data.max(),
        ]
        rec = aligned_sinogram.reconstruct(
            "DART",
            iterations=2,
            cuda=False,
            gray_levels=gray_levels,
            dart_iterations=1,
            ncores=1,
        )
        assert isinstance(rec, RecStack)
        assert rec.data.shape[2] == aligned_sinogram.data.shape[1]


class TestReconRun:
    """Test the reconstruction "run" method."""

    def test_run_fbp_no_cuda(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        rec = recon.run(aligned_sinogram.data, tilts, "FBP", cuda=False)
        data_shape = cast("tuple[int, int, int]", rec.data.shape)
        assert data_shape == (
            1,
            aligned_sinogram.data.shape[1],
            aligned_sinogram.data.shape[1],
        )
        assert data_shape[0] == aligned_sinogram.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_fbp_no_cuda_2d_array(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        rec = recon.run(aligned_sinogram.data.squeeze(), tilts, "FBP", cuda=False)
        data_shape = cast("tuple[int, int, int]", rec.data.shape)
        assert data_shape == (
            1,
            aligned_sinogram.data.shape[1],
            aligned_sinogram.data.shape[1],
        )
        assert data_shape[0] == aligned_sinogram.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_sirt_no_cuda(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        rec = recon.run(aligned_sinogram.data, tilts, "SIRT", niterations=2, cuda=False)
        data_shape = cast("tuple[int, int, int]", rec.data.shape)
        assert data_shape == (
            1,
            aligned_sinogram.data.shape[1],
            aligned_sinogram.data.shape[1],
        )
        assert data_shape[0] == aligned_sinogram.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_sart_no_cuda(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        rec = recon.run(aligned_sinogram.data, tilts, "SART", niterations=2, cuda=False)
        data_shape = cast("tuple[int, int, int]", rec.data.shape)
        assert data_shape == (
            1,
            aligned_sinogram.data.shape[1],
            aligned_sinogram.data.shape[1],
        )
        assert data_shape[0] == aligned_sinogram.data.shape[2]
        assert isinstance(rec, np.ndarray)

    def test_run_dart_no_cuda(self, aligned_sinogram):
        gray_levels = [
            0.0,
            aligned_sinogram.data.max() / 2,
            aligned_sinogram.data.max(),
        ]
        tilts = aligned_sinogram.tilts.data.squeeze()
        rec = recon.run(
            aligned_sinogram.data,
            tilts,
            "DART",
            niterations=2,
            cuda=False,
            gray_levels=gray_levels,
            dart_iterations=1,
        )
        data_shape = cast("tuple[int, int, int]", rec.data.shape)
        assert data_shape == (
            1,
            aligned_sinogram.data.shape[1],
            aligned_sinogram.data.shape[1],
        )
        assert data_shape[0] == aligned_sinogram.data.shape[2]
        assert isinstance(rec, np.ndarray)

    @pytest.mark.skipif(not astra.use_cuda(), reason="CUDA not detected")
    def test_run_dart_no_gray_levels_cuda(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        with pytest.raises(ValueError, match="gray_levels must be provided for DART"):
            recon.run(
                aligned_sinogram.data,
                tilts,
                "DART",
                niterations=2,
                cuda=True,
                gray_levels=None,
                dart_iterations=1,
            )

    def test_run_dart_no_gray_levels_no_cuda(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        with pytest.raises(ValueError, match="gray_levels must be provided for DART"):
            recon.run(
                aligned_sinogram.data,
                tilts,
                "DART",
                niterations=2,
                cuda=False,
                gray_levels=None,
                dart_iterations=1,
            )

    def test_run_dart_no_cuda_no_progress(self, aligned_sinogram):
        tilts = aligned_sinogram.tilts.data.squeeze()
        rec = recon.run(
            aligned_sinogram.data,
            tilts,
            "DART",
            niterations=2,
            cuda=False,
            gray_levels=[
                0.0,
                aligned_sinogram.data.max() / 2,
                aligned_sinogram.data.max(),
            ],
            dart_iterations=1,
            show_progressbar=False,
        )
        data_shape = cast("tuple[int, int, int]", rec.data.shape)
        assert data_shape == (
            1,
            aligned_sinogram.data.shape[1],
            aligned_sinogram.data.shape[1],
        )
        assert data_shape[0] == aligned_sinogram.data.shape[2]
        assert isinstance(rec, np.ndarray)


@pytest.fixture(scope="module")
def phantom_sinogram():
    """Create a three-level phantom and its limited-angle sinogram."""
    ny = 48
    phantom = np.zeros((ny, ny), np.float32)
    phantom[12:36, 12:36] = 0.5
    phantom[20:28, 20:28] = 1.0
    tilts = np.arange(-70, 71, 2.0)
    vol_geom = astra.create_vol_geom(ny, ny)
    proj_geom = astra.create_proj_geom("parallel", 1.0, ny, np.deg2rad(tilts))
    proj_id = astra.create_projector("linear", proj_geom, vol_geom)
    _, sino = astra.create_sino(phantom, proj_id)
    astra.projector.delete(proj_id)
    return phantom, sino[:, :, np.newaxis].astype(np.float32), tilts


class TestDart:
    """Test the accuracy of the DART algorithm."""

    def test_dart_beats_sart(self, phantom_sinogram):
        phantom, sino, tilts = phantom_sinogram
        niterations = 10
        np.random.seed(0)
        dart = recon.run(
            sino,
            tilts,
            "DART",
            niterations=niterations,
            gray_levels=[0.0, 0.5, 1.0],
            dart_iterations=5,
            p=0.85,
            show_progressbar=False,
        )
        sart = recon.run(
            sino,
            tilts,
            "SART",
            niterations=niterations,
            show_progressbar=False,
        )
        dart_err = np.sqrt(np.mean((dart[0] - phantom) ** 2))
        sart_err = np.sqrt(np.mean((sart[0] - phantom) ** 2))
        assert dart_err < 0.5 * sart_err

    def test_dart_non_integer_gray_levels(self):
        cfg = {"option": {}}
        gray_levels, thresholds = recon._dart_setup([1.0, 0.0, 0.5], cfg)  # noqa: SLF001
        np.testing.assert_allclose(gray_levels, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(thresholds, [0.25, 0.75])
        assert cfg["option"]["MinConstraint"] == 0.0
        assert cfg["option"]["MaxConstraint"] == 1.0

    def test_dart_not_clamped_to_255(self, phantom_sinogram):
        _, sino, tilts = phantom_sinogram
        scale = 1000.0
        rec = recon.run(
            sino * scale,
            tilts,
            "DART",
            niterations=1,
            gray_levels=[0.0, 0.5 * scale, scale],
            dart_iterations=2,
            show_progressbar=False,
        )
        assert rec.max() > 255  # noqa: PLR2004

    def test_run_dart_does_not_mutate_cfg(self, phantom_sinogram):
        _, sino, tilts = phantom_sinogram
        ny = sino.shape[1]
        vol_geom = astra.create_vol_geom(ny, ny)
        proj_geom = astra.create_proj_geom("parallel", 1.0, ny, np.deg2rad(tilts))
        cfg = {"type": "SART", "option": {"MinConstraint": 0.0}}
        recon.run_dart(
            sino[:, :, 0],
            10,
            1,
            0.9,
            np.array([0.25, 0.75]),
            np.array([0.0, 0.5, 1.0]),
            cfg,
            vol_geom,
            proj_geom,
        )
        assert cfg == {"type": "SART", "option": {"MinConstraint": 0.0}}


class TestAstraError:
    """Test Astra toolbox errors."""

    def test_astra_sirt_error_cpu(self, aligned_sinogram):
        _, ny, _ = aligned_sinogram.data.shape
        angles = aligned_sinogram.tilts.data.squeeze()
        rec_stack, error = recon.astra_error(
            aligned_sinogram.data[:, :, 0],
            angles,
            iterations=2,
            constrain=True,
            thresh=0,
            cuda=False,
        )
        assert isinstance(error, np.ndarray)
        assert rec_stack.shape == (2, ny, ny)

    def test_astra_sart_error_cpu(self, aligned_sinogram):
        _, ny, _ = aligned_sinogram.data.shape
        angles = aligned_sinogram.tilts.data.squeeze()
        rec_stack, error = recon.astra_error(
            aligned_sinogram.data[:, :, 0],
            angles,
            method="SART",
            iterations=2,
            constrain=True,
            thresh=0,
            cuda=False,
        )
        assert isinstance(error, np.ndarray)
        assert rec_stack.shape == (2, ny, ny)

    def test_astra_error_cpu_bad_dims(self, aligned_sinogram):
        angles = aligned_sinogram.tilts.data.squeeze()
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

    def test_astra_error_cpu_nangles_mismatch(self, aligned_sinogram):
        angles = aligned_sinogram.tilts.data.squeeze()
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


class TestRunAlgThreadSafety:
    """Test that CPU reconstructions do not share mutable ASTRA state."""

    def test_run_alg_does_not_mutate_cfg(self):
        ny = 32
        tilts = np.linspace(-70, 70, 15)
        vol_geom = astra.create_vol_geom(ny, ny)
        proj_geom = astra.create_proj_geom("parallel", 1.0, ny, np.deg2rad(tilts))
        cfg = {"type": "SIRT", "option": {"MinConstraint": 0.0}}
        sino = np.random.default_rng(0).random((len(tilts), ny), np.float32)
        recon.run_alg(sino, 2, cfg, vol_geom, proj_geom)
        assert cfg == {"type": "SIRT", "option": {"MinConstraint": 0.0}}

    def test_multithreaded_matches_serial(self):
        # every slice is different, so any mix-up between threads would show
        stack = np.random.default_rng(0).random((15, 32, 16), np.float32)
        tilts = np.linspace(-70, 70, 15)
        serial = recon.run(
            stack,
            tilts,
            "SIRT",
            niterations=5,
            ncores=1,
            show_progressbar=False,
        )
        threaded = recon.run(
            stack,
            tilts,
            "SIRT",
            niterations=5,
            ncores=4,
            show_progressbar=False,
        )
        np.testing.assert_array_equal(threaded, serial)


class TestAstraObjectCleanup:
    """Test that reconstructions free the ASTRA objects they create."""

    pytestmark = pytest.mark.usefixtures("astra_sentinel")

    @pytest.fixture
    def small_stack(self):
        return (
            np.random.default_rng(0).random((15, 32, 3), np.float32),
            np.linspace(-70, 70, 15),
        )

    def test_run_alg_frees_astra_objects(self, small_stack):
        stack, tilts = small_stack
        ny = stack.shape[1]
        vol_geom = astra.create_vol_geom(ny, ny)
        proj_geom = astra.create_proj_geom("parallel", 1.0, ny, np.deg2rad(tilts))
        before = count_astra_objects()
        recon.run_alg(stack[:, :, 0], 2, {"type": "SIRT"}, vol_geom, proj_geom)
        assert count_astra_objects() == before

    @pytest.mark.parametrize("method", ["FBP", "SIRT", "SART"])
    def test_run_frees_astra_objects(self, small_stack, method):
        stack, tilts = small_stack
        before = count_astra_objects()
        recon.run(stack, tilts, method, niterations=2, show_progressbar=False)
        assert count_astra_objects() == before

    def test_run_dart_frees_astra_objects(self, small_stack):
        stack, tilts = small_stack
        before = count_astra_objects()
        recon.run(
            stack,
            tilts,
            "DART",
            niterations=2,
            gray_levels=[0.0, 0.5, 1.0],
            dart_iterations=2,
            show_progressbar=False,
        )
        assert count_astra_objects() == before

    def test_astra_error_frees_astra_objects(self, small_stack):
        stack, tilts = small_stack
        before = count_astra_objects()
        recon.astra_error(stack[:, :, 0], tilts, iterations=3)
        assert count_astra_objects() == before

    def test_run_keeps_other_astra_objects(self, small_stack):
        stack, tilts = small_stack
        user_vol = np.ones((8, 8), np.float32)
        user_id = astra.data2d.create("-vol", astra.create_vol_geom(8, 8), user_vol)
        try:
            recon.run(stack, tilts, "SIRT", niterations=2, show_progressbar=False)
            np.testing.assert_array_equal(astra.data2d.get(user_id), user_vol)
        finally:
            astra.data2d.delete(user_id)


class TestAstraErrorProgressBar:
    """Test that astra_error respects show_progressbar."""

    @pytest.mark.parametrize("show_progressbar", [True, False])
    def test_astra_error_progressbar(self, capsys, show_progressbar):
        sino = np.random.default_rng(0).random((15, 32), np.float32)
        recon.astra_error(
            sino,
            np.linspace(-70, 70, 15),
            iterations=3,
            show_progressbar=show_progressbar,
        )
        assert ("it/s" in capsys.readouterr().err) == show_progressbar


class TestConsistentProjector:
    """Test that all CPU projections use the same (linear) forward model."""

    @pytest.fixture
    def projector_types(self, monkeypatch):
        """Record the type of every ASTRA projector created."""
        created = []
        orig = astra.create_projector

        def recording_create_projector(proj_type, *args, **kwargs):
            created.append(proj_type)
            return orig(proj_type, *args, **kwargs)

        monkeypatch.setattr(astra, "create_projector", recording_create_projector)
        return created

    def test_cpu_code_paths_use_linear_projector(self, projector_types):
        tilts = np.linspace(-70, 70, 15)
        stack = TomoStack(
            np.random.default_rng(0).random((15, 32, 2), np.float32),
            tilts=tilts,
        )
        for method in ("FBP", "SIRT", "SART"):
            stack.reconstruct(method, iterations=2, cuda=False, show_progressbar=False)
        stack.reconstruct(
            "DART",
            iterations=2,
            gray_levels=[0.0, 0.5, 1.0],
            dart_iterations=1,
            cuda=False,
            show_progressbar=False,
        )
        stack.recon_error(iterations=2, cuda=False)
        rec = stack.reconstruct("FBP", cuda=False, show_progressbar=False)
        rec.forward_project(tilts, cuda=False)
        model = np.ones((2, 32, 32), np.float32)
        sim.create_model_tilt_series(model, tilts, cuda=False)
        projmatch.ProjMatch(stack.isig[0:1, :], cuda=False).update_geometries(32)
        assert projector_types
        assert set(projector_types) == {"linear"}


class TestIterationCounting:
    """Test that iterations are full passes through the projections."""

    @pytest.fixture
    def astra_runs(self, monkeypatch):
        """Record the number of ASTRA iterations requested in each run."""
        runs = []
        orig = astra.algorithm.run

        def recording_run(alg_id, iterations=1):
            runs.append(iterations)
            return orig(alg_id, iterations)

        monkeypatch.setattr(astra.algorithm, "run", recording_run)
        return runs

    @pytest.mark.parametrize(
        ("method", "iterations", "expected"),
        [
            ("FBP", None, (1, 1)),
            ("FBP", 7, (1, 1)),
            ("SIRT", None, (50, 50)),
            ("SIRT", 7, (7, 7)),
            ("SART", None, (5, 5 * 15)),
            ("SART", 3, (3, 3 * 15)),
            ("DART", None, (5, 5 * 15)),
        ],
    )
    def test_astra_iterations(self, method, iterations, expected):
        assert recon.astra_iterations(method, iterations, 15) == expected

    def test_sart_many_passes_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="etspy.recon"):
            recon.astra_iterations("SART", 20, 15)
        assert "counted as full passes through all 15 projections" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="etspy.recon"):
            recon.astra_iterations("SART", 5, 15)
            recon.astra_iterations("SART", None, 3)
        assert "counted as full passes" not in caplog.text

    @pytest.mark.parametrize(
        ("method", "iterations", "expected"),
        [("SIRT", None, 50), ("SART", None, 75), ("SART", 2, 30)],
    )
    def test_reconstruct_runs_passes(self, astra_runs, method, iterations, expected):
        stack = TomoStack(
            np.random.default_rng(0).random((15, 32, 2), np.float32),
            tilts=np.linspace(-70, 70, 15),
        )
        stack.reconstruct(
            method,
            iterations=iterations,
            cuda=False,
            ncores=1,
            show_progressbar=False,
        )
        assert astra_runs == [expected, expected]  # one run per slice

    @pytest.mark.xfail(reason="Requires further investigation", strict=True)
    def test_recon_error_sart_records_each_pass(self, astra_runs):
        stack = TomoStack(
            np.random.default_rng(0).random((15, 32, 4), np.float32),
            tilts=np.linspace(-70, 70, 15),
        )
        rec_stack, error = stack.recon_error(algorithm="SART", cuda=False)
        assert error.data.shape == (50,)
        assert rec_stack.data.shape[0] == 50  # noqa: PLR2004
        # each pass runs 15 SART iterations, followed by a forward projection (1)
        assert astra_runs == [50, 1]


def _no_gpu():
    return False


def _gpu():
    return True


class TestResolveCuda:
    """Test the decision whether to use CUDA."""

    def test_none_with_gpu(self, caplog):
        with (
            patch("astra.use_cuda", new=_gpu),
            caplog.at_level(logging.INFO, logger="etspy.recon"),
        ):
            assert recon.resolve_cuda(None) is True
        assert "CUDA detected with Astra" in caplog.text

    def test_none_without_gpu(self, caplog):
        with (
            patch("astra.use_cuda", new=_no_gpu),
            caplog.at_level(logging.INFO, logger="etspy.recon"),
        ):
            assert recon.resolve_cuda(None) is False
        assert "CUDA not detected with Astra" in caplog.text

    def test_false_does_not_check_gpu(self):
        with patch("astra.use_cuda", side_effect=AssertionError("not called")):
            assert recon.resolve_cuda(False) is False  # noqa: FBT003

    def test_true_with_gpu(self):
        with patch("astra.use_cuda", new=_gpu):
            assert recon.resolve_cuda(True) is True  # noqa: FBT003

    def test_true_without_gpu(self):
        with (
            patch("astra.use_cuda", new=_no_gpu),
            pytest.raises(RuntimeError, match="ASTRA did not detect a CUDA GPU"),
        ):
            recon.resolve_cuda(True)  # noqa: FBT003

    @pytest.mark.parametrize(
        "call",
        [
            lambda s: s.reconstruct("SIRT", iterations=1, cuda=True),
            lambda s: s.recon_error(iterations=1, cuda=True),
            lambda s: s.test_align(cuda=True),
            lambda s: RecStack(np.ones((2, 8, 16))).forward_project(
                s.tilts,
                cuda=True,
            ),
            lambda _: sim.create_model_tilt_series(np.ones((2, 8, 16)), cuda=True),
        ],
        ids=[
            "reconstruct",
            "recon_error",
            "test_align",
            "forward_project",
            "create_model_tilt_series",
        ],
    )
    def test_entry_points_raise_without_gpu(self, call):
        stack = TomoStack(
            np.ones((5, 16, 4), np.float32),
            tilts=np.linspace(-60, 60, 5),
        )
        with (
            patch("astra.use_cuda", new=_no_gpu),
            pytest.raises(RuntimeError, match="ASTRA did not detect a CUDA GPU"),
        ):
            call(stack)


class TestAlgorithmConfig:
    """Test the ASTRA configuration built for each reconstruction method."""

    @pytest.mark.parametrize("cuda", [False, True])
    @pytest.mark.parametrize(
        ("method", "algorithm", "options"),
        [
            ("FBP", "FBP", {"FilterType": "ram-lak"}),
            ("SIRT", "SIRT", {"MinConstraint": 0.5}),
            ("SART", "SART", {"MinConstraint": 0.5}),
            ("DART", "SART", {"MinConstraint": 0.0, "MaxConstraint": 2.0}),
        ],
    )
    def test_config(self, method, algorithm, options, cuda):
        cfg, gray_levels, thresholds = recon._algorithm_config(  # noqa: SLF001
            method.lower(),
            cuda=cuda,
            constrain=True,
            thresh=0.5,
            bp_filter="Ram-Lak",
            gray_levels=[2.0, 0.0, 1.0],
        )
        assert cfg == {
            "type": algorithm + "_CUDA" if cuda else algorithm,
            "option": options,
        }
        if method == "DART":
            np.testing.assert_array_equal(gray_levels, [0.0, 1.0, 2.0])
            np.testing.assert_array_equal(thresholds, [0.5, 1.5])
        else:
            assert thresholds.size == 0

    def test_unconstrained(self):
        cfg, _, _ = recon._algorithm_config("SIRT", cuda=False)  # noqa: SLF001
        assert cfg == {"type": "SIRT", "option": {}}

    def test_run_unknown_method(self):
        with pytest.raises(ValueError, match='Invalid reconstruction method "WBP"'):
            recon.run(np.ones((3, 8)), np.array([-10.0, 0, 10]), "WBP", cuda=False)  # type: ignore[arg-type]

    @pytest.mark.parametrize("method", ["FBP", "DART"])
    def test_astra_error_unsupported_method(self, method):
        with pytest.raises(ValueError, match="Must be SIRT or SART"):
            recon.astra_error(
                np.ones((3, 8)),
                np.array([-10.0, 0, 10]),
                method,  # type: ignore[arg-type]
            )

    def test_run_resolves_cuda_none(self, caplog):
        # cuda=None previously meant the CPU without checking for a GPU
        with (
            patch("astra.use_cuda", new=_no_gpu),
            caplog.at_level(logging.INFO, logger="etspy.recon"),
        ):
            recon.run(
                np.ones((3, 8, 2), np.float32),
                np.array([-10.0, 0, 10]),
                "SIRT",
                niterations=1,
                cuda=None,
                show_progressbar=False,
            )
        assert "CUDA not detected with Astra" in caplog.text
        assert "Reconstructing with CPU-based SIRT algorithm" in caplog.text

    def test_run_at_least_one_worker(self):
        with (
            patch("etspy.recon.mp.cpu_count", return_value=1),
            patch("etspy.recon.dask_compute", wraps=recon.dask_compute) as compute,
        ):
            recon.run(
                np.ones((3, 8, 2), np.float32),
                np.array([-10.0, 0, 10]),
                "SIRT",
                niterations=1,
                cuda=False,
                show_progressbar=False,
            )
        assert compute.call_args.kwargs["num_workers"] == 1
