"""Tests for the TV operators of ETSpy's TV-regularized reconstruction."""

import logging

import astra
import numpy as np
import pytest

from etspy import recon
from etspy import recon_tv as tv
from etspy.base import RecStack, TomoStack
from etspy.recon import create_projector

from . import count_astra_objects

try:
    import cupy as cp  # type: ignore

    has_cupy = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    has_cupy = False


def _fgp_objective(x, b, weight, tv_dims=3):
    return 0.5 * ((x - b) ** 2).sum() + weight * tv.total_variation(x, tv_dims)


@pytest.fixture
def noisy_block():
    """Create a bright block in a dark volume, with Gaussian noise."""
    rng = np.random.default_rng(0)
    volume = np.zeros((6, 24, 24))
    volume[:, 6:18, 6:18] = 1.0
    return volume + rng.normal(0, 0.2, volume.shape)


class TestGradientDivergence:
    """Test the finite-difference gradient and divergence."""

    @pytest.mark.parametrize("tv_dims", [2, 3])
    @pytest.mark.parametrize("shape", [(5, 6, 7), (1, 6, 7), (2, 6, 7)])
    def test_divergence_is_negative_adjoint(self, tv_dims, shape):
        rng = np.random.default_rng(0)
        u = rng.random(shape)
        p = rng.random((tv_dims, *shape))
        lhs = (tv.gradient(u, tv_dims) * p).sum()
        rhs = -(u * tv.divergence(p, tv_dims)).sum()
        assert lhs == pytest.approx(rhs, rel=1e-12)

    def test_gradient_boundaries_and_components(self):
        u = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
        grad = tv.gradient(u, 3)
        np.testing.assert_array_equal(grad[0, :-1], 12)  # x step
        np.testing.assert_array_equal(grad[1, :, :-1], 4)  # z step
        np.testing.assert_array_equal(grad[2, :, :, :-1], 1)  # y step
        # zero at the last index along each axis (no wrap-around)
        np.testing.assert_array_equal(grad[0, -1], 0)
        np.testing.assert_array_equal(grad[1, :, -1], 0)
        np.testing.assert_array_equal(grad[2, :, :, -1], 0)
        assert tv.gradient(u, 2).shape == (2, 2, 3, 4)

    def test_invalid_tv_dims(self):
        with pytest.raises(ValueError, match="tv_dims must be 2 or 3"):
            tv.gradient(np.zeros((2, 2, 2)), 4)  # type: ignore[arg-type]


class TestTotalVariation:
    """Test the total variation and its gradient."""

    def test_total_variation_of_step(self):
        # a unit step along y gives a TV of 1 per (x, z) line
        volume = np.zeros((3, 4, 10))
        volume[:, :, 5:] = 1.0
        assert tv.total_variation(volume, 3) == pytest.approx(12.0)
        assert tv.total_variation(volume, 2) == pytest.approx(12.0)
        assert tv.total_variation(np.ones((3, 4, 10)), 3) == 0

    def test_tv_dims_2_ignores_x(self):
        # variation only along x (between slices) has no 2D TV
        volume = np.zeros((4, 5, 6))
        volume[2:] = 1.0
        assert tv.total_variation(volume, 2) == 0
        assert tv.total_variation(volume, 3) == pytest.approx(30.0)

    @pytest.mark.parametrize("tv_dims", [2, 3])
    def test_tv_gradient_matches_finite_differences(self, tv_dims):
        rng = np.random.default_rng(1)
        u = rng.random((4, 5, 6))
        eps, h = 1e-3, 1e-6
        dtv = tv.tv_gradient(u, tv_dims, eps)
        for index in [(0, 0, 0), (1, 2, 3), (3, 4, 5), (2, 0, 5)]:
            up, down = u.copy(), u.copy()
            up[index] += h
            down[index] -= h
            numeric = (
                tv.total_variation(up, tv_dims, eps)
                - tv.total_variation(down, tv_dims, eps)
            ) / (2 * h)
            assert dtv[index] == pytest.approx(numeric, rel=1e-5, abs=1e-8)


class TestTVGradientDescent:
    """Test the normalized TV gradient descent (ASD-POCS TV step)."""

    def test_reduces_tv(self, noisy_block):
        result = tv.tv_gradient_descent(noisy_block, 20, 0.5, positivity=False)
        assert tv.total_variation(result) < 0.8 * tv.total_variation(noisy_block)

    def test_step_length(self, noisy_block):
        result = tv.tv_gradient_descent(noisy_block, 1, 0.25, positivity=False)
        assert np.linalg.norm(result - noisy_block) == pytest.approx(0.25)
        result = tv.tv_gradient_descent(noisy_block, 4, 0.25, positivity=False)
        assert np.linalg.norm(result - noisy_block) <= 1.0 + 1e-9

    def test_positivity_and_input_unchanged(self, noisy_block):
        original = noisy_block.copy()
        result = tv.tv_gradient_descent(noisy_block, 5, 0.5)
        assert result.min() >= 0
        np.testing.assert_array_equal(noisy_block, original)

    def test_constant_volume_unchanged(self):
        volume = np.full((3, 4, 5), 2.0)
        np.testing.assert_array_equal(tv.tv_gradient_descent(volume, 5, 1.0), volume)


class TestTVFGP:
    """Test FGP total variation denoising (FISTA-TV proximal step)."""

    def test_decreases_objective_and_converges(self, noisy_block):
        weight = 0.3
        objectives = [
            _fgp_objective(tv.tv_fgp(noisy_block, weight, n), noisy_block, weight)
            for n in (1, 20, 100, 500)
        ]
        assert objectives[0] < _fgp_objective(
            np.clip(noisy_block, 0, None),
            noisy_block,
            weight,
        )
        assert objectives == sorted(objectives, reverse=True)
        assert objectives[2] == pytest.approx(objectives[3], rel=0.005)

    def test_denoises(self, noisy_block):
        truth = np.zeros_like(noisy_block)
        truth[:, 6:18, 6:18] = 1.0
        result = tv.tv_fgp(noisy_block, 0.3, 100)
        assert np.abs(result - truth).mean() < 0.5 * np.abs(noisy_block - truth).mean()

    def test_positivity(self, noisy_block):
        assert tv.tv_fgp(noisy_block, 0.3, 20).min() >= 0
        assert tv.tv_fgp(noisy_block, 0.01, 20, positivity=False).min() < 0

    def test_zero_weight(self, noisy_block):
        np.testing.assert_array_equal(
            tv.tv_fgp(noisy_block, 0.0),
            np.clip(noisy_block, 0, None),
        )

    def test_tv_dims_2_treats_slices_independently(self, noisy_block):
        whole = tv.tv_fgp(noisy_block, 0.3, 20, tv_dims=2)
        for i in range(noisy_block.shape[0]):
            single = tv.tv_fgp(noisy_block[i : i + 1], 0.3, 20, tv_dims=2)
            np.testing.assert_allclose(whole[i], single[0], atol=1e-12)


@pytest.mark.skipif(not has_cupy, reason="CuPy with a GPU is not available")
class TestCupy:
    """Test that the TV operators give the same results with CuPy."""

    def test_cupy_matches_numpy(self, noisy_block):
        gpu = cp.asarray(noisy_block)
        for func in (
            lambda v: tv.tv_gradient(v, 3),
            lambda v: tv.tv_gradient_descent(v, 3, 0.5),
            lambda v: tv.tv_fgp(v, 0.3, 10),
        ):
            np.testing.assert_allclose(
                cp.asnumpy(func(gpu)),
                func(noisy_block),
                rtol=1e-6,
                atol=1e-9,
            )


@pytest.fixture(scope="module")
def limited_angle_phantom():
    """Two-level phantom and its limited-angle (+/-60 degree) sinogram."""
    n = 48
    phantom = np.zeros((n, n), np.float32)
    phantom[12:30, 14:34] = 0.5
    phantom[18:24, 20:26] = 1.0
    tilts = np.arange(-60, 61, 4.0)
    vol_geom = astra.create_vol_geom(n, n)
    proj_geom = astra.create_proj_geom("parallel", 1.0, n, np.deg2rad(tilts))
    proj_id = create_projector(proj_geom, vol_geom)
    sino_id, sino = astra.create_sino(phantom, proj_id)
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)
    return phantom, sino.astype(np.float32), tilts


def _relative_error(rec, phantom):
    return np.linalg.norm(rec - phantom) / np.linalg.norm(phantom)


class TestASDPOCS:
    """Test ASD-POCS total variation reconstruction."""

    def test_history(self, limited_angle_phantom):
        _, sino, tilts = limited_angle_phantom
        _, history = tv.asd_pocs(
            sino,
            tilts,
            iterations=20,
            sart_relaxation=0.5,
            sart_relaxation_reduction=0.9,
            show_progressbar=False,
        )
        assert set(history) == {"data_distance", "tv", "tv_step", "relaxation"}
        assert all(values.shape == (20,) for values in history.values())
        assert history["data_distance"][-1] < history["data_distance"][0]
        assert np.all(np.diff(history["tv_step"]) <= 0)
        np.testing.assert_allclose(history["relaxation"], 0.5 * 0.9 ** np.arange(20))

    def test_scale_invariance(self, limited_angle_phantom):
        phantom, sino, tilts = limited_angle_phantom
        rec, history = tv.asd_pocs(sino, tilts, iterations=30, show_progressbar=False)
        scaled, scaled_history = tv.asd_pocs(
            sino * 1000,
            tilts,
            iterations=30,
            show_progressbar=False,
        )
        assert _relative_error(scaled[0] / 1000, phantom) == pytest.approx(
            _relative_error(rec[0], phantom),
            abs=0.01,
        )
        assert scaled_history["data_distance"][-1] == pytest.approx(
            history["data_distance"][-1],
            abs=0.005,
        )

    def test_repeatable(self, limited_angle_phantom):
        _, sino, tilts = limited_angle_phantom
        first, _ = tv.asd_pocs(sino, tilts, iterations=10, show_progressbar=False)
        second, _ = tv.asd_pocs(sino, tilts, iterations=10, show_progressbar=False)
        np.testing.assert_array_equal(first, second)

    def test_tv_dims_2_reconstructs_slices_independently(self, limited_angle_phantom):
        _, sino, tilts = limited_angle_phantom
        # two different slices with the same maximum (so the same internal scaling)
        stack = np.stack([sino, sino[:, ::-1]], axis=2)
        both, history = tv.asd_pocs(
            stack,
            tilts,
            iterations=10,
            tv_dims=2,
            show_progressbar=False,
        )
        for s in range(2):
            alone, _ = tv.asd_pocs(
                stack[:, :, s],
                tilts,
                iterations=10,
                tv_dims=2,
                show_progressbar=False,
            )
            np.testing.assert_allclose(both[s], alone[0], rtol=1e-5, atol=1e-6)
        assert history["data_distance"].shape == (10,)

    def test_shapes(self, limited_angle_phantom):
        _, sino, tilts = limited_angle_phantom
        rec, _ = tv.asd_pocs(
            sino,
            tilts,
            iterations=2,
            thickness=32,
            show_progressbar=False,
        )
        assert rec.shape == (1, 32, 48)

    def test_errors(self, limited_angle_phantom):
        _, sino, tilts = limited_angle_phantom
        with pytest.raises(ValueError, match="Number of tilts"):
            tv.asd_pocs(sino, tilts[:-1], iterations=1)
        with pytest.raises(ValueError, match="tv_dims must be 2 or 3"):
            tv.asd_pocs(sino, tilts, iterations=1, tv_dims=1)  # type: ignore[arg-type]

    @pytest.mark.usefixtures("astra_sentinel")
    @pytest.mark.parametrize("tv_dims", [2, 3])
    def test_frees_astra_objects(self, limited_angle_phantom, tv_dims):
        _, sino, tilts = limited_angle_phantom
        stack = np.repeat(sino[:, :, np.newaxis], 2, axis=2)
        before = count_astra_objects()
        tv.asd_pocs(stack, tilts, iterations=2, tv_dims=tv_dims, show_progressbar=False)
        assert count_astra_objects() == before


class TestFISTATV:
    """Test FISTA total variation reconstruction."""

    def test_operator_norm_estimate(self):
        # compare the power iteration estimate with |A|**2 of the explicit matrix
        n, tilts = 8, np.deg2rad(np.arange(-60, 61, 15.0))
        vol_geom = astra.create_vol_geom(n, n)
        proj_geom = astra.create_proj_geom("parallel", 1.0, n, tilts)
        proj_id = create_projector(proj_geom, vol_geom)
        columns = []
        for k in range(n * n):
            basis = np.zeros(n * n, np.float32)
            basis[k] = 1
            sino_id, sino = astra.create_sino(basis.reshape(n, n), proj_id)
            astra.data2d.delete(sino_id)
            columns.append(sino.ravel())
        estimate = tv._operator_norm_squared(proj_id, (n, n))  # noqa: SLF001
        astra.projector.delete(proj_id)
        exact = np.linalg.norm(np.stack(columns, axis=1), 2) ** 2
        assert estimate == pytest.approx(exact, rel=1e-3)

    def test_more_accurate_than_sirt_and_sart(self, limited_angle_phantom):
        phantom, sino, tilts = limited_angle_phantom
        stack = np.repeat(sino[:, :, np.newaxis], 2, axis=2)
        rec, _ = tv.fista_tv(stack, tilts, iterations=60, show_progressbar=False)
        fista_error = _relative_error(rec[0], phantom)
        for method, iterations in (("SIRT", 50), ("SART", 20)):
            other = recon.run(
                stack,
                tilts,
                method,
                niterations=iterations,
                constrain=True,
                show_progressbar=False,
            )
            assert fista_error < 0.85 * _relative_error(other[0], phantom)

    def test_noisy_data(self, limited_angle_phantom):
        phantom, sino, tilts = limited_angle_phantom
        rng = np.random.default_rng(0)
        noisy = sino + rng.normal(0, 0.05 * sino.max(), sino.shape).astype(np.float32)
        rec, _ = tv.fista_tv(
            noisy,
            tilts,
            iterations=60,
            tv_weight=0.3,
            show_progressbar=False,
        )
        sirt = recon.run(noisy, tilts, "SIRT", constrain=True, show_progressbar=False)
        assert _relative_error(rec[0], phantom) < 0.7 * _relative_error(
            sirt[0],
            phantom,
        )

    def test_history_and_positivity(self, limited_angle_phantom):
        _, sino, tilts = limited_angle_phantom
        rec, history = tv.fista_tv(sino, tilts, iterations=30, show_progressbar=False)
        assert set(history) == {"objective", "data_distance", "tv"}
        assert all(values.shape == (30,) for values in history.values())
        objective = history["objective"]
        assert objective[-1] < 0.1 * objective[0]
        assert objective[-1] <= 1.01 * objective.min()
        assert rec.min() >= 0

    def test_scale_invariance(self, limited_angle_phantom):
        _, sino, tilts = limited_angle_phantom
        rec, _ = tv.fista_tv(sino, tilts, iterations=20, show_progressbar=False)
        scaled, _ = tv.fista_tv(
            sino * 1000, tilts, iterations=20, show_progressbar=False
        )
        np.testing.assert_allclose(scaled / 1000, rec, rtol=1e-3, atol=1e-6)

    def test_tv_dims_2_reconstructs_slices_independently(self, limited_angle_phantom):
        _, sino, tilts = limited_angle_phantom
        stack = np.stack([sino, sino[:, ::-1]], axis=2)
        both, _ = tv.fista_tv(
            stack,
            tilts,
            iterations=10,
            tv_dims=2,
            show_progressbar=False,
        )
        for s in range(2):
            alone, _ = tv.fista_tv(
                stack[:, :, s],
                tilts,
                iterations=10,
                tv_dims=2,
                show_progressbar=False,
            )
            np.testing.assert_allclose(both[s], alone[0], rtol=1e-4, atol=1e-6)

    def test_shapes_and_errors(self, limited_angle_phantom):
        _, sino, tilts = limited_angle_phantom
        rec, _ = tv.fista_tv(
            sino,
            tilts,
            iterations=2,
            thickness=32,
            show_progressbar=False,
        )
        assert rec.shape == (1, 32, 48)
        with pytest.raises(ValueError, match="Number of tilts"):
            tv.fista_tv(sino, tilts[:-1], iterations=1)

    @pytest.mark.usefixtures("astra_sentinel")
    def test_frees_astra_objects(self, limited_angle_phantom):
        _, sino, tilts = limited_angle_phantom
        stack = np.repeat(sino[:, :, np.newaxis], 2, axis=2)
        before = count_astra_objects()
        tv.fista_tv(stack, tilts, iterations=2, show_progressbar=False)
        assert count_astra_objects() == before


class TestReconstructTV:
    """Test TomoStack.reconstruct_tv."""

    @pytest.fixture
    def stack(self, limited_angle_phantom):
        _, sino, tilts = limited_angle_phantom
        data = np.repeat(sino[:, :, np.newaxis], 3, axis=2)
        return TomoStack(data, tilts=tilts)

    @pytest.mark.parametrize("method", ["FISTA", "ASD-POCS"])
    def test_matches_algorithm_and_records_metadata(self, stack, method):
        rec = stack.reconstruct_tv(
            method,
            iterations=5,
            tv_dims=2,
            cuda=False,
            show_progressbar=False,
        )
        func = tv.fista_tv if method == "FISTA" else tv.asd_pocs
        expected, history = func(
            stack.data,
            stack.tilts.data,
            iterations=5,
            tv_dims=2,
            show_progressbar=False,
        )
        assert isinstance(rec, RecStack)
        np.testing.assert_array_equal(rec.data, expected)
        assert [ax.name for ax in rec.axes_manager._axes] == ["x", "z", "y"]  # noqa: SLF001
        meta = rec.metadata.Reconstruction
        assert meta.method == method
        assert meta.parameters.iterations == 5  # noqa: PLR2004
        assert meta.parameters.tv_dims == 2  # noqa: PLR2004
        for key, values in history.items():
            np.testing.assert_array_equal(meta.history[key], values)

    def test_thickness(self, stack):
        rec = stack.reconstruct_tv(
            iterations=2,
            thickness=32,
            cuda=False,
            show_progressbar=False,
        )
        assert rec.data.shape == (3, 32, 48)
        assert rec.axes_manager["z"].size == 32  # noqa: PLR2004

    def test_invalid_method(self, stack):
        with pytest.raises(ValueError, match='Invalid TV reconstruction method "TV"'):
            stack.reconstruct_tv("TV")  # type: ignore[arg-type]

    def test_no_tilts(self, stack):
        del stack.tilts
        with pytest.raises(ValueError, match="Tilts are not defined"):
            stack.reconstruct_tv(cuda=False)

    def test_background_warning(self, stack, caplog):
        offset = stack.deepcopy()
        offset.data = offset.data + 0.2 * offset.data.max()
        with caplog.at_level(logging.WARNING, logger="etspy.base"):
            offset.reconstruct_tv(iterations=1, cuda=False, show_progressbar=False)
        assert "background (vacuum) level" in caplog.text
