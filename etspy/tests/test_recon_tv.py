"""Tests for the TV operators of ETSpy's TV-regularized reconstruction."""

import numpy as np
import pytest

from etspy import recon_tv as tv

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
