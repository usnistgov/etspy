"""Test simulation functionality in ETSpy."""

import re

import numpy as np
import pytest
from hyperspy.signals import Signal2D

from etspy import simulation as sim
from etspy.align import fourier_rigid_transform
from etspy.base import TomoStack

from . import count_astra_objects


@pytest.fixture(scope="module")
def catalyst_proj():
    """Create full projection series of simulated catalyst model for testing."""
    model = sim.create_catalyst_model(0, volsize=(10, 10, 10))
    proj = sim.create_model_tilt_series(model)
    return proj


@pytest.fixture(scope="module")
def catalyst_model():
    """Create simulated catalyst model for testing."""
    model = sim.create_catalyst_model(0, volsize=(10, 10, 10))
    return model


class TestModels:
    """Test model creation."""

    def test_catalyst_model(self, catalyst_model):
        assert catalyst_model.data.shape == (10, 10, 10)
        assert isinstance(catalyst_model, Signal2D)

    def test_catalyst_model_with_particle(self):
        stack = sim.create_catalyst_model(
            1,
            volsize=(20, 20, 20),
            support_radius=5,
            size_interval=(2, 3),
        )
        assert stack.data.shape == (20, 20, 20)
        assert isinstance(stack, Signal2D)

    def test_cylinder_model(self):
        model = sim.create_cylinder_model()
        assert model.data.shape == (200, 200, 200)
        assert isinstance(model, Signal2D)

    def test_cylinder_model_with_others(self):
        model = sim.create_cylinder_model(add_others=True)
        assert model.data.shape == (400, 400, 400)
        assert isinstance(model, Signal2D)

    def test_tilt_series_model(self, catalyst_model):
        proj = sim.create_model_tilt_series(catalyst_model, np.arange(0, 15, 5))
        assert isinstance(proj, TomoStack)
        assert proj.data.shape == (3, 10, 10)
        assert proj.tilts.data.shape == (3, 1)
        assert proj.shifts.data.shape == (3, 2)
        assert np.all(proj.tilts.data == np.array([[0], [5], [10]]))

    def test_tilt_series_model_no_cuda(self, catalyst_model):
        proj = sim.create_model_tilt_series(
            catalyst_model,
            np.arange(0, 15, 5),
            cuda=False,
        )
        assert isinstance(proj, TomoStack)
        assert proj.data.shape == (3, 10, 10)
        assert proj.tilts.data.shape == (3, 1)
        assert proj.shifts.data.shape == (3, 2)
        assert np.all(proj.tilts.data == np.array([[0], [5], [10]]))

    def test_tilt_series_model_no_tilts(self, catalyst_model):
        proj = sim.create_model_tilt_series(catalyst_model, None)
        assert isinstance(proj, TomoStack)
        assert proj.data.shape == (90, 10, 10)
        assert proj.tilts.data.shape == (90, 1)
        assert proj.shifts.data.shape == (90, 2)
        assert np.all(proj.tilts.data.squeeze() == np.arange(0, 180, 2))


class TestModifications:
    """Test modifications to model during creation."""

    def test_misalign_stack(self, catalyst_proj):
        shifted = sim.misalign_stack(catalyst_proj)
        assert catalyst_proj.shifts.data.sum() == 0
        assert shifted.data.shape == (90, 10, 10)
        assert abs(shifted.shifts.data).sum() > 0

    def test_misalign_stack_with_shift(self, catalyst_proj):
        shifted = sim.misalign_stack(catalyst_proj, tilt_shift=2)
        assert catalyst_proj.shifts.data.sum() == 0
        assert shifted.data.shape == (90, 10, 10)
        assert abs(shifted.shifts.data).sum() > 0

    def test_misalign_stack_with_xonly(self, catalyst_proj):
        shifted = sim.misalign_stack(catalyst_proj, y_only=True)
        assert catalyst_proj.shifts.data.sum() == 0
        assert shifted.data.shape == (90, 10, 10)
        assert np.all(shifted.shifts.data[:, 1] == 0)
        assert abs(shifted.shifts.data[:, 0]).sum() > 0

    def test_misalign_stack_with_rotation(self, catalyst_proj):
        shifted = sim.misalign_stack(catalyst_proj, tilt_rotate=2)
        assert catalyst_proj.shifts.data.sum() == 0
        assert shifted.data.shape == (90, 10, 10)
        assert abs(shifted.shifts.data).sum() > 0

    @pytest.mark.filterwarnings("ignore:divide by zero encountered")
    def test_add_noise_gaussian(self, catalyst_proj):
        # omit the first image because its std is 0
        stack_snr = (
            catalyst_proj.data[1:].mean(axis=(1, 2))
            / catalyst_proj.data[1:].std(axis=(1, 2))
        ).mean()
        noisy = sim.add_noise(catalyst_proj, "gaussian")
        noisy_snr = (
            noisy.data[1:].mean(axis=(1, 2)) / noisy.data[1:].std(axis=(1, 2))
        ).mean()
        assert noisy.data.shape == (90, 10, 10)
        # ensure signal to noise is higher for non-noisy signal
        assert stack_snr > noisy_snr

    @pytest.mark.filterwarnings("ignore:divide by zero encountered")
    def test_add_noise_poissanian(self, catalyst_proj):
        stack_snr = (
            catalyst_proj.data[1:].mean(axis=(1, 2))
            / catalyst_proj.data[1:].std(axis=(1, 2))
        ).mean()
        noisy = sim.add_noise(catalyst_proj, "poissonian")
        noisy_snr = (
            noisy.data[1:].mean(axis=(1, 2)) / noisy.data[1:].std(axis=(1, 2))
        ).mean()
        assert noisy.data.shape == (90, 10, 10)
        # ensure signal to noise is higher for non-noisy signal
        assert stack_snr > noisy_snr

    def test_add_noise_invalid_type(self, catalyst_proj):
        with pytest.raises(
            ValueError,
            match=re.escape(
                'Invalid noise type "bad value". Must be one of '
                '["gaussian", "poissonian", or "shot"].',
            ),
        ):
            sim.add_noise(catalyst_proj, "bad value")  # type: ignore


class TestAstraObjectCleanup:
    """Test that simulations free the ASTRA objects they create."""

    pytestmark = pytest.mark.usefixtures("astra_sentinel")

    def test_create_model_tilt_series_frees_astra_objects(self):
        model = np.random.default_rng(0).random((3, 32, 32), np.float32)
        before = count_astra_objects()
        sim.create_model_tilt_series(model, np.linspace(-70, 70, 15), cuda=False)
        assert count_astra_objects() == before


@pytest.fixture(scope="module")
def alignment_series():
    """Small alignment test series with a known misalignment."""
    return sim.create_alignment_test_series(
        shape=(48, 40),
        tilts=np.arange(-60, 61, 10),
        tilt_axis=1.5,
        blur=1.0,
        features=(6, 3, 3, 2),
        seed=4,
    )


class TestAlignmentTestSeries:
    """Test the analytic tilt series with a known misalignment."""

    def test_shapes_and_types(self, alignment_series):
        ts = alignment_series
        assert isinstance(ts.misaligned, TomoStack)
        assert isinstance(ts.aligned, TomoStack)
        assert ts.misaligned.data.shape == (13, 48, 40)
        assert ts.aligned.data.shape == (13, 48, 40)
        assert ts.shifts.shape == (13, 2)
        assert ts.tilt_axis == 1.5  # noqa: PLR2004
        np.testing.assert_array_equal(
            ts.misaligned.tilts.data.squeeze(),
            np.arange(-60, 61, 10),
        )

    def test_background_and_scale(self, alignment_series):
        # the projections are scaled to a maximum of 1, on a background of 0.2
        aligned = alignment_series.aligned.data
        assert aligned.max() == pytest.approx(1.2, abs=1e-5)
        assert aligned.min() == pytest.approx(0.2, abs=1e-5)

    def test_matches_astra_projection(self):
        # the analytic line integrals follow the parallel beam geometry of ASTRA
        # (the projections of a voxelized copy of the features agree)
        ny, nx, nz, ss = 32, 8, 32, 4
        box = (
            sim._BOX,  # noqa: SLF001
            np.array([0.7, 6.2, -4.4]),
            sim._random_rotation(np.random.default_rng(1)),  # noqa: SLF001
            np.array([2.0, 4.0, 1.5]),
            1.0,
        )
        tilts = np.arange(-60, 61, 30.0)
        analytic = sim._render_projections(  # noqa: SLF001
            [box],
            tilts,
            ny,
            nx,
            np.zeros((len(tilts), 2)),
            0.0,
            ss,
        )
        x = (np.arange(nx * ss) + 0.5) / ss - nx / 2
        a = (np.arange(ny * ss) + 0.5) / ss - ny / 2
        b = -((np.arange(nz * ss) + 0.5) / ss - nz / 2)
        xx, bb, aa = np.meshgrid(x, b, a, indexing="ij")
        local = np.stack([xx - box[1][0], aa - box[1][1], bb - box[1][2]], -1)
        inside = np.all(np.abs(local @ box[2]) <= box[3], axis=-1)
        volume = inside.reshape(nx, ss, nz, ss, ny, ss).mean(axis=(1, 3, 5))
        numeric = sim.create_model_tilt_series(volume, tilts, cuda=False).data
        corr = np.corrcoef(numeric.ravel(), analytic.ravel())[0, 1]
        assert corr > 0.99  # noqa: PLR2004

    def test_known_alignment_recovers_projections(self, alignment_series):
        # correcting with the known shifts and rotation gives the ideal series
        ts = alignment_series
        phi = np.deg2rad(ts.tilt_axis)
        rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        corrected = fourier_rigid_transform(
            ts.misaligned.data.astype(float),
            ts.tilt_axis,
            ts.shifts[:, ::-1] @ rotation.T,
        )
        inner = (slice(None), slice(8, -8), slice(8, -8))
        ideal = ts.aligned.data[inner] - 0.2
        error = corrected[inner] - 0.2 - ideal
        before = ts.misaligned.data[inner] - 0.2 - ideal
        assert np.sqrt(np.mean(error**2)) < 0.1 * np.sqrt(np.mean(ideal**2))
        assert np.sqrt(np.mean(before**2)) > 0.4 * np.sqrt(np.mean(ideal**2))

    def test_reproducible_with_seed(self):
        kwargs = {
            "shape": (24, 16),
            "tilts": np.arange(-30, 31, 30),
            "features": (2, 1, 1, 1),
        }
        first = sim.create_alignment_test_series(seed=7, **kwargs)
        second = sim.create_alignment_test_series(seed=7, **kwargs)
        np.testing.assert_array_equal(first.misaligned.data, second.misaligned.data)
        np.testing.assert_array_equal(first.shifts, second.shifts)

    def test_poisson_noise(self):
        kwargs = {
            "shape": (24, 16),
            "tilts": np.arange(-30, 31, 30),
            "features": (2, 1, 1, 1),
        }
        clean = sim.create_alignment_test_series(seed=7, **kwargs)
        noisy = sim.create_alignment_test_series(seed=7, dose=50, **kwargs)
        diff = noisy.misaligned.data - clean.misaligned.data
        # Poisson noise with 50 counts at an intensity of 1
        expected = np.sqrt(clean.misaligned.data.mean() / 50)
        assert np.std(diff) == pytest.approx(expected, rel=0.3)
