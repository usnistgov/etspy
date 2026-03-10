"""Test simulation functionality in ETSpy."""

import re

import numpy as np
import pytest
from hyperspy.signals import Signal2D

from etspy import simulation as sim
from etspy.base import TomoStack


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
