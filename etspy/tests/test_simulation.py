"""Test simulation functionality in ETSpy."""

import numpy as np
from hyperspy.signals import Signal2D

from etspy import simulation as sim


class TestModels:
    """Test model creation."""

    def test_catalyst_model(self):
        stack = sim.create_catalyst_model(0, volsize=(10, 10, 10))
        assert stack.data.shape == (10, 10, 10)

    def test_catalyst_model_with_particle(self):
        stack = sim.create_catalyst_model(
            1,
            volsize=(20, 20, 20),
            support_radius=5,
            size_interval=(2, 3),
        )
        assert stack.data.shape == (20, 20, 20)

    def test_cylinder_model(self):
        model = sim.create_cylinder_model()
        assert model.data.shape == (200, 200, 200)
        assert isinstance(model, Signal2D)

    def test_cylinder_model_with_others(self):
        model = sim.create_cylinder_model(add_others=True)
        assert model.data.shape == (400, 400, 400)
        assert isinstance(model, Signal2D)

    def test_tilt_series_model(self):
        stack = sim.create_catalyst_model(0, volsize=(10, 10, 10))
        proj = sim.create_model_tilt_series(stack, np.arange(0, 15, 5))
        assert proj.data.shape == (3, 10, 10)

    def test_tilt_series_model_no_cuda(self):
        stack = sim.create_catalyst_model(0, volsize=(10, 10, 10))
        proj = sim.create_model_tilt_series(stack, np.arange(0, 15, 5), cuda=False)
        assert proj.data.shape == (3, 10, 10)

    def test_tilt_series_model_no_tilts(self):
        stack = sim.create_catalyst_model(0, volsize=(10, 10, 10))
        proj = sim.create_model_tilt_series(stack, None)
        assert proj.data.shape == (90, 10, 10)


class TestModifications:
    """Test modifications to model during creation."""

    def test_misalign_stack(self):
        model = sim.create_catalyst_model(0, volsize=(10, 10, 10))
        stack = sim.create_model_tilt_series(model)
        shifted = sim.misalign_stack(stack)
        assert shifted.data.shape == (90, 10, 10)

    def test_misalign_stack_with_shift(self):
        model = sim.create_catalyst_model(0, volsize=(10, 10, 10))
        stack = sim.create_model_tilt_series(model)
        shifted = sim.misalign_stack(stack, tilt_shift=2)
        assert shifted.data.shape == (90, 10, 10)

    def test_misalign_stack_with_xonly(self):
        model = sim.create_catalyst_model(0, volsize=(10, 10, 10))
        stack = sim.create_model_tilt_series(model)
        shifted = sim.misalign_stack(stack, y_only=True)
        assert shifted.data.shape == (90, 10, 10)

    def test_misalign_stack_with_rotation(self):
        model = sim.create_catalyst_model(0, volsize=(10, 10, 10))
        stack = sim.create_model_tilt_series(model)
        shifted = sim.misalign_stack(stack, tilt_rotate=2)
        assert shifted.data.shape == (90, 10, 10)

    def test_add_noise_gaussian(self):
        model = sim.create_catalyst_model(0, volsize=(10, 10, 10))
        stack = sim.create_model_tilt_series(model)
        noisy = sim.add_noise(stack, "gaussian")
        assert noisy.data.shape == (90, 10, 10)

    def test_add_noise_poissanian(self):
        model = sim.create_catalyst_model(0, volsize=(10, 10, 10))
        stack = sim.create_model_tilt_series(model)
        noisy = sim.add_noise(stack, "poissonian")
        assert noisy.data.shape == (90, 10, 10)
