from tomotools import simulation as sim
import numpy as np


class TestModels:
    def test_catalyst_model(self):
        stack = sim.create_catalyst_model(0, volsize=[10, 10, 10])
        assert stack.data.shape == (10, 10, 10)

    def test_tilt_series_model(self):
        stack = sim.create_catalyst_model(0, volsize=[10, 10, 10])
        proj = sim.create_model_tilt_series(stack, np.arange(0, 15, 5))
        assert proj.data.shape == (3, 10, 10)

    def test_tilt_series_model_no_tilts(self):
        stack = sim.create_catalyst_model(0, volsize=[10, 10, 10])
        proj = sim.create_model_tilt_series(stack, None)
        assert proj.data.shape == (90, 10, 10)


class TestModifications:
    def test_misalign_stack(self):
        model = sim.create_catalyst_model(0, volsize=[10, 10, 10])
        stack = sim.create_model_tilt_series(model)
        shifted = sim.misalign_stack(stack)
        assert shifted.data.shape == (90, 10, 10)

    def test_misalign_stack_with_shift(self):
        model = sim.create_catalyst_model(0, volsize=[10, 10, 10])
        stack = sim.create_model_tilt_series(model)
        shifted = sim.misalign_stack(stack, tilt_shift=2)
        assert shifted.data.shape == (90, 10, 10)

    def test_misalign_stack_with_xonly(self):
        model = sim.create_catalyst_model(0, volsize=[10, 10, 10])
        stack = sim.create_model_tilt_series(model)
        shifted = sim.misalign_stack(stack, x_only=True)
        assert shifted.data.shape == (90, 10, 10)

    def test_misalign_stack_with_rotation(self):
        model = sim.create_catalyst_model(0, volsize=[10, 10, 10])
        stack = sim.create_model_tilt_series(model)
        shifted = sim.misalign_stack(stack, tilt_rotate=2)
        assert shifted.data.shape == (90, 10, 10)

    def test_add_noise(self):
        model = sim.create_catalyst_model(0, volsize=[10, 10, 10])
        stack = sim.create_model_tilt_series(model)
        noisy = sim.add_noise(stack)
        assert noisy.data.shape == (90, 10, 10)
