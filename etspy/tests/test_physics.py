"""Physics regression tests on synthetic phantoms with known ground truth."""

import numpy as np
import pytest
from scipy import ndimage

from etspy import simulation
from etspy.base import TomoStack
from etspy.tests.phantoms import (
    TILTS,
    make_phantom,
    ncc,
    project,
    shift_images,
)

MIN_REC_NCC = 0.9
MIN_HANDEDNESS_MARGIN = 0.5
MIN_REPROJ_NCC = 0.99
JITTER_MAX = 4.0
SUBPIXEL_TOL = 0.3
REGISTRATION_TOL = 0.5
COM_TOL = 1.0
AXIS_SHIFT = 5.0
AXIS_ROTATION = 3.0
ROTATION_TOL = 0.5
MAXIMAGE_SHIFT = 4.0
MAXIMAGE_SHIFT_TOL = 1.0
PARTICLE_DENSITY = 255
TILTS_FULL = np.arange(-90, 90, 2.0)


@pytest.fixture(scope="module")
def phantom():
    """Default blob phantom."""
    return make_phantom()


@pytest.fixture(scope="module")
def tilt_series(phantom):
    """Perfectly aligned tilt series of the default phantom."""
    return project(phantom)


@pytest.fixture(scope="module")
def jitter():
    """Known random (dy, dx) misalignment for each projection."""
    rng = np.random.default_rng(1)
    return rng.uniform(-JITTER_MAX, JITTER_MAX, (len(TILTS), 2))


@pytest.fixture(scope="module")
def jittered(tilt_series, jitter):
    """Tilt series misaligned by the known jitter."""
    return shift_images(tilt_series, jitter)


def _registration_rms(stack: TomoStack, truth: np.ndarray) -> np.ndarray:
    """RMS (y, x) error of applied shifts vs. truth; global offset is unobservable."""
    err = stack.shifts.data + truth
    err -= err.mean(0)
    return np.sqrt((err**2).mean(0))


@pytest.fixture(scope="module")
def narrow_phantom():
    """Narrow phantom for fast reconstruction checks."""
    return make_phantom(nx=16)


@pytest.fixture(scope="module")
def full_range_series(narrow_phantom):
    """Full 180 degree tilt series of the narrow phantom."""
    return project(narrow_phantom, TILTS_FULL)


class TestReconstructionGeometry:
    """Test reconstruction against the phantom it was projected from."""

    def test_handedness(self, narrow_phantom, full_range_series):
        rec = full_range_series.reconstruct(
            "FBP",
            cuda=False,
            thickness=narrow_phantom.shape[1],
            show_progressbar=False,
        ).data
        correct = ncc(rec, narrow_phantom)
        mirrored = ncc(rec, narrow_phantom[:, ::-1])
        assert correct > MIN_REC_NCC
        assert correct - mirrored > MIN_HANDEDNESS_MARGIN

    def test_reprojection(self, full_range_series):
        rec = full_range_series.reconstruct("FBP", cuda=False, show_progressbar=False)
        reproj = rec.forward_project(TILTS_FULL, cuda=False)
        assert ncc(reproj.data, full_range_series.data) > MIN_REPROJ_NCC


class TestTranslationalRegistration:
    """Test recovery of a known per-projection jitter."""

    def test_pc_x(self, jittered, jitter):
        out = jittered.stack_register("PC")
        assert _registration_rms(out, jitter)[1] < REGISTRATION_TOL

    def test_com_cl(self, jittered, jitter):
        rms = _registration_rms(jittered.stack_register("COM-CL"), jitter)
        assert rms[1] < SUBPIXEL_TOL
        assert rms[0] < COM_TOL

    def test_com_narrow_image(self, jittered, jitter):
        rms = _registration_rms(jittered.stack_register("COM"), jitter)
        assert np.all(rms < REGISTRATION_TOL)


def _misaligned_axis(nx, ny, shift, rotation):
    stack = project(make_phantom(nx=nx, ny=ny))
    stack.data = ndimage.shift(stack.data, [0, shift, 0], order=3)
    if rotation:
        stack.data = ndimage.rotate(
            stack.data,
            rotation,
            axes=(1, 2),
            reshape=False,
            order=3,
        )
    return stack


class TestTiltAxisAlignment:
    """Test recovery of a known tilt-axis offset and rotation."""

    @pytest.mark.parametrize(("nx", "ny"), [(96, 96), (48, 96), (128, 64)])
    def test_com_offset(self, nx, ny):
        ali = _misaligned_axis(nx, ny, AXIS_SHIFT, 0).tilt_align("CoM")
        assert abs(ali.metadata.Tomography.yshift - AXIS_SHIFT) < SUBPIXEL_TOL

    def test_maximage_shift(self):
        stack = _misaligned_axis(48, 96, MAXIMAGE_SHIFT, 0)
        ali = stack.tilt_align("MaxImage", also_shift=True, shift_limit=6)
        yshift = ali.metadata.Tomography.yshift
        assert abs(yshift - MAXIMAGE_SHIFT) <= MAXIMAGE_SHIFT_TOL


class TestSimulationPhysics:
    """Test that simulated misalignment and noise follow the intended physics."""

    def test_tilt_shift_is_perpendicular(self, tilt_series):
        shifted = simulation.misalign_stack(
            tilt_series,
            min_shift=0,
            max_shift=0,
            tilt_shift=int(AXIS_SHIFT),
        )
        before = np.array(ndimage.center_of_mass(tilt_series.data[35]))
        after = np.array(ndimage.center_of_mass(shifted.data[35]))
        dy, dx = after - before
        assert abs(dy - AXIS_SHIFT) < SUBPIXEL_TOL
        assert abs(dx) < SUBPIXEL_TOL

    @pytest.mark.parametrize(
        "noise_type",
        [
            pytest.param("gaussian"),
            pytest.param("poissonian"),
        ],
    )
    def test_noise_preserves_mean(self, tilt_series, noise_type):
        np.random.seed(0)
        noisy = simulation.add_noise(tilt_series, noise_type, 0.2)
        assert noisy.data.mean() / tilt_series.data.mean() == pytest.approx(1, abs=0.03)

    def test_particle_size(self):
        np.random.seed(0)
        size_interval = (3, 5)
        model = simulation.create_catalyst_model(
            1,
            volsize=(64, 64, 64),
            support_radius=20,
            size_interval=size_interval,
        )
        max_voxels = 4 / 3 * np.pi * size_interval[1] ** 3
        assert (model.data == PARTICLE_DENSITY).sum() < max_voxels

    def test_misalign_reproducible(self, tilt_series):
        a = simulation.misalign_stack(tilt_series, rng=np.random.default_rng(0))
        b = simulation.misalign_stack(tilt_series, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(a.data, b.data)
