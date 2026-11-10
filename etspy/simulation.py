"""Simulation module for ETSpy package."""

from typing import Literal, NamedTuple, cast

import astra
import hyperspy.api as hs
import numpy as np
from hyperspy._signals.signal2d import Signal2D
from scipy import ndimage

from etspy import _format_choices as _fmt
from etspy import _get_literal_hint_values as _get_lit
from etspy.api import TomoStack
from etspy.recon import create_projector, resolve_cuda


def create_catalyst_model(
    nparticles: int = 15,
    particle_density: int = 255,
    support_density: int = 100,
    volsize: tuple[int, int, int] = (600, 600, 600),
    support_radius: int = 200,
    size_interval: tuple[int, int] = (5, 12),
) -> hs.signals.Signal2D:
    """
    Create a model data array that mimics a hetergeneous catalyst.

    Parameters
    ----------
    nparticles
        Number of particles to add
    particle_density
        Grayscale value to assign to the particles
    support_density
        Grayscale value to assign to the support material
    volsize
        X, Y, Z shape (in that order) of the volume
    support_radius
        Radius (in pixels) of the support material
    size_interval
        Lower and upper bounds (in that order) of the particle size

    Returns
    -------
    catalyst : :py:class:`~hyperspy.api.signals.Signal2D`
        Simulated model

    Group
    -----
    simulation

    Order
    -----
    1
    """
    volsize_np = np.array(volsize)
    center = np.array(volsize_np / 2, dtype=np.int32)

    catalyst = np.zeros(volsize_np, np.uint8)

    coords = np.zeros([nparticles, 4])
    for i in range(nparticles):
        size = 2 * np.random.randint(size_interval[0], size_interval[1])
        r = support_radius * 2
        overlap = False
        x = np.random.randint(0, volsize_np[0])  # initalize x and y before while loop
        y = np.random.randint(0, volsize_np[1])
        while r > support_radius or overlap:
            x = np.random.randint(0, volsize_np[0])
            y = np.random.randint(0, volsize_np[1])
            r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            distance = np.abs(coords[:, 0:2] - np.array([x, y]))
            overlap = np.min(distance) < size_interval[1]

        z_exact = np.int32(
            np.sqrt(support_radius**2 - (x - center[0]) ** 2 - (y - center[1]) ** 2)
            + center[2],
        )
        zmin = z_exact - np.int32(size / 2)
        zmax = z_exact + np.int32(size / 2)
        z_rand = np.random.randint(zmin, zmax)
        test = np.random.randint(-1, 1)
        z = center[2] - (z_rand - center[2]) if test < 0 else z_rand
        coords[i, :] = [x, y, z, size]

    xx, yy, zz = np.mgrid[: volsize_np[0], : volsize_np[1], : volsize_np[2]]

    support = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2

    catalyst[support < support_radius**2] = support_density

    for i in range(nparticles):
        x, y, z, particle_radius = coords[i, :]
        particle = (xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2
        catalyst[particle < particle_radius**2] = particle_density

    catalyst = hs.signals.Signal2D(catalyst)
    catalyst.axes_manager[0].name = "Z"
    catalyst.axes_manager[1].name = "X"
    catalyst.axes_manager[2].name = "Y"
    return catalyst


def create_cylinder_model(
    radius: int = 30,
    blur: bool = True,
    blur_sigma: float = 1.5,
    add_others: bool = False,
) -> hs.signals.Signal2D:
    """
    Create a model data array that mimics a needle shaped sample.

    Parameters
    ----------
    radius
        Radius of the cylinder to create
    blur
        If True, apply a Gaussian blur to the volume
    blur_sigma
        Sigma value for the Gaussiuan blur
    add_others
        If ``True``, add a second and third cylinder to the model near the periphery.
        This is useful for testing the effects of additional objects entering the
        tilt series field of view.

    Returns
    -------
    cylinder : :py:class:`~hyperspy.api.signals.Signal2D`
        Simulated cylinder object

    Group
    -----
    simulation

    Order
    -----
    2
    """
    vol_shape = np.array([400, 400, 400]) if add_others else np.array([200, 200, 200])

    cylinder = np.zeros(vol_shape, np.uint16)
    xx, yy = np.ogrid[: vol_shape[1], : vol_shape[2]]
    center_x, center_y, _ = vol_shape // 2

    # Create first cylinder
    cylinder1 = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2

    if not add_others:
        # Add the cylinder to the volume
        for i in range(vol_shape[2]):
            cylinder[:, :, i] = cylinder1

    else:
        # Create second cylinder
        radius_cylinder2 = 10
        center_x, center_y = [30, 30]
        cylinder2 = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius_cylinder2**2

        # Create third cylinder
        radius_cylinder3 = 15
        center_x, center_y = [370, 350]
        cylinder3 = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius_cylinder3**2

        # Add the cylinders to the volume
        low_thresh, mid_thresh, high_thresh = 150, 230, 270
        for i in range(vol_shape[2]):
            if i < low_thresh:
                cylinder[:, :, i] = 50 * cylinder1 + 10 * cylinder2
            elif i < high_thresh and i > mid_thresh:
                cylinder[:, :, i] = 50 * cylinder1 + 20 * cylinder3
            else:
                cylinder[:, :, i] = 50 * cylinder1

    if blur:
        cylinder = ndimage.gaussian_filter(cylinder, sigma=blur_sigma)

    cylinder = hs.signals.Signal2D(np.transpose(cylinder, [2, 0, 1]))
    cylinder.axes_manager[0].name = "X"
    cylinder.axes_manager[1].name = "Y"
    cylinder.axes_manager[2].name = "Z"
    return cylinder


def create_model_tilt_series(
    model: np.ndarray | hs.signals.Signal2D,
    angles: np.ndarray | None = None,
    cuda: bool | None = None,
) -> TomoStack:
    """
    Create a tilt series from a 3D volume.

    Parameters
    ----------
    model
        3D array or HyperSpy signal containing the model volume to project to
        a tilt series
    angles
        Projection angles for tilt series in degrees (optional). If ``None``,
        an evenly spaced range from 0 to 180 degrees will be used.
    cuda
        Whether or not to use CUDA-accelerated reconstruction algorithms. If
        ``None`` (the default), the decision to use CUDA will be left to
        :py:func:`astra.astra.use_cuda`.

    Returns
    -------
    model : TomoStack
        Tilt series of the model data

    Group
    -----
    simulation

    Order
    -----
    3
    """
    cuda = resolve_cuda(cuda)

    if angles is None:
        angles = np.arange(0, 180, 2)
    angles = cast("np.ndarray", angles)

    if isinstance(model, Signal2D):
        model = model.data

    xdim, zdim, ydim = model.shape
    ntilts = len(angles)

    proj_data = np.zeros([ntilts, ydim, xdim])
    vol_geom = astra.create_vol_geom([zdim, ydim])
    thetas = np.radians(angles)

    proj_geom = astra.create_proj_geom("parallel", 1.0, ydim, thetas)
    proj_id = create_projector(proj_geom, vol_geom, cuda=bool(cuda))

    try:
        for i in range(model.shape[0]):
            sino_id, proj_data[:, :, i] = astra.create_sino(model[i, :, :], proj_id)
            astra.data2d.delete(sino_id)
    finally:
        astra.projector.delete(proj_id)

    stack = TomoStack(proj_data, angles)
    return stack


def misalign_stack(
    stack: TomoStack,
    min_shift: int = -5,
    max_shift: int = 5,
    tilt_shift: int = 0,
    tilt_rotate: int = 0,
    y_only: bool = False,
    interp_order: int = 3,
) -> TomoStack:
    """
    Apply misalignment to a model tilt series.

    Parameters
    ----------
    stack
        TomoStack simluation
    min_shift
        Minimum amount of jitter to apply to the stack
    max_shift
        Maximum amount of jitter to apply to the stack
    tilt_shift
        Number of pixels by which to offset the tilt axis from the center, i.e. to
        shift the stack along the y-axis (perpendicular to the tilt axis)
    tilt_rotate
        Amount of rotation to apply to the stack
    y_only
        If True, limit the application of jitter to the y-direction only.
        Default is False
    interp_order
        The order of spline interpolation used by the :py:func:`scipy.ndimage.shift`
        or :py:func:`scipy.ndimage.rotate` function. The order must be in the range 0-5.

    Returns
    -------
    misaligned : TomoStack
        Misaligned copy of the input TomoStack

    Group
    -----
    simulation

    Order
    -----
    5
    """
    misaligned = stack.deepcopy()

    if tilt_shift != 0:
        misaligned.data = ndimage.shift(
            misaligned.data,
            shift=[0, tilt_shift, 0],
            order=interp_order,
        )
    if tilt_rotate != 0:
        misaligned.data = ndimage.rotate(
            misaligned.data,
            axes=(1, 2),
            angle=-tilt_rotate,
            order=interp_order,
            reshape=False,
        )

    if (min_shift != 0) or (max_shift != 0):
        jitter = np.random.uniform(min_shift, max_shift, size=(stack.data.shape[0], 2))
        for i in range(stack.data.shape[0]):
            if y_only:
                jitter[i, 1] = 0  # set the x jitter to 0

            misaligned.data[i, :, :] = ndimage.shift(
                misaligned.data[i, :, :],
                shift=[jitter[i, 0], jitter[i, 1]],
                order=interp_order,
            )
        misaligned.shifts = jitter
    return misaligned


def add_noise(
    stack: TomoStack,
    noise_type: Literal["gaussian", "poissonian", "shot"] = "gaussian",
    scale_factor: float = 0.2,
):
    """
    Apply noise to a model tilt series and return as a copy.

    Parameters
    ----------
    stack
        TomoStack simluation
    noise_type
        Type of noise. Must be ``"gaussian"`` or ``"poissonian"``/``"shot"``
    scale_factor
        Amount of noise to add

    Returns
    -------
    noisy : TomoStack
        Noisy copy of the input TomoStack

    Group
    -----
    simulation

    Order
    -----
    4
    """
    noisy = stack.deepcopy()

    if noise_type == "gaussian":
        noise = np.random.normal(
            stack.data.mean(),
            scale_factor * stack.data.mean(),
            stack.data.shape,
        )
        noisy.data = noisy.data + noise
        if noisy.data.min() < 0:
            noisy.data -= noisy.data.min()
        scale_factor = noisy.data.max() / stack.data.max()
        noisy.data = noisy.data / scale_factor

    elif noise_type in ["poissonian", "shot"]:
        noise = np.random.poisson(stack.data * scale_factor) / scale_factor
        noisy.data = noisy.data + noise

    else:
        msg = (
            f'Invalid noise type "{noise_type}". Must be one of '
            f"{_fmt(_get_lit(add_noise, 'noise_type'))}."
        )
        raise ValueError(msg)

    return noisy


class AlignmentTestSeries(NamedTuple):
    """
    A simulated misaligned tilt series with its known alignment.

    Returned by :py:func:`create_alignment_test_series`.

    Group
    -----
    simulation
    """

    misaligned: TomoStack
    """The misaligned (and optionally noisy) tilt series"""
    aligned: TomoStack
    """The noise-free tilt series without misalignment"""
    shifts: np.ndarray
    """Shifts ``(Y, X)`` of each projection, shape ``(ntilts, 2)``, that correct the
    misalignment (in the same order as :py:func:`etspy.align.apply_shifts`)"""
    tilt_axis: float
    """In-plane rotation (degrees) that corrects the tilt axis direction, applied
    after the shifts, about the image center (as in
    :py:func:`etspy.align.fourier_rigid_transform`)"""


# Features of the alignment test phantom: ellipsoids and boxes
_ELLIPSOID = 0
_BOX = 1


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly distributed random 3D rotation matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _alignment_phantom(
    rng: np.random.Generator,
    ny: int,
    nx: int,
    nparticles: int,
    nboxes: int,
    nplates: int,
    nrods: int,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Create the features of the alignment test phantom.

    Each feature is ``(kind, center, rotation, half_size, density)``, with
    coordinates ``(X, A, B)`` in pixels from the center of the volume: ``X`` along
    the tilt axis (the image x-axis) and ``(A, B)`` in the plane of each slice. All
    features lie within a radius of ``0.42 * ny`` of the tilt axis, so that they
    stay inside the field of view at all tilts.
    """
    scale = ny / 128
    features = []
    # weak support film, which also contributes smooth, low-frequency contrast
    support = np.array([0.48 * nx, 0.38 * ny, 0.12 * ny])
    features.append((_ELLIPSOID, np.zeros(3), np.eye(3), support, 0.01))

    def center(radius: float) -> np.ndarray:
        # a random position inside the support, keeping the feature in the field
        # of view
        while True:
            pos = rng.uniform(-1, 1, 3) * (support - [0, radius, 0])
            if np.hypot(pos[1], pos[2]) + radius < 0.42 * ny:
                return pos

    for _ in range(nparticles):
        # small dense particles (like gold fiducials), sharp in both x and y
        radius = rng.uniform(0.8, 2.5) * scale
        features.append(
            (_ELLIPSOID, center(radius), np.eye(3), np.full(3, radius), 1.0),
        )
    for i in range(nboxes):
        # boxes, half of them aligned with the axes (edges sharp along x and y)
        half = rng.uniform(2, 6, 3) * scale
        rotation = np.eye(3) if i % 2 == 0 else _random_rotation(rng)
        features.append(
            (_BOX, center(np.linalg.norm(half)), rotation, half, 0.3),
        )
    for i in range(nplates):
        # thin plates, normal to the tilt axis (sharp along x) or in the slice
        # plane (sharp along y at some tilts)
        half = np.array([rng.uniform(0.6, 1.2), *rng.uniform(4, 10, 2)]) * scale
        if i % 2 == 0:
            rotation = np.eye(3)
        else:
            angle = rng.uniform(0, np.pi)
            rotation = np.array(
                [
                    [0, 0, 1],
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                ],
            )
        features.append(
            (_BOX, center(np.linalg.norm(half)), rotation, half, 0.4),
        )
    for _ in range(nrods):
        # long thin rods at oblique angles
        half = np.array([rng.uniform(12, 25), 1.0, 1.0]) * scale
        features.append(
            (_BOX, center(half[0]), _random_rotation(rng), half, 0.4),
        )
    return features


def _line_integrals(
    feature: tuple[int, np.ndarray, np.ndarray, np.ndarray, float],
    x: np.ndarray,
    u: np.ndarray,
    theta: float,
) -> np.ndarray:
    """
    Calculate the line integrals through a phantom feature.

    The rays are parallel to ``(0, sin(theta), -cos(theta))`` and pass through
    ``(x, u cos(theta), u sin(theta))``, where ``u`` is the detector coordinate
    perpendicular to the tilt axis (the parallel beam geometry of ASTRA).
    """
    kind, center, rotation, half, density = feature
    origin = (
        np.stack(
            [
                x - center[0],
                u * np.cos(theta) - center[1],
                u * np.sin(theta) - center[2],
            ],
            axis=-1,
        )
        @ rotation
    )
    direction = np.array([0.0, np.sin(theta), -np.cos(theta)]) @ rotation
    if kind == _ELLIPSOID:
        o = origin / half
        v = direction / half
        a = v @ v
        b = 2 * (o @ v)
        c = np.sum(o**2, axis=-1) - 1
        return density * np.sqrt(np.maximum(b**2 - 4 * a * c, 0)) / a
    # box: intersect the ray with the three slabs |coordinate| <= half
    t_min = np.full(x.shape, -np.inf)
    t_max = np.full(x.shape, np.inf)
    for k in range(3):
        if abs(direction[k]) > 1e-12:  # noqa: PLR2004
            t1 = (-half[k] - origin[..., k]) / direction[k]
            t2 = (half[k] - origin[..., k]) / direction[k]
            t_min = np.maximum(t_min, np.minimum(t1, t2))
            t_max = np.minimum(t_max, np.maximum(t1, t2))
        else:
            outside = np.abs(origin[..., k]) > half[k]
            t_min[outside] = np.inf
    return density * np.maximum(t_max - t_min, 0)


def _render_projections(
    features: list,
    tilts: np.ndarray,
    ny: int,
    nx: int,
    shifts: np.ndarray,
    tilt_axis: float,
    supersample: int,
) -> np.ndarray:
    """
    Render the projections of the phantom, with a known misalignment.

    The content of the ideal projection at ``p`` appears in the rendered image at
    ``q``, where ``p = R (q + s - c) + c``: correcting the image by the shift ``s``
    and then rotating it by ``tilt_axis`` about ``c`` recovers the ideal projection.
    Each pixel is the average of ``supersample**2`` samples over its area.
    """
    ss = supersample
    # coordinates of the samples from the image center (pixel i spans
    # [i - n/2, i + 1 - n/2]); the rotation center is pixel (nx/2, ny/2), as in
    # fourier_rigid_transform
    x_samples = (np.arange(nx * ss) + 0.5) / ss - nx / 2
    u_samples = (np.arange(ny * ss) + 0.5) / ss - ny / 2
    rot_center = np.array([0.5, 0.5])
    image_center = np.array([nx / 2, ny / 2])
    phi = np.deg2rad(tilt_axis)
    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    out = np.zeros((len(tilts), ny, nx))
    for i, theta in enumerate(np.deg2rad(tilts)):
        sampled = np.zeros((ny * ss, nx * ss))
        for feature in features:
            # bounding box of the feature's ideal projection, mapped to the
            # rendered image
            _, center, _, half, _ = feature
            radius = np.linalg.norm(half) if feature[0] == _BOX else half.max()
            u_center = center[1] * np.cos(theta) + center[2] * np.sin(theta)
            corners = np.array(
                [
                    [center[0] + dx * radius, u_center + du * radius]
                    for dx in (-1, 1)
                    for du in (-1, 1)
                ],
            )
            corners = (corners - rot_center) @ rotation + rot_center - shifts[i, ::-1]
            lo = np.floor((corners.min(0) + image_center - 1) * ss).astype(int)
            hi = np.ceil((corners.max(0) + image_center + 1) * ss).astype(int)
            x0, u0 = np.maximum(lo, 0)
            x1, u1 = np.minimum(hi, [nx * ss, ny * ss])
            if x1 <= x0 or u1 <= u0:
                continue
            qx, qu = np.meshgrid(x_samples[x0:x1], u_samples[u0:u1])
            q = np.stack([qx, qu], axis=-1) + shifts[i, ::-1] - rot_center
            p = q @ rotation.T + rot_center
            sampled[u0:u1, x0:x1] += _line_integrals(
                feature,
                p[..., 0],
                p[..., 1],
                theta,
            )
        out[i] = sampled.reshape(ny, ss, nx, ss).mean(axis=(1, 3))
    return out


def create_alignment_test_series(  # noqa: PLR0913
    shape: tuple[int, int] = (128, 128),
    tilts: np.ndarray | None = None,
    max_shift: float = 3.0,
    drift: float = 2.0,
    tilt_axis: float = 1.0,
    background: float = 0.2,
    dose: float | None = None,
    blur: float = 0.0,
    supersample: int = 4,
    features: tuple[int, int, int, int] = (30, 12, 10, 6),
    seed: int = 0,
) -> AlignmentTestSeries:
    """
    Create a misaligned tilt series with a known alignment, for testing alignment.

    The phantom is a weak support film carrying small dense particles, boxes, thin
    plates and rods, with edges that are sharp along both the x-axis (the tilt
    axis) and the y-axis, to test sub-pixel alignment in both directions. The
    projections are calculated analytically (exact line integrals through
    ellipsoids and boxes, averaged over the area of each pixel), and the
    misalignment is applied while rendering them, so that no interpolation is
    involved and the correcting alignment is known exactly.

    Parameters
    ----------
    shape
        Image shape ``(ny, nx)``; the tilt axis is along x
    tilts
        Tilt angles in degrees (default: -70 to 70 degrees in 2 degree steps)
    max_shift
        Each projection is shifted by a random jitter, uniform in
        ``[-max_shift, max_shift]`` pixels in both x and y, plus a drift
    drift
        Amplitude (pixels) of a smooth drift of the projections across the tilt
        series, in both x and y
    tilt_axis
        Rotation (degrees) of the tilt axis away from the x-axis
    background
        Constant background (vacuum) level added to the projections, relative to
        their maximum (which is scaled to 1)
    dose
        If given, Poisson noise is added for this number of counts per pixel at an
        intensity of 1
    blur
        Standard deviation (pixels) of a Gaussian point spread function applied to
        the projections (default: none)
    supersample
        Number of samples per pixel along each axis used to integrate over the
        pixel area
    features
        Number of particles, boxes, plates and rods in the phantom
    seed
        Seed of the random number generator, for a reproducible series

    Returns
    -------
    AlignmentTestSeries
        The misaligned series, the aligned (noise-free) series, and the shifts and
        tilt axis rotation that correct the misalignment

    Group
    -----
    simulation

    Order
    -----
    6
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    if tilts is None:
        tilts = np.arange(-70, 71, 2)
    tilts = np.asarray(tilts, dtype=float)
    ntilts = len(tilts)
    phantom = _alignment_phantom(rng, ny, nx, *features)

    # the correcting shifts: a smooth drift (a few low-order cosines) plus jitter
    position = np.linspace(0, 1, ntilts)
    shifts = rng.uniform(-max_shift, max_shift, (ntilts, 2))
    for k in range(1, 4):
        amplitude = rng.normal(0, drift / k, 2)
        shifts += (
            np.cos(np.pi * k * position + rng.uniform(0, 2 * np.pi))[
                :,
                np.newaxis,
            ]
            * amplitude
        )

    ideal = _render_projections(
        phantom,
        tilts,
        ny,
        nx,
        np.zeros((ntilts, 2)),
        0.0,
        supersample,
    )
    misaligned = _render_projections(
        phantom,
        tilts,
        ny,
        nx,
        shifts,
        tilt_axis,
        supersample,
    )
    if blur > 0:
        ideal = ndimage.gaussian_filter(ideal, (0, blur, blur))
        misaligned = ndimage.gaussian_filter(misaligned, (0, blur, blur))
    scale = 1 / ideal.max()
    ideal = ideal * scale + background
    misaligned = misaligned * scale + background
    if dose is not None:
        misaligned = rng.poisson(misaligned * dose) / dose

    return AlignmentTestSeries(
        misaligned=TomoStack(misaligned.astype(np.float32), tilts),
        aligned=TomoStack(ideal.astype(np.float32), tilts),
        shifts=shifts,
        tilt_axis=tilt_axis,
    )
