"""Simulation module for ETSpy package."""

from typing import Literal, Optional, Tuple, Union, cast

import astra
import hyperspy.api as hs
import numpy as np
from hyperspy._signals.signal2d import Signal2D
from scipy import ndimage

from etspy import _format_choices as _fmt
from etspy import _get_literal_hint_values as _get_lit
from etspy.api import TomoStack


def create_catalyst_model(
    nparticles: int = 15,
    particle_density: int = 255,
    support_density: int = 100,
    volsize: Tuple[int, int, int] = (600, 600, 600),
    support_radius: int = 200,
    size_interval: Tuple[int, int] = (5, 12),
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
    model: Union[np.ndarray, hs.signals.Signal2D],
    angles: Optional[np.ndarray] = None,
    cuda: Optional[bool] = None,
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
    if cuda is None:
        cuda = astra.use_cuda()

    if angles is None:
        angles = np.arange(0, 180, 2)
    angles = cast(np.ndarray, angles)

    if isinstance(model, Signal2D):
        model = model.data

    xdim, zdim, ydim = model.shape
    ntilts = len(angles)

    proj_data = np.zeros([ntilts, ydim, xdim])
    vol_geom = astra.create_vol_geom([zdim, ydim])
    thetas = np.radians(angles)

    proj_geom = astra.create_proj_geom("parallel", 1.0, ydim, thetas)
    if cuda is False:
        proj_id = astra.create_projector("linear", proj_geom, vol_geom)
    else:
        proj_id = astra.create_projector("cuda", proj_geom, vol_geom) # coverage: nocuda

    for i in range(model.shape[0]):
        sino_id, proj_data[:, :, i] = astra.create_sino(model[i, :, :], proj_id)

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
        Number of pixels by which to offset the tilt axis from the center
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
            shift=[0, 0, tilt_shift],
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
