# -*- coding: utf-8 -*-
#
# This file is part of TomoTools

"""
Simulation module for TomoTools package.

@author: Andrew Herzing
"""
import numpy as np
from scipy import ndimage
import astra
from tomotools.io import create_stack
import hyperspy.api as hs


def create_catalyst_model(nparticles=15, particle_density=255, support_density=100, volsize=[600, 600, 600], support_radius=200, size_interval=[5, 12]):
    """
    Create a model data array that mimics a hetergeneous catalyst.

    Args
    ----------
    nparticles : int
        Number of particles to add
    particle_density : int
        Grayscale value to assign to the particles
    support_density : int
        Grayscale value to assign to the support
    volsize : list
        X, Y, Z shape of the volume
    support_radius : int
        Radius (in pixels) of the support
    size_interval : list
        Upper and lower bounds of the particle size

    Returns
    ----------
    catalyst : Hyperspy Signal2D
        Simulated model

    """
    volsize = np.array(volsize)
    center = np.int32(volsize / 2)
    size_interval = [5, 12]

    catalyst = np.zeros(volsize, np.uint8)

    coords = np.zeros([nparticles, 4])
    for i in range(0, nparticles):
        size = 2 * np.random.randint(size_interval[0], size_interval[1])
        r = support_radius * 2
        overlap = False
        while r > support_radius or overlap:
            x = np.random.randint(0, volsize[0])
            y = np.random.randint(0, volsize[1])
            r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            distance = np.abs(coords[:, 0:2] - np.array([x, y]))
            if np.min(distance) < size_interval[1]:
                overlap = True
            else:
                overlap = False

        z_exact = np.int32(
            np.sqrt(support_radius**2 - (x - center[0]) ** 2 - (y - center[1]) ** 2) + center[2]
        )
        zmin = z_exact - np.int32(size / 2)
        zmax = z_exact + np.int32(size / 2)
        z_rand = np.random.randint(zmin, zmax)
        test = np.random.randint(-1, 1)
        if test < 0:
            z = center[2] - (z_rand - center[2])
        else:
            z = z_rand
        coords[i, :] = [x, y, z, size]

    xx, yy, zz = np.mgrid[: volsize[0], : volsize[1], : volsize[2]]

    support = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2

    catalyst[support < support_radius**2] = support_density

    for i in range(0, nparticles):
        x, y, z, particle_radius = coords[i, :]
        particle = (xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2
        catalyst[particle < particle_radius**2] = particle_density

    catalyst = hs.signals.Signal2D(catalyst)
    catalyst.axes_manager[0].name = "Z"
    catalyst.axes_manager[1].name = "X"
    catalyst.axes_manager[2].name = "Y"
    return catalyst


def create_cylinder_model(radius=30, blur=True, blur_sigma=1.5, add_others=False):
    """
    Create a model data array that mimics a needle shaped sample.

    Args
    ----------
    vol_size : int
        Size of the volume for the model
    radius : int
        Radius of the cylinder to create
    blur : bool
        If True, apply a Gaussian blur to the volume
    blur_sigma : float
        Sigma value for the Gaussiuan blur
    Returns
    ----------
    cylinder : Signal2D
        Simulated cylinder object

    """
    if add_others:
        vol_shape = np.array([400, 400, 400])
    else:
        vol_shape = np.array([200, 200, 200])

    cylinder = np.zeros(vol_shape, np.uint16)
    xx, yy = np.ogrid[:vol_shape[1], :vol_shape[2]]
    center_x, center_y, _ = vol_shape // 2

    # Create first cylinder
    cylinder1 = (xx - center_x)**2 + (yy - center_y)**2 <= radius**2

    if not add_others:
        # Add the cylinder to the volume
        for i in range(vol_shape[2]):
            cylinder[:, :, i] = cylinder1

    else:
        # Create second cylinder
        radius_cylinder2 = 10
        center_x, center_y = [30, 30]
        cylinder2 = (xx - center_x)**2 + (yy - center_y)**2 <= radius_cylinder2**2

        # Create third cylinder
        radius_cylinder3 = 15
        center_x, center_y = [370, 350]
        cylinder3 = (xx - center_x)**2 + (yy - center_y)**2 <= radius_cylinder3**2

        # Add the cylinders to the volume
        for i in range(vol_shape[2]):
            if i < 150:
                cylinder[:, :, i] = 50 * cylinder1 + 10 * cylinder2
            elif i < 270 and i > 230:
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


def create_model_tilt_series(model, angles=None, cuda=None):
    """
    Create a tilt series from a 3D volume.

    Args
    ----------
    model : NumPy array or Hyperspy Signal2D
        3D array or signal containing the model volume to project to a tilt series
    angles : NumPy array
        Projection angles for tilt series in degrees

    Returns
    ----------
    model : TomoStack object
        Tilt series of the model data

    """
    if cuda is None:
        cuda = astra.use_cuda()

    if type(angles) is not np.ndarray:
        angles = np.arange(0, 180, 2)

    if type(model) is hs.signals.Signal2D:
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
        proj_id = astra.create_projector("cuda", proj_geom, vol_geom)

    for i in range(0, model.shape[0]):
        sino_id, proj_data[:, :, i] = astra.create_sino(model[i, :, :], proj_id)

    stack = create_stack(proj_data, angles)
    return stack


def misalign_stack(stack, min_shift=-5, max_shift=5, tilt_shift=0, tilt_rotate=0, y_only=False, interp_order=3):
    """
    Apply misalignment to a model tilt series.

    Args
    ----------
    stack : TomoStack object
        TomoStack simluation
    min_shift : int
        Minimum amount of jitter to apply to the stack
    max_shift : int
        Maximum amount of jitter to apply to the stack
    tilt_shift : int
        Number of pixels by which to offset the tilt axis from the center
    tilt_rotate : int
        Amount of rotation to apply to the stack
    y_only : bool
        If True, limit the application of jitter to the x-direction only.
        Default is False

    Returns
    ----------
    misaligned : TomoStack object
        Misaligned copy of the input TomoStack

    """
    misaligned = stack.deepcopy()

    if tilt_shift != 0:
        misaligned.data = ndimage.shift(
            misaligned.data, shift=[0, 0, tilt_shift], order=interp_order
        )
    if tilt_rotate != 0:
        misaligned.data = ndimage.rotate(
            misaligned.data, axes=(1, 2), angle=-tilt_rotate, order=interp_order, reshape=False
        )

    if (min_shift != 0) or (max_shift != 0):
        jitter = np.random.uniform(min_shift, max_shift, size=(stack.data.shape[0], 2))
        for i in range(stack.data.shape[0]):
            if y_only:
                jitter[i, 1] = 0

            misaligned.data[i, :, :] = ndimage.shift(misaligned.data[i, :, :], shift=[jitter[i, 0], jitter[i, 1]], order=interp_order)
    misaligned.metadata.Tomography.shifts = jitter
    return misaligned


def add_noise(stack, noise_type="gaussian", scale_factor=0.2):
    """
    Apply misalignment to a model tilt series.

    Args
    ----------
    stack : TomoStack object
        TomoStack simluation
    noise_type : str
        Type of noise. Must be gaussian or poissonian/shot
    factor : float
        Amount of noise to add

    Returns
    ----------
    noisy : TomoStack object
        Noisy copy of the input TomoStack

    """
    noisy = stack.deepcopy()

    if noise_type == "gaussian":
        noise = np.random.normal(
            stack.data.mean(), scale_factor * stack.data.mean(), stack.data.shape
        )
        noisy.data = noisy.data + noise
        if noisy.data.min() < 0:
            noisy.data -= noisy.data.min()
        scale_factor = noisy.data.max() / stack.data.max()
        noisy.data = noisy.data / scale_factor

    elif noise_type in ["poissonian", "shot"]:
        noise = np.random.poisson(stack.data * scale_factor) / scale_factor
        noisy.data = noisy.data + noise

    return noisy
