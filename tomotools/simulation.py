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
from tomotools.io import convert_to_tomo_stack
import hyperspy.api as hs


def create_catalyst_model(nparticles=15, particle_density=255,
                          support_density=100, volsize=[600, 600, 600],
                          support_radius=200, size_interval=[5, 12]):
    """
    Create a model tilt series that mimics a hetergeneous catalyst.

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
    catalyst : TomoStack object
        Simulated tilt series

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
            r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            distance = np.abs(coords[:, 0:2] - np.array([x, y]))
            if np.min(distance) < size_interval[1]:
                overlap = True
            else:
                overlap = False

        z_exact = np.int32(np.sqrt(support_radius**2 - (x - center[0])**2 - (y - center[1])**2) + center[2])
        zmin = z_exact - np.int32(size / 2)
        zmax = z_exact + np.int32(size / 2)
        z_rand = np.random.randint(zmin, zmax)
        test = np.random.randint(-1, 1)
        if test < 0:
            z = center[2] - (z_rand - center[2])
        else:
            z = z_rand
        coords[i, :] = [x, y, z, size]

    xx, yy, zz = np.mgrid[:volsize[0], :volsize[1], :volsize[2]]

    support = (xx - center[0]) ** 2 +\
        (yy - center[1]) ** 2 + (zz - center[2]) ** 2

    catalyst[support < support_radius**2] = support_density

    for i in range(0, nparticles):
        x, y, z, particle_radius = coords[i, :]
        particle = (xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2
        catalyst[particle < particle_radius**2] = particle_density

    catalyst = hs.signals.Signal2D(catalyst)
    catalyst.axes_manager[0].name = 'Z'
    catalyst.axes_manager[1].name = 'X'
    catalyst.axes_manager[2].name = 'Y'
    return catalyst


def create_model_tilt_series(model, angles=None):
    """
    Create a tilt series from a 3D volume.

    Args
    ----------
    model : NumPy array
        3D array containing the model volume to project to a tilt series
    angles : NumPy array
        Projection angles for tilt series

    Returns
    ----------
    model : TomoStack object
        Tilt series of the model data

    """
    if type(angles) is not np.ndarray:
        angles = np.arange(0, 180, 2)

    if type(model) is hs.signals.Signal2D:
        model = model.data

    xdim = model.shape[2]
    ydim = model.shape[1]
    thickness = model.shape[0]

    proj_data = np.zeros([len(angles), ydim, xdim])
    vol_geom = astra.create_vol_geom(thickness, xdim, ydim)
    tilts = angles * np.pi / 180
    proj_geom = astra.create_proj_geom('parallel', 1, xdim, tilts)
    proj_id = astra.create_projector('strip', proj_geom, vol_geom)

    for i in range(0, model.shape[1]):
        sino_id, proj_data[:, i, :] = astra.create_sino(model[:, i, :],
                                                        proj_id)

    stack = convert_to_tomo_stack(proj_data)
    stack.axes_manager[0].offset = angles[0]
    stack.axes_manager[0].scale = np.abs(angles[1] - angles[0])
    return stack


def misalign_stack(stack, min_shift=-5, max_shift=5, tilt_shift=0,
                   tilt_rotate=0, x_only=False):
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
    x_only : bool
        If True, limit the application of jitter to the x-direction only.
        Default is False

    Returns
    ----------
    misaligned : TomoStack object
        Misaligned copy of the input TomoStack

    """
    misaligned = stack.deepcopy()

    if tilt_shift != 0:
        misaligned.data = ndimage.shift(misaligned.data,
                                        shift=[0, 0, tilt_shift],
                                        order=0)
    if tilt_rotate != 0:
        misaligned.data = ndimage.rotate(misaligned.data, axes=(1, 2),
                                         angle=-tilt_rotate, order=0,
                                         reshape=False)

    if (min_shift != 0) or (max_shift != 0):
        jitter = np.random.uniform(min_shift,
                                   max_shift,
                                   size=(stack.data.shape[0], 2))
        for i in range(stack.data.shape[0]):
            if x_only:
                jitter[i, 0] = 0

            misaligned.data[i, :, :] =\
                ndimage.shift(misaligned.data[i, :, :],
                              shift=[jitter[i, 0], jitter[i, 1]],
                              order=0)
    return misaligned


def add_gaussian_noise(stack, factor=0.2):
    """
    Apply misalignment to a model tilt series.

    Args
    ----------
    stack : TomoStack object
        TomoStack simluation
    factor : float
        Amount of noise to add

    Returns
    ----------
    noisy : TomoStack object
        Noisy copy of the input TomoStack

    """
    noisy = stack.deepcopy()
    noise = np.random.normal(stack.data.mean(),
                             factor * stack.data.mean(),
                             stack.data.shape)
    noisy.data = noisy.data + noise
    if noisy.data.min() < 0:
        noisy.data -= noisy.data.min()
    scale_factor = noisy.data.max() / stack.data.max()
    noisy.data = noisy.data / scale_factor
    return noisy
