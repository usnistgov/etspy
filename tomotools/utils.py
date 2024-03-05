# -*- coding: utf-8 -*-
#
# This file is part of TomoTools

"""
Utility module for TomoTools package.

@author: Andrew Herzing
"""

import numpy as np
from tomotools.io import convert_to_tomo_stack
import logging
import tqdm
from scipy import ndimage
from tomotools.align import calculate_shifts_stackreg
from pystackreg import StackReg
from multiprocessing import Pool


def multiaverage(stack, nframes, ny, nx):
    """
    Register a multi-frame series collected by SerialEM.

    Parameters
    ----------
    stack : NumPy array
        Array of shape [nframes, ny, nx].
    nframes : int
        Number of frames per tilt.
    ny : int
        Pixels in y-dimension.
    nx : int
        Pixels in x-dimension.

    Returns
    ----------
    average : NumPy array
        Average of all frames at given tilt
    """
    def _calc_sr_shifts(stack):
        sr = StackReg(StackReg.TRANSLATION)
        shifts = sr.register_stack(stack, reference='previous')
        shifts = -np.array([i[0:2, 2][::-1] for i in shifts])
        return shifts

    shifted = np.zeros([nframes, ny, nx])
    shifts = _calc_sr_shifts(stack)
    for k in range(0, nframes):
        shifted[k, :, :] = ndimage.shift(stack[k, :, :], shift=[shifts[k, 0], shifts[k, 1]])
    average = shifted.mean(0)
    return average


def register_serialem_stack(stack, ncpus=1):
    """
    Register a multi-frame series collected by SerialEM.

    Parameters
    ----------
    stack : Hyperspy Signal2D
        Signal of shape [ntilts, nframes, ny, nx].

    Returns
    ----------
    reg : TomoStack object
        Result of aligning and averaging frames at each tilt with shape [ntilts, ny, nx]

    """
    align_logger = logging.getLogger("tomotools.align")
    log_level = align_logger.getEffectiveLevel()
    align_logger.setLevel(logging.ERROR)
    ntilts, nframes, ny, nx = stack.data.shape

    if ncpus == 1:
        reg = np.zeros([ntilts, ny, nx], stack.data.dtype)
        for i in tqdm.tqdm(range(0, ntilts)):
            shifted = np.zeros([nframes, ny, nx])
            shifts = calculate_shifts_stackreg(stack.inav[:, i])
            for k in range(0, nframes):
                shifted[k, :, :] = ndimage.shift(stack.data[i, k, :, :], shift=[shifts[k, 0], shifts[k, 1]])
            reg[i, :, :] = shifted.mean(0)
    else:
        with Pool(ncpus) as pool:
            reg = pool.starmap(multiaverage,
                               [(stack.inav[:, i].data, nframes, ny, nx) for i in range(0, ntilts)])
        reg = np.array(reg)

    reg = convert_to_tomo_stack(reg)
    reg.axes_manager[0].scale = stack.axes_manager[1].scale
    reg.axes_manager[0].offset = stack.axes_manager[1].offset
    reg.axes_manager[0].units = stack.axes_manager[1].units

    reg.axes_manager[1].scale = stack.axes_manager[2].scale
    reg.axes_manager[1].offset = stack.axes_manager[2].offset
    reg.axes_manager[1].units = stack.axes_manager[2].units

    reg.axes_manager[2].scale = stack.axes_manager[3].scale
    reg.axes_manager[2].offset = stack.axes_manager[3].offset
    reg.axes_manager[2].units = stack.axes_manager[3].units

    if stack.metadata.has_item("Acquisition_instrument"):
        reg.metadata.Acquisition_instrument = stack.metadata.Acquisition_instrument
    if stack.metadata.has_item("Tomography"):
        reg.metadata.Tomography = stack.metadata.Tomography
    align_logger.setLevel(log_level)
    return reg


def weight_stack(stack, accuracy='medium'):
    """
    Apply a weighting window to a stack along the direction perpendicular to the tilt axis.

    This weighting is useful for reducing the effects of mass introduced at the edges of as stack when
    determining alignments based on the center of mass.  As described in:

            T. Sanders. Physically motivated global alignment method for electron
            tomography, Advanced Structural and Chemical Imaging vol. 1 (2015) pp 1-11.
            https://doi.org/10.1186/s40679-015-0005-7

    Parameters
    ----------
    stack : TomoStack
        Stack to be weighted.

    accuracy : string
        Level of accuracy for determining the weighting.  Acceptable values are 'low', 'medium', and 'high'.

    Returns
    ----------
    reg : TomoStack object
        Result of aligning and averaging frames at each tilt with shape [ntilts, ny, nx]

    """
    stackw = stack.deepcopy()

    [ntilts, ny, nx] = stack.data.shape
    alpha = np.sum(stack.data, (1, 2)).min()
    beta = np.sum(stack.data, (1, 2)).argmin()
    v = np.arange(ntilts)
    v[beta] = 0

    wg = np.zeros([ny, nx])

    if accuracy.lower() == 'low':
        num = 800
        delta = .025
    elif accuracy.lower() == 'medium':
        num = 2000
        delta = .01
    elif accuracy.lower() == 'high':
        num = 20000
        delta = .001
    else:
        raise ValueError("Unknown accuracy level.  Must be 'low', 'medium', or 'high'.")

    r = np.arange(1, ny + 1)
    r = 2 / (ny - 1) * (r - 1) - 1
    r = np.cos(np.pi * r**2) / 2 + 1 / 2
    s = np.zeros(ntilts)
    for p in range(1, int(num / 10) + 1):
        rp = r**(p * delta * 10)
        for x in range(0, nx):
            wg[:, x] = rp
        for i in range(0, ntilts):
            if v[i]:
                if np.sum(stack.data[i, :, :] * wg) < alpha:
                    v[i] = 0
                    s[i] = (p - 1) * 10
        if v.sum() == 0:
            break
    for i in range(0, ntilts):
        if v[i]:
            s[i] = (p - 1) * 10

    v = np.arange(1, ntilts + 1)
    v[beta] = 0
    for j in range(0, ntilts):
        if j != beta:
            for p in range(1, 10):
                rp = r**((p + s[j]) * delta)
                for x in range(0, nx):
                    wg[:, x] = rp
                    if np.sum(stack.data[i, :, :] * wg) < alpha:
                        s[j] = p + s[j]
                        v[i] = 0
                        break
    for i in range(0, ntilts):
        if v[i]:
            s[i] = s[i] + 10

    for i in range(0, ntilts):
        for x in range(0, nx):
            wg[:, x] = r**(s[i] * delta)
        stackw.data[i, :, :] = stack.data[i, :, :] * wg
    return stackw


def calc_EST_angles(N):
    """
    Caculate angles used for equally sloped tomography (EST).

    See:
            J. Miao, F. Forster, and O. Levi. Equally sloped tomography with oversampling reconstruction.
            Phys. Rev. B, 72 (2005) 052103.
            https://doi.org/10.1103/PhysRevB.72.052103

    Parameters
    ----------
    N : integer
        Number of points in scan.

    Returns
    ----------
    angles : Numpy array
        Angles in degrees for equally sloped tomography.

    """
    if np.mod(N, 2) != 0:
        raise ValueError("N must be an even number")

    angles = np.zeros(2 * N)

    n = np.arange(N / 2 + 1, N + 1, dtype='int')
    theta1 = -np.arctan((N + 2 - 2 * n) / N)
    theta1 = np.pi / 2 - theta1

    n = np.arange(1, N + 1, dtype='int')
    theta2 = np.arctan((N + 2 - 2 * n) / N)

    n = np.arange(1, N / 2 + 1, dtype='int')
    theta3 = -np.pi / 2 + np.arctan((N + 2 - 2 * n) / N)

    angles = np.concatenate([theta1, theta2, theta3], axis=0)
    angles = angles * 180 / np.pi
    angles.sort()
    return angles


def calc_golden_ratio_angles(tilt_range, nangles):
    """
    Calculate golden ratio angles for a given tilt range.

    See:

            A. P. Kaestner, B. Munch and P. Trtik, Opt. Eng., 2011, 50, 123201.
            https://doi.org/10.1117/1.3660298

    Parameters
    ----------
    tilt_range : integer
        Tilt range in degrees.

    nangles : integer
        Number of angles to calculate.

    Returns
    ----------
    thetas : Numpy Array
        Angles in degrees for golden ratio sampling over the provided tilt range.

    """
    alpha = tilt_range / 180 * np.pi
    i = np.arange(nangles) + 1
    thetas = np.mod(i * alpha * ((1 + np.sqrt(5)) / 2), alpha) - alpha / 2
    thetas = thetas * 180 / np.pi
    return thetas


def get_radial_mask(mask_shape, center=None):
    """
    Calculate a radial mask given a shape and center position.

    Parameters
    ----------
    mask_shape : list
        Shape (rows, cols) of the resulting mask.

    center : list
        Location of mask center (x,y).

    Returns
    ----------
    mask : Numpy Array
        Logical array that is True in the masked region and False outside of it.

    """
    if center is None:
        center = [int(i / 2) for i in mask_shape]
    radius = min(center[0], center[1], mask_shape[1] - center[0], mask_shape[0] - center[1])
    yy, xx = np.ogrid[0:mask_shape[0], 0:mask_shape[1]]
    mask = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    mask = mask < radius
    return mask
