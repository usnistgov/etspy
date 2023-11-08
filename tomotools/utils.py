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
        Level of accuracy for determining the weighting.  Acceptable values are 'good', 'medium', 'super', and 'unbelievable'.

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
