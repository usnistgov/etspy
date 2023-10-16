# -*- coding: utf-8 -*-
#
# This file is part of TomoTools

"""
Utility module for TomoTools package.

@author: Andrew Herzing
"""

import numpy as np
from tomotools.io import convert_to_tomo_stack
from tomotools.base import TomoStack
import logging
import tqdm


def register_serialem_stack(stack, method='PC'):
    """
    Register a multi-frame series collected by SerialEM.

    Parameters
    ----------
    stack : Hyperspy Signal2D
        Signal of shape [ntilts, nframes, ny, nx].

    method : string
        Stack registration method to use.

    Returns
    ----------
    reg : TomoStack object
        Result of aligning and averaging frames at each tilt with shape [ntilts, ny, nx]

    """
    align_logger = logging.getLogger("tomotools.align")
    log_level = align_logger.getEffectiveLevel()
    align_logger.setLevel(logging.ERROR)

    reg = np.zeros([stack.data.shape[0], stack.data.shape[2],
                   stack.data.shape[3]], stack.data.dtype)
    for i in tqdm.tqdm(range(0, stack.data.shape[0])):
        temp = TomoStack(np.float32(stack.data[i]))
        reg[i, :, :] = temp.stack_register(method=method).data.mean(0)
    reg = convert_to_tomo_stack(reg)

    if stack.metadata.has_item("Tomography"):
        reg.metadata.Tomography.tilts = stack.metadata.Tomography.tilts
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
