# -*- coding: utf-8 -*-
#
# This file is part of TomoTools

"""
Test dataset handling module for TomoTools package.

@author: Andrew Herzing
"""

import tomotools.api as tomotools
from tomotools.simulation import misalign_stack, add_gaussian_noise
import os

tomotools_path = os.path.dirname(tomotools.__file__)


def get_needle_data(aligned=False):
    """
    Retrieve experimental tilt series of needle-shaped specimen.

    Returns
    ----------
    needle : TomoStack object
        TomoStack containing the simulated catalyst tilt series

    """

    if aligned:
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        needle =\
            tomotools.load(filename)
    else:
        filename = os.path.join(tomotools_path, "tests",
                                "test_data", "HAADF.mrc")
        needle =\
            tomotools.load(filename)
    return needle


def get_catalyst_tilt_series(misalign=False, minshift=-5, maxshift=5,
                             tiltshift=0, tiltrotate=0, xonly=False,
                             noise=False, noise_factor=0.2):
    """
    Retrieve model catalyst tilt series.

    Arguments
    ----------
    misalign : bool
        If True, apply random shifts to each projection to simulated drift
    minshift : float
        Lower bound for random shifts
    maxshift : float
        Upper bound for random shifts
    tiltshift : float
        Number of pixels by which to shift entire tilt series. Simulates
        offset tilt axis.
    rotate : float
        Angle by which to rotate entire tilt series. Simulates non-vertical
        tilt axis.
    xonly : bool
        If True, shifts are only applied along the X-axis
    noise : bool
        If True, add Gaussian noise to the stack
    noise_factor : float
        Percentage noise to be added. Must be between 0 and 1.

    Returns
    ----------
    catalyst : TomoStack object
        TomoStack containing the simulated catalyst tilt series

    """

    filename = os.path.join(tomotools_path, "tests", "test_data",
                            "Catalyst3DModel_TiltSeries180.hdf5")
    catalyst =\
        tomotools.load(filename)
    if misalign:
        catalyst = misalign_stack(catalyst, min_shift=minshift,
                                  max_shift=maxshift, tilt_shift=tiltshift,
                                  tilt_rotate=tiltrotate, x_only=xonly)
    if noise:
        catalyst = add_gaussian_noise(catalyst, noise_factor)
    return catalyst
