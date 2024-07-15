# -*- coding: utf-8 -*-
#
# This file is part of ETSpy

"""
Test dataset handling module for ETSpy package.

@author: Andrew Herzing
"""

import etspy.api as etspy
from etspy.simulation import misalign_stack, add_noise
import os

etspy_path = os.path.dirname(etspy.__file__)


def get_needle_data(aligned=False):
    """
    Retrieve experimental tilt series of needle-shaped specimen.

    Returns
    ----------
    needle : TomoStack object
        TomoStack containing the simulated catalyst tilt series

    """
    if aligned:
        filename = os.path.join(
            etspy_path, "tests", "test_data", "HAADF_Aligned.hdf5"
        )
        needle = etspy.load(filename)
    else:
        filename = os.path.join(etspy_path, "tests", "test_data", "HAADF.mrc")
        needle = etspy.load(filename)
    return needle


def get_catalyst_data(
    misalign=False,
    minshift=-5,
    maxshift=5,
    tiltshift=0,
    tiltrotate=0,
    yonly=False,
    noise=False,
    noise_factor=0.2,
):
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
    filename = os.path.join(
        etspy_path, "tests", "test_data", "Catalyst3DModel_TiltSeries180.hdf5"
    )
    catalyst = etspy.load(filename)
    if misalign:
        catalyst = misalign_stack(
            catalyst,
            min_shift=minshift,
            max_shift=maxshift,
            tilt_shift=tiltshift,
            tilt_rotate=tiltrotate,
            y_only=yonly,
        )
    if noise:
        catalyst = add_noise(catalyst, "gaussian", noise_factor)
    return catalyst
