"""Module to provide example tomographic datasets."""

import etspy.api as etspy
from etspy.api import etspy_path
from etspy.simulation import add_noise, misalign_stack


def get_needle_data(aligned: bool = False):
    """
    Load an experimental tilt series of needle-shaped specimen.

    Data size is 77 tilt projections and 256 x 256 pixel images.

    Returns
    -------
    needle : TomoStack
        TomoStack containing the simulated catalyst tilt series

    Group
    -----
    datasets
    """
    if aligned:
        filename = etspy_path / "tests" / "test_data" / "HAADF_Aligned.hdf5"
        needle = etspy.load(filename)
    else:
        filename = etspy_path / "tests" / "test_data" / "HAADF.mrc"
        needle = etspy.load(filename)
    return needle


def get_catalyst_data(
    misalign: bool = False,
    minshift: int = -5,
    maxshift: int = 5,
    tiltshift: int = 0,
    tiltrotate: int = 0,
    y_only: bool = False,
    noise: bool = False,
    noise_factor: float = 0.2,
) -> etspy.TomoStack:
    """
    Load a model-simulated catalyst tilt series.

    Data size is 90 tilt projections and 600 x 600 pixel images.

    Parameters
    ----------
    misalign
        If True, apply random shifts to each projection to simulated drift
    minshift
        Lower bound for random shifts
    maxshift
        Upper bound for random shifts
    tiltshift
        Number of pixels by which to shift entire tilt series. Simulates
        offset tilt axis.
    tiltrotate
        Angle by which to rotate entire tilt series. Simulates non-vertical
        tilt axis.
    y_only
        If ``True``, shifts are only applied along the Y-axis
    noise
        If ``True``, add Gaussian noise to the stack
    noise_factor
        Percentage noise to be added. Must be between 0 and 1.

    Returns
    -------
    catalyst : TomoStack
        TomoStack containing the simulated catalyst tilt series

    Group
    -----
    datasets
    """
    filename = etspy_path / "tests" / "test_data" / "Catalyst3DModel_TiltSeries180.hdf5"
    catalyst = etspy.load(filename)
    if misalign:
        catalyst = misalign_stack(
            catalyst,
            min_shift=minshift,
            max_shift=maxshift,
            tilt_shift=tiltshift,
            tilt_rotate=tiltrotate,
            y_only=y_only,
        )
    if noise:
        catalyst = add_noise(catalyst, "gaussian", noise_factor)
    return catalyst
