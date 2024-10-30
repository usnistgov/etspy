"""Utility module for ETSpy package."""

import logging
from multiprocessing import Pool
from typing import Literal, Optional, cast

import numpy as np
import tqdm
from hyperspy._signals.signal2d import Signal2D
from hyperspy.axes import UniformDataAxis as Uda
from pystackreg import StackReg
from scipy import ndimage

from etspy import _format_choices as _fmt
from etspy import _get_literal_hint_values as _get_lit
from etspy.align import calculate_shifts_stackreg
from etspy.base import TomoStack


def multiaverage(stack: np.ndarray, nframes: int, ny: int, nx: int) -> np.ndarray:
    """
    Register a multi-frame series collected by SerialEM.

    Parameters
    ----------
    stack
        Array of shape [nframes, ny, nx].
    nframes
        Number of frames per tilt.
    ny
        Pixels in y-dimension.
    nx
        Pixels in x-dimension.

    Returns
    -------
    average : :py:class:`~numpy.ndarray`
        Average of all frames at given tilt

    Group
    -----
    utilities
    """

    def _calc_sr_shifts(stack):
        sr = StackReg(StackReg.TRANSLATION)
        shifts = sr.register_stack(stack, reference="previous")
        shifts = -np.array([i[0:2, 2][::-1] for i in shifts])
        return shifts

    shifted = np.zeros([nframes, ny, nx])
    shifts = _calc_sr_shifts(stack)
    for k in range(nframes):
        shifted[k, :, :] = ndimage.shift(
            stack[k, :, :],
            shift=[shifts[k, 0], shifts[k, 1]],
        )
    average = shifted.mean(0)
    return average


def register_serialem_stack(stack: Signal2D, ncpus: int = 1) -> TomoStack:
    """
    Register a multi-frame series collected by SerialEM.

    Parameters
    ----------
    stack
        Signal of shape [ntilts, nframes, ny, nx].

    Returns
    -------
    reg : TomoStack
        Result of aligning and averaging frames at each tilt with shape [ntilts, ny, nx]

    Group
    -----
    utilities
    """
    align_logger = logging.getLogger("etspy.align")
    log_level = align_logger.getEffectiveLevel()
    align_logger.setLevel(logging.ERROR)
    ntilts, nframes, ny, nx = stack.data.shape

    if ncpus == 1:
        reg = np.zeros([ntilts, ny, nx], stack.data.dtype)
        for i in tqdm.tqdm(range(ntilts)):
            shifted = np.zeros([nframes, ny, nx])
            shifts = calculate_shifts_stackreg(
                stack.inav[:, i],
                start=None,
                show_progressbar=False,
            )
            for k in range(nframes):
                shifted[k, :, :] = ndimage.shift(
                    stack.data[i, k, :, :],
                    shift=[shifts[k, 0], shifts[k, 1]],
                )
            reg[i, :, :] = shifted.mean(0)
    else:
        with Pool(ncpus) as pool:
            reg = pool.starmap(
                multiaverage,
                [(stack.inav[:, i].data, nframes, ny, nx) for i in range(ntilts)],
            )
        reg = np.array(reg)

    reg = TomoStack(reg)
    reg_ax_0, reg_ax_1, reg_ax_2 = (cast(Uda, reg.axes_manager[i]) for i in range(3))
    stack_ax_1, stack_ax_2, stack_ax_3 = (
        cast(Uda, stack.axes_manager[i]) for i in range(1, 4)
    )
    reg_ax_0.scale = stack_ax_1.scale
    reg_ax_0.offset = stack_ax_1.offset
    reg_ax_0.units = stack_ax_1.units

    reg_ax_1.scale = stack_ax_2.scale
    reg_ax_1.offset = stack_ax_2.offset
    reg_ax_1.units = stack_ax_2.units

    reg_ax_2.scale = stack_ax_3.scale
    reg_ax_2.offset = stack_ax_3.offset
    reg_ax_2.units = stack_ax_3.units

    if stack.metadata.has_item("Acquisition_instrument"):
        reg.metadata.Acquisition_instrument = stack.metadata.Acquisition_instrument
    if stack.metadata.has_item("Tomography"):
        reg.metadata.Tomography = stack.metadata.Tomography
    align_logger.setLevel(log_level)
    return reg


def weight_stack(
    stack: TomoStack,
    accuracy: Literal["low", "medium", "high"] = "medium",
) -> TomoStack:
    """
    Apply a weighting window to a stack perpendicular to the tilt axis.

    This weighting is useful for reducing the effects of mass introduced at the
    edges of as stack when determining alignments based on the center of mass.
    As described in:

        T. Sanders. Physically motivated global alignment method for electron
        tomography, Advanced Structural and Chemical Imaging vol. 1 (2015) pp 1-11.
        https://doi.org/10.1186/s40679-015-0005-7

    Parameters
    ----------
    stack
        The stack to be weighted.
    accuracy
        A string indicating the accuracy level for weighting. Options are:
        'low', 'medium' (default), or 'high'.

    Returns
    -------
    stackw : TomoStack
        The weighted version of the input stack.

    Group
    -----
    utilities
    """
    # Set the parameters based on the accuracy input
    # with default of "medium"
    niterations = 2000
    delta = 0.01
    if accuracy.lower():
        if accuracy == "low":
            niterations = 800
            delta = 0.025
        elif accuracy == "medium":
            pass
        elif accuracy == "high":
            niterations = 20000
            delta = 0.001
        else:
            msg = (
                f'Invalid accuracy level "{accuracy}". Must be one of '
                f"{_fmt(_get_lit(weight_stack, 'accuracy'))}."
            )
            raise ValueError(msg)

    weighted_stack = stack.deepcopy()

    # Get stack dimensions
    ntilts, ny, nx = weighted_stack.data.shape

    # Compute the minimum total projected mass and the corresponding
    # slice index (min_slice)
    min_mass, min_slice = (
        np.min(
            np.sum(np.sum(weighted_stack.data, axis=2), axis=1),
        ),
        np.argmin(np.sum(np.sum(weighted_stack.data, axis=2), axis=1)),
    )

    # Initialize the window array
    window = np.zeros([ny, nx])

    # Initialize the status vector (1 means unmarked, 0 means marked) and mark
    # the reference slice (min_slice)
    status = np.ones(ntilts)
    status[min_slice] = 0

    # Generate the weighting profile `r` based on a non-linear cosine function
    r = np.arange(ny)
    r = 2 / (ny - 1) * r - 1
    r = np.cos(np.pi * r**2) / 2 + 0.5

    # Initialize adjustment factors for each slice
    adjustments = np.zeros(ntilts)

    # Coarse adjustment loop
    # In this step, the applied window is made increasingly restrictive in 10 pixel
    # increments. Whenever the the windowed mass of a projection drops below the value
    # of min_alpha, that projection is marked and the window restriction is not carried
    # any further for that projection.

    power = 10  # initialize power
    for power in np.linspace(10, niterations, niterations // 10):
        # Compute the power-weighted profile for the current iteration
        r_power = r ** (power * delta)
        window = r_power[:, np.newaxis]  # Broadcasting across all columns

        # Compute the weighted sum for all slices at once using vectorization
        weighted_mass = np.sum(
            weighted_stack.data * window[np.newaxis, :, :],
            axis=(1, 2),
        )

        # Update the status and adjustments for slices with weighted sums below min_mass
        update_mask = (status != 0) & (weighted_mass < min_mass)
        status[update_mask] = 0
        adjustments[update_mask] = power - 10

        # Break early if all slices are marked
        if not np.any(status):  # More efficient than np.sum(status)
            break

    # Set window for any unmarked slices to the most restricive used
    # in the rest of the slices
    adjustments[np.where(status != 0)] = power - 10

    # Fine adjustment loop
    # In this step the severity of the window is calculated again using the value
    # calculated in the coarse step and the window is made more restrictive in 1
    # pixel increments.
    status = np.ones(ntilts)
    status[min_slice] = 0

    for j in range(ntilts):
        if j != min_slice:
            for power in np.linspace(1, 10, 10):
                # Apply fine adjustments to the weight profile and
                # update the weight grid
                r_power = r ** ((power + adjustments[j]) * delta)
                window[:] = r_power[:, np.newaxis]

                if np.sum(weighted_stack.data[j, :, :] * window) < min_mass:
                    adjustments[j] = (power - 1) + adjustments[j]
                    status[j] = 0
                    break

    # Restrict the window of any unmarked projections
    adjustments[status != 0] += 10

    # Apply the final window to the entire stack
    for i in range(ntilts):
        window[:] = (r ** (adjustments[i] * delta))[:, np.newaxis]
        weighted_stack.data[i, :, :] *= window

    return weighted_stack


def calc_est_angles(num_points: int) -> np.ndarray:
    """
    Caculate angles used for equally sloped tomography (EST).

    See:
        J. Miao, F. Forster, and O. Levi. Equally sloped tomography with
        oversampling reconstruction. Phys. Rev. B, 72 (2005) 052103.
        https://doi.org/10.1103/PhysRevB.72.052103

    Parameters
    ----------
    num_points
        Number of points in scan.

    Returns
    -------
    angles : :py:class:`~numpy.ndarray`
        Angles in degrees for equally sloped tomography.

    Group
    -----
    utilities
    """
    if np.mod(num_points, 2) != 0:
        msg = "N must be an even number"
        raise ValueError(msg)

    angles = np.zeros(2 * num_points)

    n = np.arange(num_points / 2 + 1, num_points + 1, dtype="int")
    theta1 = -np.arctan((num_points + 2 - 2 * n) / num_points)
    theta1 = np.pi / 2 - theta1

    n = np.arange(1, num_points + 1, dtype="int")
    theta2 = np.arctan((num_points + 2 - 2 * n) / num_points)

    n = np.arange(1, num_points / 2 + 1, dtype="int")
    theta3 = -np.pi / 2 + np.arctan((num_points + 2 - 2 * n) / num_points)

    angles = np.concatenate([theta1, theta2, theta3], axis=0)
    angles = angles * 180 / np.pi
    angles.sort()
    return angles


def calc_golden_ratio_angles(tilt_range: int, nangles: int) -> np.ndarray:
    """
    Calculate golden ratio angles for a given tilt range.

    See:

        A. P. Kaestner, B. Munch and P. Trtik, Opt. Eng., 2011, 50, 123201.
        https://doi.org/10.1117/1.3660298

    Parameters
    ----------
    tilt_range
        Tilt range in degrees.

    nangles
        Number of angles to calculate.

    Returns
    -------
    thetas : :py:class:`~numpy.ndarray`
        Angles in degrees for golden ratio sampling over the provided tilt range.

    Group
    -----
    utilities
    """
    alpha = tilt_range / 180 * np.pi
    i = np.arange(nangles) + 1
    thetas = np.mod(i * alpha * ((1 + np.sqrt(5)) / 2), alpha) - alpha / 2
    thetas = thetas * 180 / np.pi
    return thetas


def get_radial_mask(
    mask_shape: tuple[int, int],
    center: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Calculate a radial mask given a shape and center position.

    Parameters
    ----------
    mask_shape
        Shape (rows, cols) of the resulting mask.

    center
        (x, y) location of mask center, optional. If ``None``, the center of
        the ``mask_shape`` will be used by default.

    Returns
    -------
    mask : :py:class:`~numpy.ndarray`
        Logical array that is True in the masked region and False outside of it.

    Group
    -----
    utilities
    """
    if center is None:
        center = cast(tuple[int, int], tuple(int(i / 2) for i in mask_shape))
    radius = min(
        center[0],
        center[1],
        mask_shape[1] - center[0],
        mask_shape[0] - center[1],
    )
    yy, xx = np.ogrid[0 : mask_shape[0], 0 : mask_shape[1]]
    mask = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mask = mask < radius
    return mask


def filter_stack(
    stack: TomoStack,
    filter_name: Literal[
        "ram-lak",
        "shepp-logan",
        "hanning",
        "hann",
        "cosine",
        "cos",
    ] = "shepp-logan",
    cutoff: float = 0.5,
) -> TomoStack:
    """
    Apply a Fourier filter to a sinogram or series of sinograms.

    Parameters
    ----------
    stack
        TomoStack with projection data
    filter_name
        Type of filter to apply.
    cutoff
        Factor of sampling rate to use as the cutoff.  Default is 0.5 which
        corresponds to the Nyquist frequency.

    Returns
    -------
    result : TomoStack
        Filtered version of the input TomoStack.

    Group
    -----
    utilities
    """
    _, ny = stack.data.shape[0:2]

    filter_length = max(64, 2 ** (int(np.ceil(np.log2(2 * ny)))))
    freq_indices = np.arange(filter_length // 2 + 1)
    ffilter = np.linspace(
        cutoff / filter_length,
        1 - cutoff / filter_length,
        len(freq_indices),
    )
    omega = 2 * np.pi * freq_indices / filter_length

    if filter_name == "ram-lak":
        pass
    elif filter_name == "shepp-logan":
        ffilter[1:] = ffilter[1:] * np.sinc(omega[1:] / (2 * np.pi))
    elif filter_name in [
        "hanning",
        "hann",
    ]:
        ffilter[1:] = ffilter[1:] * (1 + np.cos(omega[1:])) / 2
    elif filter_name in [
        "cosine",
        "cos",
    ]:
        ffilter[1:] = ffilter[1:] * np.cos(omega[1:] / 2)
    else:
        msg = (
            f'Invalid filter type "{filter_name}". Must be one of '
            f"{_fmt(_get_lit(filter_stack, 'filter_name'))}."
        )
        raise ValueError(msg)

    ffilter = np.concatenate((ffilter, ffilter[-2:0:-1]))

    nfilter = ffilter.shape[0]
    pad_length = int((nfilter - ny) / 2)

    if len(stack.data.shape) == 2:  # noqa: PLR2004
        padded = np.pad(stack.data, [[0, 0], [pad_length, pad_length]])
        proj_fft = np.fft.fft(padded, axis=1)
        filtered = np.fft.ifft(proj_fft * ffilter, axis=1).real
        filtered = filtered[:, pad_length:-pad_length]

    elif len(stack.data.shape) == 3:  # noqa: PLR2004
        padded = np.pad(stack.data, [[0, 0], [pad_length, pad_length], [0, 0]])
        proj_fft = np.fft.fft(padded, axis=1)
        filtered = np.fft.ifft(proj_fft * ffilter[:, np.newaxis], axis=1).real
        filtered = filtered[:, pad_length:-pad_length, :]
    else:
        msg = "Method can only be applied to 2 or 3-dimensional stacks"
        raise ValueError(msg)
    result = stack.deepcopy()
    result.data = filtered
    return result
