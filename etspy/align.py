"""Alignment module for ETSpy package."""

# pyright: reportPossiblyUnboundVariable=false

import copy
import logging
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import astra
import matplotlib.pylab as plt
import numpy as np
import tqdm
from hyperspy.misc.utils import DictionaryTreeBrowser as Dtb
from hyperspy.signal import BaseSignal
from pystackreg import StackReg
from scipy import ndimage, optimize
from skimage.feature import canny
from skimage.filters import sobel
from skimage.registration import phase_cross_correlation as pcc
from skimage.transform import hough_line, hough_line_peaks

from etspy import AlignmentMethod, AlignmentMethodType

if TYPE_CHECKING:
    from etspy.base import TomoShifts, TomoStack  # pragma: no cover

has_cupy = True
try:
    import cupy as cp  # type: ignore
except ImportError:
    has_cupy = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CL_RES_THRESHOLD = 0.5  # threshold for common line registration method


def get_best_slices(stack: "TomoStack", nslices: int) -> np.ndarray:
    """
    Get best nslices for center of mass analysis.

    Slices which have the highest ratio of total mass to mass variance
    and their location are returned.

    Parameters
    ----------
    stack
        Tilt series from which to select the best slices
    nslices
        Number of slices to return

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Location along the x-axis of the best slices

    Group
    -----
    align
    """
    total_mass = stack.data.sum((0, 1))
    mass_std = stack.data.sum(1).std(0)
    mass_std[mass_std == 0] = 1e-5
    mass_ratio = total_mass / mass_std
    best_slice_locations = mass_ratio.argsort()[::-1][0:nslices]
    return best_slice_locations


def get_coms(stack: "TomoStack", slices: np.ndarray) -> np.ndarray:
    """
    Calculate the center of mass for indicated slices.

    Parameters
    ----------
    stack
        Tilt series from which to calculate the centers of mass.
    slices
        Location of slices to use for center of mass calculation.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Center of mass as a function of tilt for each slice [ntilts, nslices].

    Group
    -----
    align
    """
    sinos = stack.data[:, :, slices]
    com_range = int(sinos.shape[1] / 2)
    y_coordinates = np.linspace(-com_range, com_range, sinos.shape[1], dtype="int")
    total_mass = sinos.sum(1)
    coms = np.sum(np.transpose(sinos, [0, 2, 1]) * y_coordinates, 2) / total_mass
    return coms


def apply_shifts(
    stack: "TomoStack",
    shifts: Union["TomoShifts", np.ndarray],
) -> "TomoStack":
    """
    Apply a series of shifts to a TomoStack.

    Parameters
    ----------
    stack
        The image series to be aligned
    shifts
        The X- (tilt parallel) and Y-shifts (tilt perpendicular) to be applied to
        each image. Should be of size
        ``(*stack.axes_manager.navigation_shape[::-1], 2)``,
        with Y-shifts in the ``shifts[:, 0]`` position and X-shifts in ``shifts[:, 1]``
        position (if ``shifts`` is a :py:class:`~numpy.ndarray`).

    Returns
    -------
    shifted : TomoStack
        Copy of input stack after shifts are applied

    Group
    -----
    align
    """
    shifted = stack.deepcopy()
    if isinstance(shifts, BaseSignal):
        shifts = shifts.data
    shifts = cast(np.ndarray, shifts)

    if len(shifts.data) != stack.data.shape[0]:
        msg = (
            f"Number of shifts ({len(shifts)}) is not consistent "
            f"with number of images in the stack ({stack.data.shape[0]})"
        )
        raise ValueError(msg)
    for i in range(shifted.data.shape[0]):
        shifted.data[i, :, :] = ndimage.shift(
            shifted.data[i, :, :],
            shift=[shifts[i, 0], shifts[i, 1]],
        )

    shifted.shifts.data = shifted.shifts.data + shifts
    return shifted


def pad_line(line: np.ndarray, paddedsize: int) -> np.ndarray:
    """
    Pad a 1D array for FFT treatment without altering center location.

    Parameters
    ----------
    line
        The data to be padded (should be 1D)
    paddedsize
        The size of the desired padded data.

    Returns
    -------
    padded : :py:class:`~numpy.ndarray`
        Padded version of input data (1 dimensional)

    Group
    -----
    align
    """
    npix = len(line)
    start_index = (paddedsize - npix) // 2
    end_index = start_index + npix
    padded_line = np.zeros(paddedsize)
    padded_line[start_index:end_index] = line
    return padded_line


def calc_shifts_cl(
    stack: "TomoStack",
    cl_ref_index: Optional[int],
    cl_resolution: float,
    cl_div_factor: int,
) -> np.ndarray:
    """
    Calculate shifts using the common line method.

    Used to align stack in dimension parallel to the tilt axis

    Parameters
    ----------
    stack
        The stack on which to calculate shifts
    cl_ref_index
        Tilt index of reference projection. If not provided the projection
        closest to the middle of the stack will be chosen.
    cl_resolution
        Degree of sub-pixel analysis
    cl_div_factor
        Factor used to determine number of iterations of alignment.

    Returns
    -------
    yshifts : :py:class:`~numpy.ndarray`
        Shifts parallel to tilt axis for each projection

    Group
    -----
    align
    """

    def align_line(ref_line, line, cl_resolution, cl_div_factor):
        npad = len(ref_line) * 2 - 1

        # Pad with zeros while preserving the center location
        ref_line_pad = pad_line(ref_line, npad)
        line_pad = pad_line(line, npad)

        niters = int(np.abs(np.floor(np.log(cl_resolution) / np.log(cl_div_factor))))
        start, end = -0.5, 0.5

        ref_line_pad_ft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(ref_line_pad)))
        line_pad_ft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(line_pad)))

        midpoint = (npad - 1) / 2
        kx = np.arange(-midpoint, midpoint + 1)

        for _ in range(niters):
            boundary = np.linspace(start, end, cl_div_factor, endpoint=False)
            index = (boundary[:-1] + boundary[1:]) / 2

            max_vals = np.zeros(len(index))
            for j, idx in enumerate(index):
                pfactor = np.exp(2 * np.pi * 1j * (idx * kx / npad))
                conjugate = np.conj(ref_line_pad_ft) * line_pad_ft * pfactor
                xcorr = np.abs(
                    np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(conjugate))),
                )
                max_vals[j] = np.max(xcorr)

            max_loc = np.argmax(max_vals)
            start, end = boundary[max_loc], boundary[max_loc + 1]

        subpixel_shift = index[max_loc]
        max_pfactor = np.exp(2 * np.pi * 1j * (subpixel_shift * kx / npad))

        # Determine integer shift via cross correlation
        conjugate = np.conj(ref_line_pad_ft) * line_pad_ft * max_pfactor
        xcorr = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(conjugate))))
        max_loc = np.argmax(xcorr)

        integer_shift = max_loc
        integer_shift = integer_shift - midpoint

        # Calculate full shift
        shift = integer_shift + subpixel_shift
        return -shift

    if cl_ref_index is None:
        cl_ref_index = stack.data.shape[0] // 2

    yshifts = np.zeros(stack.data.shape[0])
    ref_cm_line = stack.data[cl_ref_index].sum(0)

    for i in tqdm.tqdm(range(stack.data.shape[0])):
        if i == cl_ref_index:
            continue
        curr_cm_line = stack.data[i].sum(0)
        yshifts[i] = align_line(ref_cm_line, curr_cm_line, cl_resolution, cl_div_factor)
    return yshifts


def calculate_shifts_conservation_of_mass(
    stack: "TomoStack",
    xrange: Optional[Tuple[int, int]] = None,
    p: int = 20,
) -> np.ndarray:
    """
    Calculate shifts parallel to the tilt axis using conservation of mass.

    Slices which have the highest ratio of total mass to mass variance
    and their location are returned.

    Parameters
    ----------
    stack
        Tilt series to be aligned.
    xrange
        The range for performing alignment
    p
        Padding element

    Returns
    -------
    xshifts : :py:class:`~numpy.ndarray`
        Calculated shifts parallel to tilt axis.

    Group
    -----
    align
    """
    logger.info("Refinining X-shifts using conservation of mass method")
    ntilts, _, nx = stack.data.shape

    if xrange is None:
        xrange = (round(nx / 5), round(4 / 5 * nx))
    else:
        xrange = (round(xrange[0]) + p, round(xrange[1]) - p)

    xshifts = np.zeros([ntilts, 1])
    total_mass = np.zeros([ntilts, xrange[1] - xrange[0] + 2 * p + 1])

    for i in range(ntilts):
        total_mass[i, :] = np.sum(
            stack.data[i, :, xrange[0] - p - 1 : xrange[1] + p],
            0,
        )

    mean_mass = np.mean(total_mass[:, p:-p], 0)

    for i in range(ntilts):
        s = 0
        for j in range(-p, p):
            resid = np.linalg.norm(mean_mass - total_mass[i, p + j : -p + j])
            if resid < s or j == -p:
                s = resid
                xshifts[i] = -j
    return xshifts[:, 0]


def calculate_shifts_com(stack: "TomoStack", nslices: int) -> np.ndarray:
    """
    Align stack using a center of mass method.

    Data is first registered using PyStackReg. Then, the shifts
    perpendicular to the tilt axis are refined by a center of
    mass analysis.

    Parameters
    ----------
    stack
        The image series to be aligned

    nslices
        Number of slices to return

    Returns
    -------
    shifts : :py:class:`~numpy.ndarray`
        The X- and Y-shifts to be applied to each image

    Group
    -----
    align
    """
    logger.info("Refinining Y-shifts using center of mass method")
    slices = get_best_slices(stack, nslices)

    angles = stack.tilts.data.squeeze()
    ntilts, _, _ = stack.data.shape
    thetas = np.pi * cast(np.ndarray, angles) / 180

    coms = get_coms(stack, slices)
    i_tilts = np.eye(ntilts)
    gam = np.array([np.cos(thetas), np.sin(thetas)]).T
    gam = np.dot(gam, np.linalg.pinv(gam)) - i_tilts
    b = np.dot(gam, coms)

    cx = np.linalg.lstsq(gam, b, rcond=-1)[0]

    yshifts = -cx[:, 0]
    return yshifts


def _upsampled_dft(
    data,
    upsampled_region_size,
    upsample_factor,
    axis_offsets,
):
    # missing coverage because of CUDA
    upsampled_region_size = [
        upsampled_region_size,
    ] * data.ndim

    im2pi = 1j * 2 * cp.pi

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        kernel = (cp.arange(ups_size) - ax_offset)[:, None] * cp.fft.fftfreq(
            n_items,
            upsample_factor,
        )
        kernel = cp.exp(-im2pi * kernel)
        # use kernel with same precision as the data
        kernel = kernel.astype(data.dtype, copy=False)
        data = cp.tensordot(kernel, data, axes=(1, -1)) # type: ignore
    return data


def _cupy_phase_correlate(ref_cp, mov_cp, upsample_factor, shape):
    # missing coverage b/c of CUDA
    ref_fft = cp.fft.fftn(ref_cp)
    mov_fft = cp.fft.fftn(mov_cp)

    cross_power_spectrum = ref_fft * mov_fft.conj()
    eps = cp.finfo(cross_power_spectrum.real.dtype).eps
    cross_power_spectrum /= cp.maximum(cp.abs(cross_power_spectrum), 100 * eps)
    phase_correlation = cp.fft.ifft2(cross_power_spectrum)

    maxima = cp.unravel_index(
        cp.argmax(cp.abs(phase_correlation)),
        phase_correlation.shape,
    )
    midpoint = cp.array([cp.fix(axis_size / 2) for axis_size in shape])

    float_dtype = cross_power_spectrum.real.dtype

    shift = cp.stack(maxima).astype(float_dtype, copy=False)
    shift[shift > midpoint] -= cp.array(shape)[shift > midpoint]

    if upsample_factor > 1:
        upsample_factor = cp.array(upsample_factor, dtype=float_dtype)
        upsampled_region_size = cp.ceil(upsample_factor * 1.5)
        dftshift = cp.fix(upsampled_region_size / 2.0)

        shift = cp.round(shift * upsample_factor) / upsample_factor

        sample_region_offset = dftshift - shift * upsample_factor
        phase_correlation = _upsampled_dft(
            cross_power_spectrum.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset,
        ).conj()
        maxima = np.unravel_index(
            cp.argmax(np.abs(phase_correlation)),
            phase_correlation.shape,
        )

        maxima = cp.stack(maxima).astype(float_dtype, copy=False)
        maxima -= dftshift

        shift += maxima / upsample_factor
    return shift


def _cupy_calculate_shifts(stack, start, show_progressbar, upsample_factor):
    stack_cp = cp.array(stack.data)
    shifts = cp.zeros([stack_cp.shape[0], 2])
    ref_cp = stack_cp[0]
    ref_fft = cp.fft.fftn(ref_cp)
    shape = ref_fft.shape
    with tqdm.tqdm(
        total=stack.data.shape[0] - 1,
        desc="Calculating shifts",
        disable=not show_progressbar,
    ) as pbar:
        for i in range(start, 0, -1):
            shift = _cupy_phase_correlate(
                stack_cp[i],
                stack_cp[i - 1],
                upsample_factor=upsample_factor,
                shape=shape,
            )
            shifts[i - 1] = shifts[i] + shift
            pbar.update(1)
        for i in range(start, stack.data.shape[0] - 1):
            shift = _cupy_phase_correlate(
                stack_cp[i],
                stack_cp[i + 1],
                upsample_factor=upsample_factor,
                shape=shape,
            )
            shifts[i + 1] = shifts[i] + shift
            pbar.update(1)
    shifts = shifts.get()
    return shifts


def calculate_shifts_pc(
    stack: "TomoStack",
    start: int,
    show_progressbar: bool = False,
    upsample_factor: int = 3,
    cuda: bool = False,
) -> np.ndarray:
    """
    Calculate shifts using the phase correlation algorithm.

    Parameters
    ----------
    stack
        The image series to be aligned
    start
        Position in tilt series to use as starting point for the alignment
    show_progressbar
        Enable/disable progress bar
    upsample_factor
        Factor by which to resample the data
    cuda
        Enable/disable the use of GPU-accelerated processes using CUDA

    Returns
    -------
    shifts : :py:class:`~numpy.ndarray`
        The X- and Y-shifts to be applied to each image

    Group
    -----
    align
    """
    if has_cupy and astra.use_cuda() and cuda:
        shifts = _cupy_calculate_shifts(stack, start, show_progressbar, upsample_factor)

    else:
        shifts = np.zeros((stack.data.shape[0], 2))
        with tqdm.tqdm(
            total=stack.data.shape[0] - 1,
            desc="Calculating shifts",
            disable=not show_progressbar,
        ) as pbar:
            for i in range(start, 0, -1):
                shift = pcc(
                    stack.data[i],
                    stack.data[i - 1],
                    upsample_factor=upsample_factor,
                )[0]
                shifts[i - 1] = shifts[i] + shift
                pbar.update(1)

            for i in range(start, stack.data.shape[0] - 1):
                shift = pcc(
                    stack.data[i],
                    stack.data[i + 1],
                    upsample_factor=upsample_factor,
                )[0]
                shifts[i + 1] = shifts[i] + shift
                pbar.update(1)

    return shifts


def calculate_shifts_stackreg(
    stack: "TomoStack",
    start: Optional[int],
    show_progressbar: bool,
) -> np.ndarray:
    """
    Calculate shifts using PyStackReg.

    Parameters
    ----------
    stack
        The image series to be aligned
    start
        Position in tilt series to use as starting point for the alignment. If ``None``,
        the slice closest to the midpoint will be used.
    show_progressbar
        Enable/disable progress bar

    Returns
    -------
    shifts : :py:class:`~numpy.ndarray`
        The X- and Y-shifts to be applied to each image

    Group
    -----
    align
    """
    shifts = np.zeros((stack.data.shape[0], 2))

    if start is None:
        start = stack.data.shape[0] // 2  # Use the midpoint if start is not provided
    start = cast(int, start)

    # Initialize pystackreg object with TranslationTransform2D
    reg = StackReg(StackReg.TRANSLATION)

    with tqdm.tqdm(
        total=stack.data.shape[0] - 1,
        desc="Calculating shifts",
        disable=not show_progressbar,
    ) as pbar:
        # Calculate shifts relative to the image at the 'start' index
        for i in range(start, 0, -1):
            transformation = reg.register(stack.data[i], stack.data[i - 1])
            shift = -transformation[0:2, 2][::-1]
            shifts[i - 1] = shifts[i] + shift
            pbar.update(1)

        for i in range(start, stack.data.shape[0] - 1):
            transformation = reg.register(stack.data[i], stack.data[i + 1])
            shift = -transformation[0:2, 2][::-1]
            shifts[i + 1] = shifts[i] + shift
            pbar.update(1)
    return shifts


def calc_shifts_com_cl(
    stack: "TomoStack",
    com_ref_index: int,
    cl_ref_index: Optional[int] = None,
    cl_resolution: float = 0.05,
    cl_div_factor: int = 8,
) -> np.ndarray:
    """
    Calculate shifts using combined center of mass and common line methods.

    Center of mass aligns stack perpendicular to the tilt axis and
    common line is used to align the stack parallel to the tilt axis.

    Parameters
    ----------
    stack
        Tilt series to be aligned
    com_ref_index
        Reference slice for center of mass alignment.  All other slices
        will be aligned to this reference.
    cl_ref_index
        Reference slice for common line alignment.  All other slices
        will be aligned to this reference. If not provided the projection
        closest to the middle of the stack will be chosen.
    cl_resolution
        Resolution for subpixel common line alignment. Default is 0.05.
        Should be less than 0.5.
    cl_div_factor
        Factor which determines the number of iterations of common line
        alignment to perform.  Default is 8.

    Returns
    -------
    reg : :py:class:`~numpy.ndarray`
        The X- and Y-shifts to be applied to each image

    Group
    -----
    align
    """

    def calc_yshifts(stack, com_ref):
        ntilts = stack.data.shape[0]
        ali_x = stack.deepcopy()
        coms = np.zeros(ntilts)
        yshifts = np.zeros_like(coms)

        for i in tqdm.tqdm(range(ntilts)):
            im = ali_x.data[i, :, :]
            coms[i], _ = ndimage.center_of_mass(im)
            yshifts[i] = com_ref - coms[i]
        return yshifts

    if cl_resolution >= CL_RES_THRESHOLD:
        msg = f"Resolution should be less than {CL_RES_THRESHOLD}"
        raise ValueError(msg)

    logger.info("Center of mass reference slice: %s", com_ref_index)
    logger.info("Common line reference slice: %s", cl_ref_index)
    xshifts = np.zeros(stack.data.shape[0])
    yshifts = np.zeros(stack.data.shape[0])
    yshifts = calc_yshifts(stack, com_ref_index)
    xshifts = calc_shifts_cl(stack, cl_ref_index, cl_resolution, cl_div_factor)
    shifts = np.stack([yshifts, xshifts], axis=1)
    return shifts


def align_stack(  # noqa: PLR0913
    stack: "TomoStack",
    method: AlignmentMethodType,
    start: Optional[int],
    show_progressbar: bool,
    xrange: Optional[Tuple[int, int]] = None,
    p: int = 20,
    nslices: int = 20,
    cuda: bool = False,
    upsample_factor: int = 3,
    com_ref_index: Optional[int] = None,
    cl_ref_index: Optional[int] = None,
    cl_resolution: float = 0.05,
    cl_div_factor: int = 8,
) -> "TomoStack":
    """
    Compute the shifts for spatial registration.

    Shifts are determined by one of three methods:
        1.) Phase correlation (PC) as implemented in scikit-image. Based on:
            Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup.
            Efficient subpixel image registration algorithms, Optics Letters vol. 33
            (2008) pp. 156-158.
            https://doi.org/10.1364/OL.33.000156
        2.) Center of mass (COM) tracking.  A Python implementation of
            algorithms described in:
            T. Sanders. Physically motivated global alignment method for electron
            tomography, Advanced Structural and Chemical Imaging vol. 1 (2015) pp 1-11.
            https://doi.org/10.1186/s40679-015-0005-7
        3.) Rigid translation using PyStackReg for shift calculation.
            PyStackReg is a Python port of the StackReg plugin for ImageJ
            which uses a pyramidal approach to minimize the least-squares
            difference in image intensity between a source and target image.
            StackReg is described in:
            P. Thevenaz, U.E. Ruttimann, M. Unser. A Pyramid Approach to
            Subpixel Registration Based on Intensity, IEEE Transactions
            on Image Processing vol. 7, no. 1, pp. 27-41, January 1998.
            https://doi.org/10.1109/83.650848
        4.) A combination of center of mass tracking for aligment of
            projections perpendicular to the tilt axis and common line
            alignment for parallel to the tilt axis. This is a Python
            implementation of Matlab code described in:
            M. C. Scott, et al. Electron tomography at 2.4-ångström resolution,
            Nature 483, 444-447 (2012).
            https://doi.org/10.1038/nature10934

    Shifts are then applied and the aligned stack is returned.  The tilts are
    stored in stack.metadata.Tomography.shifts for later use.

    Parameters
    ----------
    stack
        3-D numpy array containing the tilt series data
    method
        Method by which to calculate the alignments. Valid options
        are controlled by the :py:class:`etspy.AlignmentMethod` enum.
    start
        Position in tilt series to use as starting point for the alignment.
        If None, the central projection is used.
    show_progressbar
        Enable/disable progress bar
    xrange
        (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.COM`)
        The range for performing alignment. See
        :py:func:`~etspy.align.calculate_shifts_com` for more details.
    p
        (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.COM`)
        Padding element. See :py:func:`~etspy.align.calculate_shifts_com` for more
        details.
    nslices
        (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.COM`)
        Number of slices to return. See
        :py:func:`~etspy.align.calculate_shifts_com` for more details.
    cuda
        (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.PC`)
        Enable/disable the use of GPU-accelerated processes using CUDA. See
        :py:func:`~etspy.align.calculate_shifts_pc` for more details.
    upsample_factor
        (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.PC`)
        Factor by which to resample the data. See
        :py:func:`~etspy.align.calculate_shifts_pc` for more details.
    com_ref_index
        (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.COM_CL`)
        Reference slice for center of mass alignment.  All other slices will be aligned
        to this reference. See :py:func:`~etspy.align.calc_shifts_com_cl` for more
        details.
    cl_ref_index
        (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.COM_CL`)
        Reference slice for common line alignment.  All other slices
        will be aligned to this reference. If not provided the projection
        closest to the middle of the stack will be chosen. See
        :py:func:`~etspy.align.calc_shifts_com_cl` for more details.
    cl_resolution
        (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.COM_CL`)
        Resolution for subpixel common line alignment. Default is 0.05.
        Should be less than 0.5. See :py:func:`~etspy.align.calc_shifts_com_cl` for
        more details.
    cl_div_factor
        (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.COM_CL`)
        Factor which determines the number of iterations of common line
        alignment to perform.  Default is 8. See
        :py:func:`~etspy.align.calc_shifts_com_cl` for more details.

    Returns
    -------
    out : TomoStack
        Spatially registered copy of the input stack

    Group
    -----
    align
    """
    if start is None:
        start = (
            stack.data.shape[0] // 2
        )  # Use the slice closest to the midpoint if start is not provided
    start = cast(int, start)  # explicit type cast for type checking

    if method == AlignmentMethod.COM:
        logger.info("Performing stack registration using center of mass method")
        shifts = np.zeros([stack.data.shape[0], 2])
        # calculate_shifts_conservation_of_mass returns x-shifts (parallel to tilt axis)
        # calculate_shifts_com returns y-shifts (perpendicular to tilt axis)
        shifts[:, 1] = calculate_shifts_conservation_of_mass(stack, xrange, p)
        shifts[:, 0] = calculate_shifts_com(stack, nslices)
    elif method == AlignmentMethod.PC:
        if cuda:
            logger.info(  # pragma: no cover
                "Performing stack registration using "
                "CUDA-accelerated phase correlation",
            )
        else:
            logger.info("Performing stack registration using phase correlation")
        shifts = calculate_shifts_pc(
            stack,
            start,
            show_progressbar,
            upsample_factor,
            cuda,
        )
    elif method == AlignmentMethod.STACK_REG:
        logger.info("Performing stack registration using PyStackReg")
        shifts = calculate_shifts_stackreg(stack, start, show_progressbar)
    elif method == AlignmentMethod.COM_CL:
        logger.info(
            "Performing stack registration using combined "
            "center of mass and common line methods",
        )
        if com_ref_index is None:
            com_ref_index = stack.data.shape[1] // 2
        if cl_ref_index is None:
            cl_ref_index = stack.data.shape[0] // 2

        # explicit type casts for type checking
        com_ref_index = cast(int, com_ref_index)
        cl_ref_index = cast(int, cl_ref_index)

        shifts = calc_shifts_com_cl(
            stack,
            com_ref_index,
            cl_ref_index,
            cl_resolution,
            cl_div_factor,
        )
    else:
        msg = f"Invalid alignment method {method}"
        raise ValueError(msg)
    aligned = apply_shifts(stack, shifts)
    logger.info("Stack registration complete")
    return aligned


def tilt_com(
    stack: "TomoStack",
    slices: Optional[np.ndarray] = None,
    nslices: Optional[int] = None,
) -> "TomoStack":
    """
    Perform tilt axis alignment using center of mass (CoM) tracking.

    Compares path of specimen to the path expected for an ideal cylinder

    Parameters
    ----------
    stack
        TomoStack containing the tilt series data
    slices
        Locations at which to perform the Center of Mass analysis. If not
        provided, an appropriate list of slices will be automatically determined.
    nslices
        Nubmer of slices to use for the analysis (only used if the ``slices``
        parameter is not specified). If ``None``, a value of 10% of the x-axis
        size will be used, clamped to the range [3, 50].

    Returns
    -------
    out : TomoStack
        Copy of the input stack after rotation and translation to center and
        make the tilt axis vertical

    Group
    -----
    align
    """

    def com_motion(theta, r, x0, z0):
        return r - x0 * np.cos(theta) - z0 * np.sin(theta)

    def fit_line(x, m, b):
        return m * x + b

    _, ny, nx = stack.data.shape
    nx_threshold = 3

    if np.all(stack.tilts.data == 0):
        msg = (
            "Tilts are not defined in stack.tilts (values were all zeros). "
            "Please set tilt values before alignment."
        )
        raise ValueError(msg)

    if nx < nx_threshold:
        msg = (
            f"Dataset is only {stack.data.shape[2]} pixels in x dimension. "
            "This method cannot be used."
        )
        raise ValueError(msg)

    # Determine the best slice locations for the analysis
    if slices is None:
        if nslices is None:
            nslices = int(0.1 * nx)
            nslices = max(min(nslices, 50), 3)  # clamp nslices to [3, 50]
        else:
            if nslices > nx:
                msg = "nslices is greater than the X-dimension of the data."
                raise ValueError(msg)
            if nslices > 0.3 * nx:
                nslices = int(0.3 * nx)
                msg = (
                    "nslices is greater than 30% of number of x pixels. "
                    f"Using {nslices} slices instead."
                )
                logger.warning(msg)

        slices = get_best_slices(stack, nslices)
        logger.info("Performing alignments using best %s slices", nslices)

    slices = np.sort(slices)

    coms = get_coms(stack, slices)
    thetas = np.pi * stack.tilts.data.squeeze() / 180.0 # remove length 1 dimension

    r, x0, z0 = np.zeros(len(slices)), np.zeros(len(slices)), np.zeros(len(slices))

    for idx, _ in enumerate(slices):
        r[idx], x0[idx], z0[idx] = optimize.curve_fit(
            com_motion,
            xdata=thetas,
            ydata=coms[:, idx],
            p0=[0, 0, 0],
        )[0]
    slope, intercept = optimize.curve_fit(fit_line, xdata=r, ydata=slices, p0=[0, 0])[0]
    tilt_shift = (ny / 2 - intercept) / slope
    tilt_rotation = -(180 * np.arctan(1 / slope) / np.pi)

    final = cast("TomoStack", stack.trans_stack(yshift=tilt_shift, angle=tilt_rotation))

    logger.info("Calculated tilt-axis shift %.2f", tilt_shift)
    logger.info("Calculated tilt-axis rotation %.2f", tilt_rotation)

    return final


def tilt_maximage(
    stack: "TomoStack",
    limit: float = 10,
    delta: float = 0.1,
    plot_results: bool = False,
    also_shift: bool = False,
    shift_limit: float = 20,
) -> "TomoStack":
    """
    Perform automated determination of the tilt axis of a TomoStack.

    The projected maximum image used to determine the tilt axis by a
    combination of Sobel filtering and Hough transform analysis.

    Parameters
    ----------
    stack
        TomoStack array containing the tilt series data
    limit
        Maximum rotation angle to use for calculation
    delta
        Angular increment for calculation
    plot_results
        If ``True``, plot the maximum image along with the lines determined
        by Hough analysis
    also_shift
        If ``True``, also calculate and apply the global shift perpendicular to the tilt
        by minimizing the sum of the reconstruction
    shift_limit
        The limit of shifts applied if ``also_shift`` is set to ``True``

    Returns
    -------
    rotated : TomoStack
        Rotated version of the input stack

    Group
    -----
    align
    """
    image = stack.data.max(0)

    edges = sobel(image)

    # Apply Canny edge detector for further edge enhancement
    edges = canny(edges)

    # Perform Hough transform to detect lines
    angles = np.pi * np.arange(-limit, limit, delta) / 180.0
    h, theta, d = hough_line(edges, angles)

    # Find peaks in Hough space
    _, angles, dists = hough_line_peaks(h, theta, d, num_peaks=5)

    # Calculate average angle from detected lines
    rotation_angle = np.degrees(np.mean(angles))

    if plot_results:
        fig, ax = plt.subplots(1)
        ax.imshow(image, cmap="gray")

        for i in range(len(angles)):
            (x0, y0) = dists[i] * np.array([np.cos(angles[i]), np.sin(angles[i])])
            ax.axline((x0, y0), slope=np.tan(angles[i] + np.pi / 2))

        plt.tight_layout()

    ali = cast("TomoStack", stack.trans_stack(angle=-rotation_angle))
    tomo_meta = cast(Dtb, ali.metadata.Tomography)
    tomo_meta.tiltaxis = -rotation_angle

    if also_shift:
        idx = ali.data.shape[2] // 2
        shifts = np.arange(-shift_limit, shift_limit, 1)
        nshifts = shifts.shape[0]
        shifted = ali.isig[0:nshifts, :].deepcopy()
        for i in range(nshifts):
            shifted.data[:, :, i] = np.roll(
                ali.isig[idx:idx+1, :].data.squeeze(),
                int(shifts[i]),
            )
        shifted_rec = shifted.reconstruct("SIRT", 100, constrain=True)
        image_sum = cast(BaseSignal, shifted_rec.sum(axis=(1, 2)))
        tilt_shift = shifts[image_sum.data.argmin()]
        tilt_shift = cast(float, tilt_shift)
        ali = cast("TomoStack", ali.trans_stack(yshift=-tilt_shift))
        tomo_meta.yshift = -tilt_shift
    return ali


def align_to_other(stack: "TomoStack", other: "TomoStack") -> "TomoStack":
    """
    Spatially register a TomoStack using previously calculated shifts.

    Parameters
    ----------
    stack
        TomoStack which was previously aligned
    other
        TomoStack to be aligned. Must be the same size as the primary stack

    Returns
    -------
    out : TomoStack
        Aligned copy of other TomoStack

    Group
    -----
    align
    """
    out = copy.deepcopy(other)
    stack_tomo_meta = cast(Dtb, stack.metadata.Tomography)
    out_tomo_meta = cast(Dtb, out.metadata.Tomography)

    out.shifts = np.zeros([out.data.shape[0], 2])

    tiltaxis = cast(float, stack_tomo_meta.tiltaxis)
    out_tomo_meta.tiltaxis = tiltaxis

    xshift = cast(float, stack_tomo_meta.xshift)
    out_tomo_meta.xshift = stack_tomo_meta.xshift

    yshift = cast(float, stack_tomo_meta.yshift)
    out_tomo_meta.yshift = stack_tomo_meta.yshift

    out = apply_shifts(out, stack.shifts)

    if stack_tomo_meta.cropped:
        out = shift_crop(out)

    out = cast("TomoStack", out.trans_stack(xshift, yshift, tiltaxis))

    logger.info("TomoStack alignment applied")
    logger.info("X-shift: %.1f", xshift)
    logger.info("Y-shift: %.1f", yshift)
    logger.info("Rotation: %.1f", tiltaxis)
    return out


def shift_crop(stack: "TomoStack") -> "TomoStack":
    """
    Crop shifted stack to common area.

    Parameters
    ----------
    stack
        TomoStack which was previously aligned

    Returns
    -------
    out : TomoStack
        Aligned copy of other TomoStack

    Group
    -----
    align
    """
    cropped = stack.deepcopy()
    x_shifts = stack.shifts.data[:, 0]
    y_shifts = stack.shifts.data[:, 1]
    x_max = np.int32(np.floor(x_shifts.min()))
    x_min = np.int32(np.ceil(x_shifts.max()))
    y_max = np.int32(np.floor(y_shifts.min()))
    y_min = np.int32(np.ceil(y_shifts.max()))
    cropped = cropped.isig[x_min:x_max, y_min:y_max]
    cropped.metadata.set_item("Tomography.cropped", value=True)
    return cropped
