# -*- coding: utf-8 -*-
#
# This file is part of TomoTools

"""
Alignment module for TomoTools package.

@author: Andrew Herzing
"""

import numpy as np
import copy
from scipy import optimize, ndimage
import tqdm
from pystackreg import StackReg
import logging
from skimage.registration import phase_cross_correlation as pcc
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.filters import sobel
import matplotlib.pylab as plt
import astra

has_cupy = True
try:
    import cupy as cp
except ImportError:
    has_cupy = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_best_slices(stack, nslices):
    """
    Get best nslices for center of mass analysis.

    Slices which have the highest ratio of total mass to mass variance
    and their location are returned.

    Args
    ----------
    stack : TomoStack object
        Tilt series from which to select the best slices.
    nslices : integer
        Number of slices to return.

    Returns
    ----------
    locs : NumPy array
        Location along the x-axis of the best slices

    """
    total_mass = stack.data.sum((0, 1))
    mass_std = stack.data.sum(1).std(0)
    mass_std[mass_std == 0] = 1e-5
    mass_ratio = total_mass / mass_std
    best_slice_locations = mass_ratio.argsort()[::-1][0:nslices]
    return best_slice_locations


def get_coms(stack, slices):
    """
    Calculate the center of mass for indicated slices.

    Args
    ----------
    stack : TomoStack object
        Tilt series from which to calculate the centers of mass.
    slices : NumPy array
        Location of slices to use for center of mass calculation.

    Returns
    ----------
    coms : NumPy array
        Center of mass as a function of tilt for each slice [ntilts, nslices].

    """
    sinos = stack.data[:, :, slices]
    com_range = int(sinos.shape[1] / 2)
    y_coordinates = np.linspace(-com_range,
                                com_range,
                                sinos.shape[1], dtype="int")
    total_mass = sinos.sum(1)
    coms = np.sum(np.transpose(sinos, [0, 2, 1]) * y_coordinates, 2) / total_mass
    return coms


def apply_shifts(stack, shifts):
    """

    Apply a series of shifts to a TomoStack.

    Args
    ----------
    stack : TomoStack object
        The image series to be aligned
    shifts : NumPy array
        The X- and Y-shifts to be applied to each image

    Returns
    ----------
    shifted : TomoStack object
        Copy of input stack after shifts are applied

    """
    shifted = stack.deepcopy()
    if len(shifts) != stack.data.shape[0]:
        raise ValueError(
            "Number of shifts (%s) is not consistent with number"
            "of images in the stack (%s)" % (len(shifts), stack.data.shape[0])
        )
    for i in range(shifted.data.shape[0]):
        shifted.data[i, :, :] = ndimage.shift(
            shifted.data[i, :, :], shift=[shifts[i, 0], shifts[i, 1]]
        )
    shifted.metadata.Tomography.shifts = shifted.metadata.Tomography.shifts + shifts
    return shifted


def pad_line(line, paddedsize):
    """

    Pad a 1D array for FFT treatment without altering center location.

    Args
    ----------
    line : 1D NumPy array
        The data to be padded
    paddedsize : int
        The size of the desired padded data.

    Returns
    ----------
    padded : 1D NumPy array
        Padded version of input data

    """
    npix = len(line)
    start_index = (paddedsize - npix) // 2
    end_index = start_index + npix
    padded_line = np.zeros(paddedsize)
    padded_line[start_index:end_index] = line
    return padded_line


def calc_shifts_cl(stack, cl_ref_index, cl_resolution, cl_div_factor):
    """

    Calculate shifts using the common line method.

    Used to align stack in dimension parallel to the tilt axis

    Args
    ----------
    stack : TomoStack object
        The stack on which to calculate shifts
    cl_ref_index : int
        Tilt index of reference projection. If not provided the projection
        closest to the middle of the stack will be chosen.
    cl_resolution : float
        Degree of sub-pixel analysis
    cl_div_factor : int
        Factor used to determine number of iterations of alignment.

    Returns
    ----------
    yshifts : NumPy array
        Shifts parallel to tilt axis for each projection

    """

    def align_line(ref_line, line, cl_resolution, cl_div_factor):
        npad = len(ref_line) * 2 - 1

        # Pad with zeros while preserving the center location
        ref_line_pad = pad_line(ref_line, npad)
        line_pad = pad_line(line, npad)

        niters = int(np.abs(np.floor(np.log(cl_resolution) / np.log(cl_div_factor))))
        start, end = -0.5, 0.5

        ref_line_pad_FT = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(ref_line_pad)))
        line_pad_FT = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(line_pad)))

        midpoint = (npad - 1) / 2
        kx = np.arange(-midpoint, midpoint + 1)

        for i in range(niters):
            boundary = np.linspace(start, end, cl_div_factor, False)
            index = (boundary[:-1] + boundary[1:]) / 2

            max_vals = np.zeros(len(index))
            for j, idx in enumerate(index):
                pfactor = np.exp(2 * np.pi * 1j * (idx * kx / npad))
                conjugate = np.conj(ref_line_pad_FT) * line_pad_FT * pfactor
                xcorr = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(conjugate))))
                max_vals[j] = np.max(xcorr)

            max_loc = np.argmax(max_vals)
            start, end = boundary[max_loc], boundary[max_loc + 1]

        subpixel_shift = index[max_loc]
        max_pfactor = np.exp(2 * np.pi * 1j * (subpixel_shift * kx / npad))

        # Determine integer shift via cross correlation
        conjugate = np.conj(ref_line_pad_FT) * line_pad_FT * max_pfactor
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


def calculate_shifts_conservation_of_mass(stack, xrange=None, p=20):
    """
    Calculate shifts parallel to the tilt axis using conservation of mass.

    Slices which have the highest ratio of total mass to mass variance
    and their location are returned.

    Args
    ----------
    stack : TomoStack object
        Tilt series to be aligned.
    xrange : tuple
        Defines range for performing alignment.
    p : int
        Padding element

    Returns
    ----------
    xshifts : NumPy array
        Calculated shifts parallel to tilt axis.

    """
    logger.info("Refinining X-shifts using conservation of mass method")
    [ntilts, ny, nx] = stack.data.shape

    if xrange is None:
        xrange = [round(nx / 5), round(4 / 5 * nx)]
    else:
        xrange = [round(xrange[0]) + p, round(xrange[1]) - p]

    xshifts = np.zeros([ntilts, 1])
    total_mass = np.zeros([ntilts, xrange[1] - xrange[0] + 2 * p + 1])

    for i in range(0, ntilts):
        total_mass[i, :] = np.sum(stack.data[i, :, xrange[0] - p - 1: xrange[1] + p], 0)

    mean_mass = np.mean(total_mass[:, p:-p], 0)

    for i in range(0, ntilts):
        s = 0
        for j in range(-p, p):
            resid = np.linalg.norm(mean_mass - total_mass[i, p + j: -p + j])
            if resid < s or j == -p:
                s = resid
                xshifts[i] = -j
    return xshifts[:, 0]


def calculate_shifts_com(stack, nslices):
    """
    Align stack using a center of mass method.

    Data is first registered using PyStackReg. Then, the shifts
    perpendicular to the tilt axis are refined by a center of
    mass analysis.

    Args
    ----------
    stack : TomoStack object
        The image series to be aligned
    nslice : integer
        Slice to use for the center of mass analysis.  If None, the slice
        nearest the center of the tilt series will be used.
    ratio : float
        Value that determines the number of projections to use for the
        center of mass analysis.  Must be less than or equal to 1.0.

    Returns
    ----------
    shifts : NumPy array
        The X- and Y-shifts to be applied to each image

    """
    logger.info("Refinining Y-shifts using center of mass method")
    slices = get_best_slices(stack, nslices)

    angles = stack.axes_manager[0].axis
    [ntilts, ydim, xdim] = stack.data.shape
    thetas = np.pi * angles / 180

    coms = get_coms(stack, slices)
    I_tilts = np.eye(ntilts)
    Gam = np.array([np.cos(thetas), np.sin(thetas)]).T
    Gam = np.dot(Gam, np.linalg.pinv(Gam)) - I_tilts
    b = np.dot(Gam, coms)

    cx = np.linalg.lstsq(Gam, b, rcond=-1)[0]

    yshifts = -cx[:, 0]
    return yshifts


def calculate_shifts_pc(stack, start, show_progressbar, upsample_factor, cuda):
    """

    Calculate shifts using the phase correlation algorithm.

    Args
    ----------
    stack : TomoStack object
        The image series to be aligned

    Returns
    ----------
    shifts : NumPy array
        The X- and Y-shifts to be applied to each image

    """
    def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
        upsampled_region_size = [upsampled_region_size,] * data.ndim

        im2pi = 1j * 2 * cp.pi

        dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

        for n_items, ups_size, ax_offset in dim_properties[::-1]:
            kernel = (cp.arange(ups_size) - ax_offset)[:, None] * cp.fft.fftfreq(n_items, upsample_factor)
            kernel = cp.exp(-im2pi * kernel)
            # use kernel with same precision as the data
            kernel = kernel.astype(data.dtype, copy=False)
            data = cp.tensordot(kernel, data, axes=(1, -1))
        return data

    def _cupy_phase_correlate(ref_cp, mov_cp, upsample_factor):
        ref_fft = cp.fft.fftn(ref_cp)
        mov_fft = cp.fft.fftn(mov_cp)

        cross_power_spectrum = ref_fft * mov_fft.conj()
        eps = cp.finfo(cross_power_spectrum.real.dtype).eps
        cross_power_spectrum /= cp.maximum(cp.abs(cross_power_spectrum), 100 * eps)
        phase_correlation = cp.fft.ifft2(cross_power_spectrum)

        maxima = cp.unravel_index(cp.argmax(cp.abs(phase_correlation)), phase_correlation.shape)
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
            phase_correlation = _upsampled_dft(cross_power_spectrum.conj(), upsampled_region_size, upsample_factor, sample_region_offset).conj()
            maxima = np.unravel_index(cp.argmax(np.abs(phase_correlation)), phase_correlation.shape)

            maxima = cp.stack(maxima).astype(float_dtype, copy=False)
            maxima -= dftshift

            shift += maxima / upsample_factor
        return shift

    if has_cupy and astra.use_cuda() and cuda:
        stack_cp = cp.array(stack.data)
        shifts = cp.zeros([stack_cp.shape[0], 2])
        ref_cp = stack_cp[0]
        ref_fft = cp.fft.fftn(ref_cp)
        shape = ref_fft.shape
        with tqdm.tqdm(total=stack.data.shape[0] - 1, desc="Calculating shifts", disable=not show_progressbar) as pbar:
            for i in range(start, 0, -1):
                shift = _cupy_phase_correlate(stack_cp[i], stack_cp[i - 1], upsample_factor=upsample_factor)
                shifts[i - 1] = shifts[i] + shift
                pbar.update(1)
            for i in range(start, stack.data.shape[0] - 1):
                shift = _cupy_phase_correlate(stack_cp[i], stack_cp[i + 1], upsample_factor=upsample_factor)
                shifts[i + 1] = shifts[i] + shift
                pbar.update(1)
        shifts = shifts.get()

    else:
        shifts = np.zeros((stack.data.shape[0], 2))
        with tqdm.tqdm(total=stack.data.shape[0] - 1, desc="Calculating shifts", disable=not show_progressbar) as pbar:
            for i in range(start, 0, -1):
                shift = pcc(stack.data[i], stack.data[i - 1], upsample_factor=upsample_factor)[0]
                shifts[i - 1] = shifts[i] + shift
                pbar.update(1)

            for i in range(start, stack.data.shape[0] - 1):
                shift = pcc(stack.data[i], stack.data[i + 1], upsample_factor=upsample_factor)[0]
                shifts[i + 1] = shifts[i] + shift
                pbar.update(1)

    return shifts


def calculate_shifts_stackreg(stack, start, show_progressbar):
    """
    Calculate shifts using PyStackReg.

    Args
    ----------
    stack : TomoStack object
        The image series to be aligned

    Returns
    ----------
    shifts : NumPy array
        The X- and Y-shifts to be applied to each image

    """
    shifts = np.zeros((stack.data.shape[0], 2))

    if start is None:
        start = stack.data.shape[0] // 2  # Use the midpoint if start is not provided

    # Initialize pystackreg object with TranslationTransform2D
    reg = StackReg(StackReg.TRANSLATION)

    with tqdm.tqdm(total=stack.data.shape[0] - 1, desc="Calculating shifts", disable=not show_progressbar) as pbar:
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


def calc_shifts_com_cl(stack, com_ref_index, cl_ref_index, cl_resolution, cl_div_factor):
    """
    Align stack using combined center of mass and common line methods.

    Center of mass aligns stack perpendicular to the tilt axis and
    common line is used to align the stack parallel to the tilt axis.

    Args
    ----------
    stack : TomoStack object
        Tilt series to be aligned
    com_ref_index : integer
        Reference slice for center of mass alignment.  All other slices
        will be aligned to this reference.  If not provided, the midpoint
        of the stack will be chosen.
    cl_ref_index : integer
        Reference slice for common line alignment.  All other slices
        will be aligned to this reference.  If not provided, the midpoint
        of the stack will be chosen.
    cl_resolution : float
        Resolution for subpixel common line alignment. Default is 0.05.
        Should be less than 0.5.
    cl_div_factor : integer
        Factor which determines the number of iterations of common line
        alignment to perform.  Default is 8.

    Returns
    ----------
    reg : TomoStack object
        Copy of stack after spatial registration.  Shift values are stored
        in reg.metadata.Tomography.shifts for later use.

    """

    def calc_yshifts(stack, com_ref):
        ntilts = stack.data.shape[0]
        aliX = stack.deepcopy()
        coms = np.zeros(ntilts)
        yshifts = np.zeros_like(coms)

        for i in tqdm.tqdm(range(0, ntilts)):
            im = aliX.data[i, :, :]
            coms[i], _ = ndimage.center_of_mass(im)
            yshifts[i] = com_ref - coms[i]
        return yshifts

    if cl_resolution >= 0.5:
        raise ValueError("Resolution should be less than 0.5")

    logger.info("Center of mass reference slice: %s" % com_ref_index)
    logger.info("Common line reference slice: %s" % cl_ref_index)
    xshifts = np.zeros(stack.data.shape[0])
    yshifts = np.zeros(stack.data.shape[0])
    yshifts = calc_yshifts(stack, com_ref_index)
    xshifts = calc_shifts_cl(stack, cl_ref_index, cl_resolution, cl_div_factor)
    shifts = np.stack([yshifts, xshifts], axis=1)
    return shifts


def align_stack(stack, method, start, show_progressbar, **kwargs):
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
            Nature 483, 444–447 (2012).
            https://doi.org/10.1038/nature10934

    Shifts are then applied and the aligned stack is returned.  The tilts are
    stored in stack.metadata.Tomography.shifts for later use.

    Args
    ----------
    stack : Numpy array
        3-D numpy array containing the tilt series data
    method : string
        Method by which to calculate the alignments. Valid options
        are 'StackReg', 'PC', 'COM', or 'COM-CL'.
    start : integer
        Position in tilt series to use as starting point for the alignment.
        If None, the central projection is used.
    show_progressbar : boolean
        Enable/disable progress bar

    Returns
    ----------
    out : TomoStack object
        Spatially registered copy of the input stack

    """
    if start is None:
        start = stack.data.shape[0] // 2  # Use the slice closest to the midpoint if start is not provided

    if method.lower() == "com":
        logger.info("Performing stack registration using center of mass method")
        xrange = kwargs.get('xrange', None)
        p = kwargs.get('p', 20)
        nslices = kwargs.get('nslices', 20)
        shifts = np.zeros([stack.data.shape[0], 2])
        shifts[:, 1] = calculate_shifts_conservation_of_mass(stack, xrange, p)
        shifts[:, 0] = calculate_shifts_com(stack, nslices)
    elif method.lower() == 'pc':
        cuda = kwargs.get('cuda', False)
        upsample_factor = kwargs.get('upsample_factor', 3)
        if cuda:
            logger.info("Performing stack registration using CUDA-accelerated phase correlation")
        else:
            logger.info("Performing stack registration using phase correlation")
        shifts = calculate_shifts_pc(stack, start, show_progressbar, upsample_factor, cuda)
    elif method.lower() in ["stackreg", 'sr']:
        logger.info("Performing stack registration using PyStackReg")
        shifts = calculate_shifts_stackreg(stack, start, show_progressbar)
    elif method.lower() == "com-cl":
        logger.info("Performing stack registration using combined center of mass and common line methods")
        com_ref_index = kwargs.get('com_ref_index', stack.data.shape[1] // 2)
        cl_ref_index = kwargs.get('cl_ref_index', stack.data.shape[0] // 2)
        cl_resolution = kwargs.get('cl_resolution', 0.05)
        cl_div_factor = kwargs.get('cl_div_factor', 8)
        shifts = calc_shifts_com_cl(stack, com_ref_index, cl_ref_index, cl_resolution, cl_div_factor)
    else:
        raise ValueError('Invalid alignment method %s' % method)
    aligned = apply_shifts(stack, shifts)
    logger.info("Stack registration complete")
    return aligned


def tilt_com(stack, slices=None, nslices=None):
    """
    Perform tilt axis alignment using center of mass (CoM) tracking.

    Compares path of specimen to the path expected for an ideal cylinder

    Args
    ----------
    stack : TomoStack object
        3-D numpy array containing the tilt series data
    slices : list
        Locations at which to perform the CoM analysis
    nslices : int
        Nubmer of slices to suer for the analysis

    Returns
    ----------
    out : TomoStack object
        Copy of the input stack after rotation and translation to center and
        make the tilt axis vertical

    """

    def com_motion(theta, r, x0, z0):
        return r - x0 * np.cos(theta) - z0 * np.sin(theta)

    def fit_line(x, m, b):
        return m * x + b

    _, ny, nx = stack.data.shape

    if stack.metadata.Tomography.tilts is None:
        logger.warning("Tilts are not defined in stack.metadata.Tomography.  Ensure that navigation axis is calibrated.")

    if nx < 3:
        raise ValueError("Dataset is only %s pixels in x dimension. This method cannot be used." % stack.data.shape[2])

    # Determine the best slice locations for the analysis
    if slices is None:
        if nslices is None:
            nslices = int(0.1 * nx)
            if nslices < 3:
                nslices = 3
            elif nslices > 50:
                nslices = 50
        else:
            if nslices > nx:
                raise ValueError("nslices is greater than the X-dimension of the data.")
            if nslices > 0.3 * nx:
                nslices = int(0.3 * nx)
                logger.warning("nslices is greater than 30%% of number of x pixels. Using %s slices instead." % nslices)

        slices = get_best_slices(stack, nslices)
        logger.info("Performing alignments using best %s slices" % nslices)

    slices = np.sort(slices)

    coms = get_coms(stack, slices)
    angles = stack.axes_manager[0].axis
    thetas = np.deg2rad(angles)

    r, x0, z0 = np.zeros(len(slices)), np.zeros(len(slices)), np.zeros(len(slices))

    for idx, i in enumerate(slices):
        r[idx], x0[idx], z0[idx] = optimize.curve_fit(com_motion, xdata=thetas, ydata=coms[:, idx], p0=[0, 0, 0])[0]
    slope, intercept = optimize.curve_fit(fit_line, xdata=r, ydata=slices, p0=[0, 0])[0]
    tilt_shift = (ny / 2 - intercept) / slope
    tilt_rotation = -(180 * np.arctan(1 / slope) / np.pi)

    final = stack.trans_stack(yshift=tilt_shift, angle=tilt_rotation)

    logger.info("Calculated tilt-axis shift %.2f" % tilt_shift)
    logger.info("Calculated tilt-axis rotation %.2f" % tilt_rotation)

    final.metadata.Tomography.tiltaxis = tilt_rotation
    final.metadata.Tomography.xshift = tilt_shift
    return final


def tilt_maximage(stack, limit=10, delta=0.1, plot_results=False):
    """
    Perform automated determination of the tilt axis of a TomoStack.

    The projected maximum image used to determine the tilt axis by a
    combination of Sobel filtering and Hough transform analysis.

    Args
    ----------
    stack : TomoStack object
        3-D numpy array containing the tilt series data
    limit : integer or float
        Maximum rotation angle to use for calculation
    delta : float
        Angular increment for calculation
    plot_results : boolean
        If True, plot the maximum image along with the lines determined
        by Hough analysis

    Returns
    ----------
    rotated : TomoStack object
        Rotated version of the input stack

    """
    image = stack.data.max(0)

    edges = sobel(image)

    # Apply Canny edge detector for further edge enhancement
    edges = canny(edges)

    # Perform Hough transform to detect lines
    angles = np.pi * np.arange(-limit, limit, delta) / 180.
    h, theta, d = hough_line(edges, angles)

    # Find peaks in Hough space
    _, angles, dists = hough_line_peaks(h, theta, d, num_peaks=5)

    # Calculate average angle from detected lines
    rotation_angle = np.degrees(np.mean(angles))

    if plot_results:
        fig, ax = plt.subplots(1)
        ax.imshow(image, cmap='gray')

        for i in range(len(angles)):
            (x0, y0) = dists[i] * np.array([np.cos(angles[i]), np.sin(angles[i])])
            ax.axline((x0, y0), slope=np.tan(angles[i] + np.pi / 2))

        plt.tight_layout()

    rotated = stack.trans_stack(angle=-rotation_angle)
    rotated.metadata.Tomography.tiltaxis = -rotation_angle
    return rotated


def align_to_other(stack, other):
    """
    Spatially register a TomoStack using previously calculated shifts.

    Args
    ----------
    stack : TomoStack object
        TomoStack which was previously aligned
    other : TomoStack object
        TomoStack to be aligned. Must be the same size as the primary stack

    Returns
    ----------
    out : TomoStack object
        Aligned copy of other TomoStack

    """
    out = copy.deepcopy(other)

    shifts = stack.metadata.Tomography.shifts
    out.metadata.Tomography.shifts = np.zeros([out.data.shape[0], 2])

    tiltaxis = stack.metadata.Tomography.tiltaxis
    out.metadata.Tomography.tiltaxis = tiltaxis

    xshift = stack.metadata.Tomography.xshift
    out.metadata.Tomography.xshift = stack.metadata.Tomography.xshift

    yshift = stack.metadata.Tomography.yshift
    out.metadata.Tomography.yshift = stack.metadata.Tomography.yshift

    out = apply_shifts(out, shifts)

    if stack.metadata.Tomography.cropped:
        out = shift_crop(out)

    out = out.trans_stack(xshift, yshift, tiltaxis)

    logger.info("TomoStack alignment applied")
    logger.info("X-shift: %.1f" % xshift)
    logger.info("Y-shift: %.1f" % yshift)
    logger.info("Rotation: %.1f" % tiltaxis)
    return out


def shift_crop(stack):
    """
    Crop shifted stack to common area.

    Args
    ----------
    stack : TomoStack object
        TomoStack which was previously aligned

    Returns
    ----------
    out : TomoStack object
        Aligned copy of other TomoStack

    """
    cropped = copy.deepcopy(stack)
    shifts = stack.metadata.Tomography.shifts
    x_shifts = shifts[:, 0]
    y_shifts = shifts[:, 1]
    x_max = np.int32(np.floor(x_shifts.min()))
    x_min = np.int32(np.ceil(x_shifts.max()))
    y_max = np.int32(np.floor(y_shifts.min()))
    y_min = np.int32(np.ceil(y_shifts.max()))
    cropped = cropped.isig[x_min:x_max, y_min:y_max]
    cropped.metadata.Tomography.cropped = True
    return cropped
