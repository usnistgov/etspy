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
import pylab as plt
import warnings
import tqdm
from pystackreg import StackReg
from scipy.ndimage import center_of_mass
import logging
from numpy.fft import fft, fftshift, ifftshift, ifft
from skimage.registration import phase_cross_correlation as pcc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        raise ValueError('Number of shifts (%s) is not consistent with number'
                         'of images in the stack (%s)'
                         % (len(shifts), stack.data.shape[0]))
    for i in range(0, shifted.data.shape[0]):
        shifted.data[i, :, :] =\
            ndimage.shift(shifted.data[i, :, :],
                          shift=[shifts[i, 1], shifts[i, 0]])
    shifted.metadata.Tomography.shifts = \
        shifted.metadata.Tomography.shifts + shifts
    return shifted


def compose_shifts(shifts, start=None):
    """

    Compose a series of calculated shifts.

    Args
    ----------
    shifts : NumPy array
        The X- and Y-shifts to be composed
    start : int
        The image index at which the alignment should start. If None,
        the mid-point of the stack will be used.

    Returns
    ----------
    composed : NumPy array
        Composed shifts

    """
    if start is None:
        start = np.int32(np.floor((shifts.shape[0] + 1) / 2))
    composed = np.zeros([shifts.shape[0] + 1, 2])
    composed[start, :] = [0., 0.]
    for i in range(start + 1, composed.shape[0]):
        composed[i, :] = composed[i - 1, :] - shifts[i - 1, :]
    for i in range(start - 1, -1, -1):
        composed[i, :] = composed[i + 1, :] + shifts[i]
    return composed


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
    padded = np.zeros(paddedsize)
    npix = len(line)
    if np.mod(npix, 2) == 0:
        if np.mod(paddedsize, 2) == 0:
            start_idx = (paddedsize - npix) / 2
            end_idx = npix + (paddedsize - npix) / 2
        else:
            start_idx = (paddedsize - npix - 1) / 2
            end_idx = npix + (paddedsize - npix - 1) / 2
    else:
        if np.mod(paddedsize, 2) == 0:
            start_idx = (paddedsize - npix + 1) / 2
            end_idx = npix + (paddedsize - npix + 1) / 2
        else:
            start_idx = (paddedsize - npix) / 2
            end_idx = npix + (paddedsize - npix) / 2
    padded[int(start_idx):int(end_idx)] = line
    return padded


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
        npix = np.shape(ref_line)[0]
        npad = npix * 2 - 1

        # Pad with zeros while preserving the center location
        ref_line_pad = pad_line(ref_line, npad)
        line_pad = pad_line(line, npad)

        niters = np.int32(
            np.abs(np.floor(np.log(cl_resolution) / np.log(cl_div_factor))))
        start = -0.5
        end = 0.5

        midpoint = (npad - 1) / 2
        kx = np.arange(-midpoint, midpoint + 1)

        ref_line_pad_FT = fftshift(fft(ifftshift(ref_line_pad)))
        line_pad_FT = fftshift(fft(ifftshift(line_pad)))

        for i in range(0, niters):
            boundary = np.arange(start, end, (end - start) / cl_div_factor)
            index = (np.roll(boundary, -1) + boundary) / 2
            index = index[:-1]

            max_vals = np.zeros(len(index))
            for j in range(0, len(index)):
                pfactor = np.exp(2 * np.pi * 1j * (index[j] * kx / npad))
                conjugate = np.conj(ref_line_pad_FT) * line_pad_FT * pfactor
                xcorr = np.abs(fftshift(ifft(ifftshift(conjugate))))
                max_vals[j] = np.max(xcorr)

            max_loc = np.argmax(max_vals)
            start = boundary[max_loc]
            end = boundary[max_loc + 1]

        subpixel_shift = index[max_loc]
        max_pfactor = np.exp(2 * np.pi * 1j * (index[max_loc] * kx / npad))

        # Determine integer shift via cross correlation
        conjugate = np.conj(ref_line_pad_FT) * line_pad_FT * max_pfactor
        xcorr = np.abs(fftshift(ifft(ifftshift(conjugate))))
        max_loc = np.argmax(xcorr)

        integer_shift = max_loc
        integer_shift = integer_shift - midpoint

        # Calculate full shift
        shift = integer_shift + subpixel_shift

        return -shift

    if not cl_ref_index:
        cl_ref_index = round(stack.data.shape[0] / 2)

    aliY = stack.deepcopy()
    yshifts = np.zeros(stack.data.shape[0])
    ref_cm_line = stack.data[cl_ref_index].sum(0)

    for i in tqdm.tqdm(range(0, stack.data.shape[0])):
        if i == cl_ref_index:
            aliY.data[i] = stack.data[i]
        else:
            curr_cm_line = stack.data[i].sum(0)
            yshifts[i] = align_line(ref_cm_line, curr_cm_line,
                                    cl_resolution, cl_div_factor)
    return yshifts


def calculate_shifts_com(stack, nslice, ratio):
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
    if not nslice:
        nslice = np.int32(stack.data.shape[2] / 2)

    stack = stack.stack_register('StackReg')
    logger.info("Refinining X-shifts using center of mass method")
    sino = np.transpose(stack.isig[nslice:nslice + 1, :].data,
                        axes=[0, 2, 1])

    angles = stack.metadata.Tomography.tilts
    [ntilts, ydim, xdim] = sino.shape
    angles = angles * np.pi / 180

    t = np.zeros([ntilts, 1, ydim])
    ss = np.zeros([ntilts, 1, ydim])
    w = np.arange(1, xdim + 1).T

    for i in range(0, ydim):
        for k in range(0, ntilts):
            ss[k, 0, i] = np.sum(sino[k, i, :])
            t[k, 0, i] = np.sum(sino[k, i, :] * w) / ss[k, 0, i]

    t = t - (xdim + 1) / 2
    ss2 = np.median(ss)

    for k in range(0, ntilts):
        ss[k, :, :] = np.abs((ss[k, :, :] - ss2) / ss2)
    ss2 = np.mean(ss, 0)

    num = round(ratio * ydim)
    if num == 0:
        num = 1
    usables = np.zeros([num, 1])
    t_select = np.zeros([ntilts * num, 1])
    disp_mat = np.zeros([ydim, 1])

    s3 = np.argsort(ss2[0, :])
    usables = np.reshape(s3[0:num], [num, 1])
    t_select[:, 0] = np.reshape(t[:, 0, np.int32(usables[:, 0])],
                                [ntilts * num])
    disp_mat[np.int32(usables[:, 0]), 0] = 1

    I_tilts = np.eye(ntilts)
    A = np.zeros([ntilts * num, ntilts])

    theta = angles
    Gam = (np.array([np.cos(theta), np.sin(theta)])).T
    Gam = np.dot(Gam, np.linalg.pinv(Gam)) - I_tilts
    for j in range(0, num):
        t_select[ntilts * j:ntilts * (j + 1), 0] =\
            np.dot(-Gam, t_select[ntilts * j:ntilts * (j + 1), 0])
        A[ntilts * j:ntilts * (j + 1), 0:ntilts] = Gam

    shifts = np.zeros([stack.data.shape[0], 2])
    shifts[:, 1] = np.dot(np.linalg.pinv(A), t_select)[:, 0]

    shifts = stack.metadata.Tomography.shifts + shifts
    return shifts


def calculate_shifts_pc(stack, start, show_progressbar):
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
    def calc_pc(source, shifted):
        shift = pcc(shifted, source, upsample_factor=3)
        return shift[0][::-1]

    shifts = np.zeros([stack.data.shape[0] - 1, 2])
    if start is None:
        start = np.int32(np.floor(stack.data.shape[0] / 2))

    for i in tqdm.tqdm(range(start, stack.data.shape[0] - 1),
                       disable=(not show_progressbar)):
        shifts[i, :] = calc_pc(stack.data[i, :, :],
                               stack.data[i + 1, :, :])

    if start != 0:
        for i in tqdm.tqdm(range(start - 1, -1, -1),
                           disable=(not show_progressbar)):
            shifts[i, :] = calc_pc(stack.data[i, :, :],
                                   stack.data[i + 1, :, :])
    return shifts


def calculate_shifts_stackreg(stack):
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
    sr = StackReg(StackReg.TRANSLATION)
    shifts = sr.register_stack(stack.data, reference='previous')
    shifts = -np.array([i[0:2, 2] for i in shifts])
    return shifts


def calc_com_cl_shifts(stack, com_ref_index, cl_ref_index, cl_resolution,
                       cl_div_factor):
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

    def calc_xshifts(stack, com_ref=None):
        ntilts = stack.data.shape[0]
        aliX = stack.deepcopy()
        coms = np.zeros(ntilts)
        xhifts = np.zeros_like(coms)

        for i in tqdm.tqdm(range(0, ntilts)):
            im = aliX.data[i, :, :]
            coms[i], _ = ndimage.center_of_mass(im)
            xhifts[i] = com_ref - coms[i]
        return xhifts

    if cl_resolution >= 0.5:
        raise ValueError("Resolution should be less than 0.5")
    if not com_ref_index:
        com_ref_index = round(stack.data.shape[1] / 2)
    logger.info("Center of mass reference slice: %s" % com_ref_index)
    logger.info("Common line reference slice: %s" % cl_ref_index)
    xshifts = np.zeros(stack.data.shape[0])
    yshifts = np.zeros(stack.data.shape[0])
    xshifts = calc_xshifts(stack, com_ref_index)
    yshifts = calc_shifts_cl(stack, cl_ref_index,
                             cl_resolution, cl_div_factor)
    shifts = np.stack([yshifts, xshifts], axis=1)
    return shifts


def align_stack(stack, method, start, show_progressbar, nslice, ratio,
                cl_ref_index, com_ref_index, cl_resolution, cl_div_factor):
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
        are 'PC', 'COM', or 'COM-CL'.
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
    method = method.lower()
    if method == 'com':
        shifts = calculate_shifts_com(stack, nslice, ratio)
    elif method == 'pc':
        logger.info("Performing stack registration using "
                    "phase correlation method")
        shifts = calculate_shifts_pc(stack, start, show_progressbar)
    elif method == 'stackreg':
        logger.info("Performing stack registration using PyStackReg")
        shifts = calculate_shifts_stackreg(stack)
    elif method == 'com-cl':
        logger.info("Performing stack registration using "
                    "combined center of mass and common line methods")
        shifts = calc_com_cl_shifts(stack, com_ref_index, cl_ref_index,
                                    cl_resolution, cl_div_factor)
    if method == 'pc':
        shifts = compose_shifts(shifts, start)
    aligned = apply_shifts(stack, shifts)
    logger.info("Stack registration complete")
    return aligned


def tilt_com(stack, locs=None, interactive=False):
    """
    Perform tilt axis alignment using center of mass (CoM) tracking.

    Compares path of specimen to the path expected for an ideal cylinder

    Args
    ----------
    stack : TomoStack object
        3-D numpy array containing the tilt series data
    locs : list
        Locations at which to perform the CoM analysis

    Returns
    ----------
    out : TomoStack object
        Copy of the input stack after rotation and translation to center and
        make the tilt axis vertical

    """
    def com_motion(theta, r, x0, z0):
        return r - x0 * np.cos(theta) - z0 * np.sin(theta)

    def get_coms(stack, nslice):
        sino = stack.isig[nslice, :].deepcopy().data
        coms = [center_of_mass(sino[i, :])[0] for i in range(0, sino.shape[0])]
        return np.array(coms)

    def fit_line(x, m, b):
        return m * x + b

    def shift_stack(stack, shifts):
        shifted = stack.deepcopy()
        for i in range(0, stack.data.shape[0]):
            shifted.data[i, :, :] = ndimage.shift(stack.data[i, :, :],
                                                  [shifts[i], 0])
        return shifted

    def calc_shifts(stack, nslice):
        # Convert tilts to rads
        thetas = np.pi * stack.metadata.Tomography.tilts / 180.

        # Calculate centers of mass for for each row of the sinogram
        coms = get_coms(stack, nslice)

        # Fit measured centers of mass with function describing
        # expected motion of a cylinder
        r, x0, z0 = optimize.curve_fit(com_motion, xdata=thetas,
                                       ydata=coms, p0=[0, 0, 0])[0]

        # Determine shifts to align centers of mass to cylindrical pathway
        shifts = com_motion(thetas, r, x0, z0) - coms
        return shifts, coms

    def tilt_analyze(stack, slices):
        # Convert tilts to rads
        thetas = np.pi * stack.metadata.Tomography.tilts / 180.
        r = np.zeros(len(slices))
        x0 = np.zeros(len(slices))
        z0 = np.zeros(len(slices))
        for i in range(0, len(slices)):
            coms = get_coms(stack, slices[i])
            r[i], x0[i], z0[i] = optimize.curve_fit(com_motion,
                                                    xdata=thetas,
                                                    ydata=coms,
                                                    p0=[0, 0, 0])[0]
        slope, intercept = optimize.curve_fit(fit_line,
                                              xdata=r,
                                              ydata=slices,
                                              p0=[0, 0])[0]
        tilt_shift = stack.data.shape[1] / 2\
            - (stack.data.shape[1] / 2 - intercept)\
            / slope
        rotation = 180 * np.arctan(1 / slope) / np.pi
        return -tilt_shift, -rotation, r

    data = stack.deepcopy()
    if locs is None:
        if interactive:
            """Prompt user for locations at which to fit the CoM"""
            warnings.filterwarnings('ignore')
            plt.figure(num='Align Tilt', frameon=False)
            if len(data.data.shape) == 3:
                plt.imshow(data.data[np.int(data.data.shape[0] / 2), :, :],
                           cmap='gray')
            else:
                plt.imshow(data, cmap='gray')
            plt.title('Choose %s points for tilt axis alignment....' %
                      str(3))
            coords = np.array(plt.ginput(3, timeout=0, show_clicks=True))
            plt.close()
            locs = np.int16(np.sort(coords[:, 0]))
        else:
            locs = np.int16(stack.data.shape[2] * np.array([0.33, 0.5, 0.67]))
            logger.info("Performing alignments using slices: [%s, %s, %s]"
                        % (locs[0], locs[1], locs[2]))
    else:
        locs = np.int16(np.sort(locs))

    if stack.metadata.Tomography.tilts is None:
        raise ValueError("No tilts in stack.metadata.Tomography.")

    shifts, coms = calc_shifts(stack, locs[1])
    shifted = shift_stack(stack, shifts)
    shifted.metadata.Tomography.shifts[:, 0] = \
        shifted.metadata.Tomography.shifts[:, 0] - shifts
    tilt_shift, tilt_rotation, r = tilt_analyze(stack, locs)

    final = shifted.trans_stack(yshift=tilt_shift, angle=tilt_rotation)

    logger.info("Calculated tilt-axis shift %.2f" % tilt_shift)
    logger.info("Calculated tilt-axis rotation %.2f" % tilt_rotation)

    final.metadata.Tomography.tiltaxis = tilt_rotation
    final.metadata.Tomography.xshift = tilt_shift
    return final


def tilt_maximage(data, limit=10, delta=0.3, show_progressbar=False):
    """
    Perform automated determination of the tilt axis of a TomoStack.

    The projected maximum image by is rotated positively and negatively,
    filtered using a Hamming window, and the rotation angle is determined by
    iterative histogram analysis

    Args
    ----------
    data : TomoStack object
        3-D numpy array containing the tilt series data
    limit : integer or float
        Maximum rotation angle to use for MaxImage calculation
    delta : float
        Angular increment for MaxImage calculation
    show_progressbar : boolean
        Enable/disable progress bar

    Returns
    ----------
    opt_angle : TomoStack object
        Calculated rotation to set the tilt axis vertical

    """
    def hamming(img):
        """
        Apply hamming window to the image to remove edge effects.

        Args
        ----------
        img : Numpy array
            Input image
        Returns
        ----------
        out : Numpy array
            Filtered image

        """
        # if img.shape[0] < img.shape[1]:
        #     center_loc = np.int32((img.shape[1] - img.shape[0]) / 2)
        #     img = img[:, center_loc:-center_loc]
        #     if img.shape[0] != img.shape[1]:
        #         img = img[:, 0:-1]
        #     h = np.hamming(img.shape[0])
        #     ham2d = np.sqrt(np.outer(h, h))
        # elif img.shape[1] < img.shape[0]:
        #     center_loc = np.int32((img.shape[0] - img.shape[1]) / 2)
        #     img = img[center_loc:-center_loc, :]
        #     if img.shape[0] != img.shape[1]:
        #         img = img[0:-1, :]
        #     h = np.hamming(img.shape[1])
        #     ham2d = np.sqrt(np.outer(h, h))
        # else:
        h = np.hamming(img.shape[0])
        ham2d = np.sqrt(np.outer(h, h))
        out = ham2d * img
        return out

    def find_score(im, angle):
        """
        Perform histogram analysis to measure the rotation angle.

        Args
        ----------
        im : Numpy array
            Input image
        angle : float
            Angle by which to rotate the input image before analysis

        Returns
        ----------
        hist : Numpy array
            Result of integrating image along the vertical axis
        score : numpy array
            Score calculated from hist

        """
        im = ndimage.rotate(im, angle, reshape=False, order=3)
        hist = np.sum(im, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score

    image = np.max(data.data, 0)

    if image.shape[0] != image.shape[1]:
        raise ValueError("Invalid data shape. Currently only square signal dimensions are supported.")
    rot_pos = ndimage.rotate(hamming(image), -limit / 2,
                             reshape=False, order=3)
    rot_neg = ndimage.rotate(hamming(image), limit / 2,
                             reshape=False, order=3)
    angles = np.arange(-limit, limit + delta, delta)
    scores_pos = []
    scores_neg = []
    for rotation_angle in tqdm.tqdm(angles, disable=(not show_progressbar)):
        hist_pos, score_pos = find_score(rot_pos, rotation_angle)
        hist_neg, score_neg = find_score(rot_neg, rotation_angle)
        scores_pos.append(score_pos)
        scores_neg.append(score_neg)

    best_score_pos = max(scores_pos)
    best_score_neg = max(scores_neg)
    pos_angle = -angles[scores_pos.index(best_score_pos)]
    neg_angle = -angles[scores_neg.index(best_score_neg)]
    opt_angle = (pos_angle + neg_angle) / 2

    logger.info('Optimum positive rotation angle: {}'.format(pos_angle))
    logger.info('Optimum negative rotation angle: {}'.format(neg_angle))
    logger.info('Optimum positive rotation angle: {}'.format(opt_angle))

    out = copy.deepcopy(data)
    out = out.trans_stack(xshift=0, yshift=0, angle=opt_angle)
    out.data = np.transpose(out.data, (0, 2, 1))
    out.metadata.Tomography.tiltaxis = opt_angle
    return out


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

    logger.info('TomoStack alignment applied')
    logger.info('X-shift: %.1f' % xshift)
    logger.info('Y-shift: %.1f' % yshift)
    logger.info('Rotation: %.1f' % tiltaxis)
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
