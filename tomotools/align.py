# -*- coding: utf-8 -*-
#
# This file is part of TomoTools

"""
Alignment module for TomoTools package.

@author: Andrew Herzing
"""

import numpy as np
import cv2
import copy
from scipy import optimize, ndimage
import pylab as plt
import warnings
import tqdm
from pystackreg import StackReg
from scipy.ndimage import center_of_mass
import logging

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
                          shift=[shifts[i, 1], shifts[i, 0]],
                          order=0)
    if not shifted.original_metadata.has_item('shifts'):
        shifted.original_metadata.add_node('shifts')
    shifted.original_metadata.shifts = shifts
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


def calculate_shifts_com(stack, nslice, ratio):
    """

    Calculate shifts using a center of mass method.

    Args
    ----------
    stack : TomoStack object
        The image series to be aligned

    Returns
    ----------
    shifts : NumPy array
        The X- and Y-shifts to be applied to each image

    """
    if not nslice:
        nslice = np.int32(stack.data.shape[2]/2)
    sino = np.transpose(stack.isig[nslice:nslice + 1, :].data,
                        axes=[0, 2, 1])
    angles = stack.axes_manager[0].axis

    [ntilts, ydim, xdim] = sino.shape
    angles = angles*np.pi/180

    t = np.zeros([ntilts, 1, ydim])
    ss = np.zeros([ntilts, 1, ydim])
    w = np.arange(1, xdim+1).T

    for i in range(0, ydim):
        for l in range(0, ntilts):
            ss[l, 0, i] = np.sum(sino[l, i, :])
            t[l, 0, i] = np.sum(sino[l, i, :] * w) / ss[l, 0, i]

    t = t-(xdim+1)/2
    ss2 = np.median(ss)

    for l in range(0, ntilts):
        ss[l, :, :] = np.abs((ss[l, :, :]-ss2)/ss2)
    ss2 = np.mean(ss, 0)

    num = round(ratio*ydim)
    if num == 0:
        num = 1
    usables = np.zeros([num, 1])
    t_select = np.zeros([ntilts*num, 1])
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
        t_select[ntilts*j:ntilts*(j+1), 0] =\
            np.dot(-Gam, t_select[ntilts * j:ntilts*(j+1), 0])
        A[ntilts*j:ntilts*(j+1), 0:ntilts] = Gam

    shifts = np.dot(np.linalg.pinv(A), t_select)
    return shifts


def calculate_shifts_ecc(stack, start, show_progressbar):
    """

    Calculate shifts using the enhanced correlation coefficient algorithm.

    Args
    ----------
    stack : TomoStack object
        The image series to be aligned

    Returns
    ----------
    shifts : NumPy array
        The X- and Y-shifts to be applied to each image

    """
    def calc_ecc(source, shifted, criteria):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        if np.int32(cv2.__version__.split('.')[0]) == 4:
            (cc, trans) = cv2.findTransformECC(
                np.float32(source),
                np.float32(shifted),
                warp_matrix,
                cv2.MOTION_TRANSLATION,
                criteria,
                inputMask=None,
                gaussFiltSize=5)
        else:
            (cc, trans) = cv2.findTransformECC(
                    np.float32(source),
                    np.float32(shifted),
                    warp_matrix,
                    cv2.MOTION_TRANSLATION,
                    criteria)
        shift = trans[:, 2]
        return shift

    number_of_iterations = 1000
    termination_eps = 1e-3
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)
    shifts = np.zeros([stack.data.shape[0] - 1, 2])
    if start is None:
        start = np.int32(np.floor(stack.data.shape[0] / 2))

    for i in tqdm.tqdm(range(start, stack.data.shape[0] - 1),
                       disable=(not show_progressbar)):
        shifts[i, :] = calc_ecc(stack.data[i, :, :],
                                stack.data[i + 1, :, :],
                                criteria)

    if start != 0:
        for i in tqdm.tqdm(range(start - 1, -1, -1),
                           disable=(not show_progressbar)):
            shifts[i, :] = calc_ecc(stack.data[i, :, :],
                                    stack.data[i + 1, :, :],
                                    criteria)
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
        shift = cv2.phaseCorrelate(source, shifted)
        return shift[0]

    shifts = np.zeros([stack.data.shape[0] - 1, 2])
    if start is None:
        start = np.int32(np.floor(stack.data.shape[0] / 2))
    for i in tqdm.tqdm(range(start, stack.data.shape[0] - 1),
                       disable=(not show_progressbar)):
        shifts[i, :] = calc_pc(np.float64(stack.data[i, :, :]),
                               np.float64(stack.data[i + 1, :, :]))
    else:
        for i in tqdm.tqdm(range(start - 1, -1, -1),
                           disable=(not show_progressbar)):
            shifts[i, :] = calc_pc(np.float64(stack.data[i, :, :]),
                                   np.float64(stack.data[i + 1, :, :]))
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


def align_stack(stack, method, start, show_progressbar, nslice, ratio):
    """
    Compute the shifts for spatial registration.

    Shifts are determined by one of three methods:
        1.) Phase correlation (PC) as implemented in OpenCV. OpenCV is
            described in:
            G. Bradski. The OpenCV Library, Dr. Dobb’s Journal of Software
            Tools vol. 120, pp. 122-125, 2000.
            https://docs.opencv.org/
        2.) Enhanced correlation coefficient (ECC) as implemented in OpenCV.
            OpenCV is described in:
            G. Bradski. The OpenCV Library, Dr. Dobb’s Journal of Software
            Tools vol. 120, pp. 122-125, 2000.
            https://docs.opencv.org/
        3.) Center of mass (COM) tracking.  A Python implementation of
            Matlab code described in:
            T. Sanders. Matlab imaging algorithms: Image reconstruction,
            restoration, and alignment, with a focus in tomography.
            http://www.toby-sanders.com/software ,
            https://doi.org/10.13140/RG.2.2.33492.60801
        4.) Rigid translation using PyStackReg for shift calculatiosn.
            PyStackReg is a Python port of the StackReg plugin for ImageJ
            which uses a pyramidal approach to minimize the least-squares
            difference in image intensity between a source and target image.
            StackReg is described in:
            P. Thevenaz, U.E. Ruttimann, M. Unser. A Pyramid Approach to
            Subpixel Registration Based on Intensity, IEEE Transactions
            on Image Processing vol. 7, no. 1, pp. 27-41, January 1998.
            https://doi.org/10.1109/83.650848

    Shifts are then applied and the aligned stack is returned.

    Args
    ----------
    stack : Numpy array
        3-D numpy array containing the tilt series data
    method : string
        Method by which to calculate the alignments. Valid options
        are 'PC', 'ECC', or 'COM'.
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
    if method == 'COM':
        shifts = calculate_shifts_com(stack, nslice, ratio)
        ali = stack.deepcopy()
        for i in range(0, stack.data.shape[0]):
            ali.data[i, :, :] = ndimage.shift(ali.data[i, :, :],
                                              [shifts[i], 0])

        ali.original_metadata.shifts = shifts
        return ali
    elif method == 'ECC':
        shifts = calculate_shifts_ecc(stack, start, show_progressbar)
    elif method == 'PC':
        shifts = calculate_shifts_pc(stack, start, show_progressbar)
    elif method == 'StackReg':
        composed = calculate_shifts_stackreg(stack)
    if method != 'StackReg':
        composed = compose_shifts(shifts, start)
    aligned = apply_shifts(stack, composed)
    return aligned


def tilt_com(stack, offset=0, locs=None, output=True):
    """
    Perform tilt axis alignment using center of mass (CoM) tracking.

    Compares path of specimen to the path expected for an ideal cylinder

    Args
    ----------
    stack : TomoStack object
        3-D numpy array containing the tilt series data
    offset : integer
        Not currently used
    locs : list
        Locations at which to perform the CoM analysis
    output : boolean
        Output alignment results to console after each iteration

    Returns
    ----------
    out : TomoStack object
        Copy of the input stack after rotation and translation to center and
        make the tilt axis vertical

    """
    data = stack.deepcopy()
    if locs is None:
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
        locs = np.int16(np.sort(locs))
    if output:
        logger.info('\nCorrecting tilt axis....')
    tilts = stack.axes_manager[0].axis * np.pi / 180
    xshift = 0
    tiltaxis = 0
    totaltilt = 0
    totalshift = 0
    count = 1

    while abs(tiltaxis) >= 1 or abs(xshift) >= 1 or count == 1:
        centers = np.zeros([data.data.shape[0], 3])
        sinos = data.data[:, :, locs]
        for k in range(0, 3):
            for i in range(0, data.data.shape[0]):
                centers[i, k] = center_of_mass(sinos[i, :, k])[0]

        com_results = np.zeros(3)

        def com_func(x, r0, a, b):
            return r0 - a * np.cos(x) - b * np.sin(x)

        guess = (0.0, 0.0, 0.0)
        # noinspection PyTypeChecker
        for i in range(0, 3):
            com_results[i] =\
                optimize.curve_fit(com_func,
                                   tilts,
                                   np.int32(centers[:, i]),
                                   guess)[0][0]

        def axis_func(x, m, b):
            return m * x + b

        guess = [0.0, 0.0]
        # noinspection PyTypeChecker
        axis_fits =\
            optimize.curve_fit(f=axis_func, xdata=locs,
                               ydata=com_results, p0=guess)[0]

        tiltaxis = 180 / np.pi * np.tanh(axis_fits[0])
        xshift = (axis_fits[1] / axis_fits[0] * np.sin(np.pi / 180 * tiltaxis))
        xshift = (data.data.shape[1] / 2) - xshift - offset
        totaltilt += tiltaxis
        totalshift += xshift

        if output:
            logger.info(('Iteration #%s' % count))
            logger.info(('Calculated tilt correction is: %s' % str(tiltaxis)))
            logger.info(('Calculated shift value is: %s' % str(xshift)))
        count += 1

        data = data.trans_stack(xshift=0, yshift=xshift, angle=-tiltaxis)

    out = stack.deepcopy()
    out.data = np.transpose(data.data, (0, 2, 1))
    if output:
        logger.info('\nTilt axis alignment complete')
    out.original_metadata.tiltaxis = -totaltilt
    out.original_metadata.xshift = totalshift
    return out


def tilt_maximage(data, limit=10, delta=0.3, output=False,
                  show_progressbar=False):
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
    output : boolean
        Output alignment results to console after each iteration
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
        if img.shape[0] < img.shape[1]:
            center_loc = np.int32((img.shape[1] - img.shape[0]) / 2)
            img = img[:, center_loc:-center_loc]
            if img.shape[0] != img.shape[1]:
                img = img[:, 0:-1]
            h = np.hamming(img.shape[0])
            ham2d = np.sqrt(np.outer(h, h))
        elif img.shape[1] < img.shape[0]:
            center_loc = np.int32((img.shape[0] - img.shape[1]) / 2)
            img = img[center_loc:-center_loc, :]
            if img.shape[0] != img.shape[1]:
                img = img[0:-1, :]
            h = np.hamming(img.shape[1])
            ham2d = np.sqrt(np.outer(h, h))
        else:
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
    if output:
        logger.info('Optimum positive rotation angle: {}'.format(pos_angle))
        logger.info('Optimum negative rotation angle: {}'.format(neg_angle))
        logger.info('Optimum positive rotation angle: {}'.format(opt_angle))

    out = copy.deepcopy(data)
    out = out.trans_stack(xshift=0, yshift=0, angle=opt_angle)
    out.data = np.transpose(out.data, (0, 2, 1))
    out.original_metadata.tiltaxis = opt_angle
    return out


def align_to_other(stack, other, verbose):
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

    shifts = stack.original_metadata.shifts
    out.original_metadata.shifts = shifts

    tiltaxis = stack.original_metadata.tiltaxis
    out.original_metadata.tiltaxis = tiltaxis

    xshift = stack.original_metadata.xshift
    out.original_metadata.xshift = stack.original_metadata.xshift

    yshift = stack.original_metadata.yshift
    out.original_metadata.yshift = stack.original_metadata.yshift

    if type(stack.original_metadata.shifts) is np.ndarray:
        for i in range(0, out.data.shape[0]):
            out.data[i, :, :] =\
                ndimage.shift(out.data[i, :, :],
                              shift=[shifts[i, 1], shifts[i, 0]],
                              order=0)

    if (tiltaxis != 0) or (xshift != 0):
        out = out.trans_stack(xshift=xshift, yshift=yshift, angle=tiltaxis)

    if verbose:
        logger.info('TomoStack alignment applied')
        logger.info('X-shift: %.1f' % xshift)
        logger.info('Y-shift: %.1f' % yshift)
        logger.info('Rotation: %.1f' % tiltaxis)
    return out
