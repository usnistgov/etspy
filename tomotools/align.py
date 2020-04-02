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


def apply_shifts(stack, shifts):
    """
    Apply a series of shifts to a TomoStack

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
    Compose a series of calculated shifts prior to applying them to an
    image stack

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
    sr = StackReg(StackReg.TRANSLATION)
    shifts = sr.register_stack(stack.data, reference='previous')
    shifts = -np.array([i[0:2, 2] for i in shifts])
    return shifts


def align_stack(stack, method, start, show_progressbar, nslice, ratio):
    """
    Compute the shifts for spatial registration.

    Shifts are determined by one of three methods:
        1.) Phase correlation (PC) as implemented in OpenCV.
        2.) Enhanced correlation coefficient (ECC) as implemented in OpenCV.
        3.) Center of mass (COM) tracking.  A Python implementation of
            Matlab code described in:
            T. Sanders. Matlab imaging algorithms: Image reconstruction,
            restoration, and alignment, with a focus in tomography.
            http://www.toby-sanders.com/software ,
            https://doi.org/10.13140/RG.2.2.33492.60801

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


def tilt_correct(stack, offset=0, locs=None, output=True):
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
    def getpoints(data, numpoints=3):
        """
        Prompt user for locations at which to fit the CoM.

        Displays the central image in a stack and prompt the user to
        choose three locations by mouse click.  Once three locations have been
        clicked, the window closes and the function returns the coordinates

        Args
        ----------
        data : Numpy array
            Tilt series datastack
        numpoints : integer
            Number of points to use in fitting the tilt axis

        Returns
        ----------
        coords : Numpy array
            array containing the XY coordinates selected interactively by the
            user

        """
        warnings.filterwarnings('ignore')
        plt.figure(num='Align Tilt', frameon=False)
        if len(data.shape) == 3:
            plt.imshow(data[np.int(data.shape[0] / 2), :, :], cmap='gray')
        else:
            plt.imshow(data, cmap='gray')
        plt.title('Choose %s points for tilt axis alignment....' %
                  str(numpoints))
        coords = np.array(plt.ginput(numpoints, timeout=0, show_clicks=True))
        plt.close()
        return coords

    def sinocalc(array, y):
        """
        Extract sinograms at three stack positions chosen by user.

        Args
        ----------
        array : Numpy array
            3-D numpy array containing the tilt series data
        y : Numpy array
            Array containing the coordinates selected by the user in
            getPoints()

        Returns
        ----------
        outvals : Numpy array
            Array containing the center of mass as a function of tilt for the
            selected sinograms

        """
        def center_of_mass(row):
            """
            Compute the center of mass for a row of pixels.

            Args
            ----------
            row : Numpy array
                Row of pixels extracted from a sinogram

            Returns
            ----------
            value : float
                Center of mass of the input row

            """
            size = np.size(row)
            value = 0.0
            for j in range(0, size):
                value = value + row[j] * (j + 1)
            value = value / np.sum(row)
            return value

        outvals = np.zeros([np.size(array, axis=0), 3])
        sinotop = array[:, :, y[0]]
        sinomid = array[:, :, y[1]]
        sinobot = array[:, :, y[2]]

        for k in range(array.shape[0]):
            outvals[k][0] = center_of_mass(sinotop[k, :])
            outvals[k][1] = center_of_mass(sinomid[k, :])
            outvals[k][2] = center_of_mass(sinobot[k, :])

        return outvals

    def fit_coms(thetas, coms):
        """
        Fit the motion of calculated centers-of-mass in a sinogram.

        Fit is to a sinusoidal function: (r0-A*cos(tilt)-B*sin(tilt)) as
        would be expected for an ideal cylinder. Return the coefficient of
        the fit equation for use in fitTiltAxis

        Args
        ----------
        thetas : Numpy array
            Array containing the stage tilt at each row in the sinogram
        coms : Numpy array
            Array containing the calculated center of mass as a function of
            tilt for the sinogram

        Returns
        ----------
        coeffs : Numpy array
            Coefficients (r0 , A , and B) resulting from the fit

        """
        def func(x, r0, a, b):
            return r0 - a * np.cos(x) - b * np.sin(x)

        guess = (0.0, 0.0, 0.0)
        # noinspection PyTypeChecker
        coeffs, covars = optimize.curve_fit(func,
                                            thetas,
                                            np.int16(coms),
                                            guess)
        return coeffs

    def fit_tilt_axis(coords, vals):
        """
        Fit the coefficients calculated by fit_coms() to a linear function.

        Fit is performed at each of the three user chosen positions to
        determine the necessary rotation to vertically align the tilt axis

        Args
        ----------
        coords : Numpy array
            Horizontal coordinates from which the sinograms were extracted
        vals : Numpy array
            Array containing the r0 coefficient calculated for each sinogram by
            fitCoMs

        Returns
        ----------
        coeffs : Numpy array
            Coefficients (m and b) resulting from the fit

        """
        def func(x, m, b):
            return m * x + b

        guess = [0.0, 0.0]
        # noinspection PyTypeChecker
        coeffs, covars = optimize.curve_fit(f=func, xdata=coords, ydata=vals,
                                            p0=guess)
        return coeffs

    data = stack.deepcopy()
    if locs is None:
        locs = np.int16(np.sort(getpoints(stack.data)[:, 0]))
    else:
        locs = np.int16(np.sort(locs))
    if output:
        print('\nCorrecting tilt axis....')
    tilts = stack.axes_manager[0].axis * np.pi / 180
    xshift = 0
    tiltaxis = 0
    totaltilt = 0
    totalshift = 0
    count = 1

    while abs(tiltaxis) >= 1 or abs(xshift) >= 1 or count == 1:
        centers = sinocalc(data.data, locs)

        com_results = np.zeros([3, 3])
        com_results[0, :] = fit_coms(tilts, centers[:, 0])
        com_results[1, :] = fit_coms(tilts, centers[:, 1])
        com_results[2, :] = fit_coms(tilts, centers[:, 2])

        r = np.zeros(3)
        r[:] = com_results[:, 0]

        axis_fits = fit_tilt_axis(locs, r)
        tiltaxis = 180 / np.pi * np.tanh(axis_fits[0])
        xshift = (axis_fits[1] / axis_fits[0] * np.sin(np.pi / 180 * tiltaxis))
        xshift = (data.data.shape[1] / 2) - xshift - offset
        totaltilt += tiltaxis
        totalshift += xshift

        if output:
            print(('Iteration #%s' % count))
            print(('Calculated tilt correction is: %s' % str(tiltaxis)))
            print(('Calculated shift value is: %s' % str(xshift)))
        count += 1

        data = data.trans_stack(xshift=0, yshift=xshift, angle=-tiltaxis)

    out = copy.deepcopy(data)
    out.data = np.transpose(data.data, (0, 2, 1))
    if output:
        print('\nTilt axis alignment complete')
    out.original_metadata.tiltaxis = -totaltilt
    out.original_metadata.xshift = totalshift
    return out


def tilt_analyze(data, limit=10, delta=0.3, output=False,
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
        print('Optimum positive rotation angle: {}'.format(pos_angle))
        print('Optimum negative rotation angle: {}'.format(neg_angle))
        print('Optimum positive rotation angle: {}'.format(opt_angle))

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
        print('TomoStack alignment applied')
        print('X-shift: %.1f' % xshift)
        print('Y-shift: %.1f' % yshift)
        print('Rotation: %.1f' % tiltaxis)
    return out
