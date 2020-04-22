# -*- coding: utf-8 -*-
#
# This file is part of TomoTools

"""
Reconstruction module for TomoTools package.

@author: Andrew Herzing
"""
import tomopy
import numpy as np
import astra
import hyperspy.api as hs
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run(stack, method, rot_center=None, iterations=None, constrain=None,
        thresh=None, cuda=True, **kwargs):
    """
    Perform reconstruction of input tilt series.

    Args
    ----------
    stack :TomoStack object
       TomoStack containing the input tilt series
    method : string
        Reconstruction algorithm to use.  Must be either 'FBP' (default) or
        'SIRT'
    rot_center : float
        Location of the rotation center.  If None, position is assumed to be
        the center of the image.
    iterations : integer (only required for SIRT)
        Number of iterations for the SIRT reconstruction (for SIRT methods
        only)
    constrain : boolean
        If True, output reconstruction is constrained above value given by
        'thresh'
    thresh : integer or float
        Value above which to constrain the reconstructed data
    cuda : boolean
        If True, use the CUDA-accelerated Astra algorithms. Otherwise,
        use the CPU-based algorithms
    **kwargs : dict
        Any other keyword arguments are passed through to ``tomopy.recon``

    Returns
    ----------
    rec : Numpy array
        Containing the reconstructed volume

    """
    theta = stack.axes_manager[0].axis * np.pi / 180
    if method == 'FBP':
        if not astra.astra.use_cuda() or not cuda:
            '''ASTRA weighted-backprojection reconstruction of single slice'''
            options = {'proj_type': 'linear', 'method': 'FBP'}
            rec = tomopy.recon(stack.data, theta, center=rot_center,
                               algorithm=tomopy.astra, filter_name='ramlak',
                               options=options, **kwargs)
            logger.info('Reconstruction complete')
        elif astra.astra.use_cuda() or cuda:
            '''ASTRA weighted-backprojection CUDA reconstruction of single
            slice'''
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
            rec = tomopy.recon(stack.data, theta, center=rot_center,
                               algorithm=tomopy.astra, filter_name='ramlak',
                               options=options, **kwargs)
            logger.info('Reconstruction complete')
        else:
            raise Exception('Error related to ASTRA Toolbox')
    elif method == 'SIRT':
        if not iterations:
            iterations = 20
        if not astra.astra.use_cuda() or not cuda:
            '''ASTRA SIRT reconstruction of single slice'''
            if constrain:
                if not thresh:
                    thresh = 0
                extra_options = {'MinConstraint': thresh}
                options = {'proj_type': 'linear', 'method': 'SIRT',
                           'num_iter': iterations,
                           'extra_options': extra_options}
            else:
                options = {'proj_type': 'linear', 'method': 'SIRT',
                           'num_iter': iterations}
            rec = tomopy.recon(stack.data, theta, center=rot_center,
                               algorithm=tomopy.astra, options=options,
                               **kwargs)
            logger.info('Reconstruction complete')
        elif astra.astra.use_cuda() or cuda:
            '''ASTRA CUDA-accelerated SIRT reconstruction'''
            if constrain:
                if not thresh:
                    thresh = 0
                extra_options = {'MinConstraint': thresh}
                options = {'proj_type': 'cuda', 'method': 'SIRT_CUDA',
                           'num_iter': iterations,
                           'extra_options': extra_options}
            else:
                options = {'proj_type': 'cuda', 'method': 'SIRT_CUDA',
                           'num_iter': iterations}
            rec = tomopy.recon(stack.data, theta, center=rot_center,
                               algorithm=tomopy.astra, options=options,
                               ncore=1, **kwargs)
            logger.info('Reconstruction complete')
        else:
            raise Exception('Error related to ASTRA Toolbox')
    else:
        raise ValueError('Unknown reconstruction algorithm:' + method)
    return rec


def check_sirt_error(sinogram, tol, verbose, constrain, cuda):
    """
    Determine the optimum number of SIRT iterations.

    Evaluates the difference between SIRT reconstruction and input data
    at each iteration and terminates when the change between iterations is
    below tolerance.

    Args
    ----------
    sinogram : Hyperspy Signal2D
        Single slice from a tomogram for reconstruction evaluation.
    tol : float
        Fractional change between iterations at which the
        evaluation will terminate.
    verbose : boolean
        If True, output the percentage change in error between the current
        iteration and the previous.
    constrain : boolean
        If True, perform SIRT reconstruction with a non-negativity
        constraint.
    cuda : boolean
        If True, perform reconstruction using the GPU-accelrated algorithm.

    Returns
    ----------
    error : Numpy array
        Sum of squared difference between the forward-projected reconstruction
        and the input sinogram at each iteration

    rec_stack : Hyperspy Signal2D
        Signal containing the SIRT reconstruction at each iteration
        for visual inspection.

    """
    tilts = sinogram.axes_manager[0].axis * np.pi / 180

    error = []
    terminate = False
    rec = None
    iteration = 0
    while not terminate:
        rec = tomopy.recon(sinogram.data, theta=tilts, algorithm='sirt',
                           num_iter=1, init_recon=rec)
        if iteration == 0:
            rec_stack = rec
        else:
            rec_stack = np.vstack((rec_stack, rec))
        forward_project = tomopy.project(rec, theta=tilts, pad=False)

        error.append(np.sum((sinogram.data - forward_project)**2))
        if len(error) > 1:
            change = np.abs((error[-2] - error[-1]) / error[-2])
            logger.info("Change after iteration %s: %.1f %%" %
                        (iteration, 100 * change))
            if change < tol:
                terminate = True
                logger.info('Change in error below tolerance after %s '
                            'iterations' % iteration)
        iteration += 1
    error = np.array(error)
    rec_stack = hs.signals.Signal2D(rec_stack)
    return error, rec_stack
