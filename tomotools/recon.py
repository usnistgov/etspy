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


def run(stack, method, rot_center=None, iterations=None, constrain=None,
        thresh=None, cuda=True):
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

    Returns
    ----------
    rec : Numpy array
        Containing the reconstructed volume

    """
    theta = stack.axes_manager[0].axis*np.pi/180
    if method == 'FBP':
        if not astra.astra.use_cuda() or not cuda:
            '''ASTRA weighted-backprojection reconstruction of single slice'''
            options = {'proj_type': 'linear', 'method': 'FBP'}
            rec = tomopy.recon(stack.data, theta, center=rot_center,
                               algorithm=tomopy.astra, options=options)
            print('Reconstruction complete')
        elif astra.astra.use_cuda() or cuda:
            '''ASTRA weighted-backprojection CUDA reconstruction of single
            slice'''
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
            rec = tomopy.recon(stack.data, theta, center=rot_center,
                               algorithm=tomopy.astra, options=options)
            print('Reconstruction complete')
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
                               algorithm=tomopy.astra, options=options)
            print('Reconstruction complete')
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
                               algorithm=tomopy.astra, options=options)
            print('Reconstruction complete')
        else:
            raise Exception('Error related to ASTRA Toolbox')
    else:
        raise ValueError('Unknown reconstruction algorithm:' + method)
    return rec
