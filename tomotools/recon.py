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
            print('Reconstruction complete')
        elif astra.astra.use_cuda() or cuda:
            '''ASTRA weighted-backprojection CUDA reconstruction of single
            slice'''
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
            rec = tomopy.recon(stack.data, theta, center=rot_center,
                               algorithm=tomopy.astra, filter_name='ramlak',
                               options=options, **kwargs)
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
                               algorithm=tomopy.astra, options=options,
                               **kwargs)
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
                               algorithm=tomopy.astra, options=options,
                               **kwargs)
            print('Reconstruction complete')
        else:
            raise Exception('Error related to ASTRA Toolbox')
    else:
        raise ValueError('Unknown reconstruction algorithm:' + method)
    return rec


def check_sirt_error(sinogram, tol, verbose, constrain, cuda):
    tilts = sinogram.axes_manager[0].axis * np.pi / 180
    vol_geom = astra.create_vol_geom(sinogram.data.shape[1],
                                     sinogram.data.shape[1])
    proj_geom = astra.create_proj_geom('parallel', 1.0, sinogram.data.shape[1],
                                       tilts)
    proj_id = astra.create_projector('strip', proj_geom, vol_geom)
    current_sinogram_id = astra.data2d.create('-sino', proj_geom,
                                              sinogram.data)
    rec_id = astra.data2d.create('-vol', vol_geom)

    if cuda:
        cfg = astra.astra_dict('SIRT_CUDA')
    else:
        cfg = astra.astra_dict('SIRT')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectorId'] = proj_id
    cfg['ProjectionDataId'] = current_sinogram_id
    if constrain:
        cfg['option'] = {}
        cfg['option']['MinConstraint'] = 0

    error = []
    terminate = False

    iteration = 1
    while not terminate:
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        rec = astra.data2d.get(rec_id)
        if iteration == 1:
            rec_stack = rec[np.newaxis, :, :]
        else:
            rec_stack = np.vstack((rec_stack, rec[np.newaxis, :, :]))
        current_sinogram_id, forward_project = astra.create_sino(rec, proj_id)
        astra.data2d.store(current_sinogram_id,
                           sinogram.data - forward_project)

        error.append(np.sum((sinogram.data - forward_project)**2))
        if len(error) > 1:
            change = np.abs((error[-2] - error[-1]) / error[-2])
            if verbose:
                print("Change after iteration %s: %.1f %%" %
                      (iteration, 100 * change))
            if change < tol:
                terminate = True
                if verbose:
                    print('Change in error below tolerance after %s iterations'
                          % iteration)
        iteration += 1
    error = np.array(error)
    rec_stack = hs.signals.Signal2D(rec_stack)
    return error, rec_stack
