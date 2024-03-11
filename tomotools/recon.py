# -*- coding: utf-8 -*-
#
# This file is part of TomoTools

"""
Reconstruction module for TomoTools package.

@author: Andrew Herzing
"""
import numpy as np
import astra
import logging
import multiprocessing as mp
import tqdm

ncpus = mp.cpu_count()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_alg(sino, iters, sino_id, alg_id, rec_id):
    """
    Run reconstruction algorithm.

    Args
    ----------
    sion : NumPy array
       Sinogram of shape (nangles, ny)
    iters : int
        Number of iterations for the SIRT reconstruction
    sino_id : int
        ASTRA sinogram identity
    alg_id : int
        ASTRA algorithm identity
    rec_id : boolean
        ASTRA reconstruction identity

    Returns
    ----------
    Numpy array
        Reconstruction of input sinogram

    """
    astra.data2d.store(sino_id, sino)
    astra.algorithm.run(alg_id, iters)
    return astra.data2d.get(rec_id)


def run(stack, method, niterations=20, constrain=None, thresh=0, cuda=None, thickness=None,
        ncores=None, filter='shepp-logan', **kwargs):
    """
    Perform reconstruction of input tilt series.

    Args
    ----------
    stack :TomoStack object
       TomoStack containing the input tilt series
    method : string
        Reconstruction algorithm to use.  Must be either 'FBP' (default) or
        'SIRT'
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
    if len(stack.data.shape) == 2:
        nangles, ny = stack.data.shape
        nx = 1
    else:
        nangles, ny, nx = stack.data.shape

    thetas = np.pi * stack.metadata.Tomography.tilts / 180.

    if thickness is None:
        thickness = ny

    rec = np.zeros([nx, thickness, ny], np.float32)

    proj_geom = astra.create_proj_geom('parallel', 1.0, ny, thetas)
    vol_geom = astra.create_vol_geom((thickness, ny))
    rec_id = astra.data2d.create('-vol', vol_geom)
    sino_id = astra.data2d.create('-sino', proj_geom, np.zeros([nangles, ny]))

    if cuda:
        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

        if method.lower() == 'fbp':
            print('Reconstructing with CUDA-accelerated FBP algorithm')
            cfg = astra.astra_dict('FBP_CUDA')
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = rec_id
            cfg['option'] = {}
            cfg['option']['FilterType'] = filter.lower()
            niterations = 1
        elif method.lower() == 'sirt':
            logger.info('Reconstructing with CUDA-accelerated SIRT algorithm (%s iterations)' % niterations)
            cfg = astra.astra_dict('SIRT_CUDA')
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = rec_id
            if constrain:
                cfg['option'] = {}
                cfg['option']['MinConstraint'] = thresh

        alg = astra.algorithm.create(cfg)

        for i in tqdm.tqdm(range(0, nx)):
            astra.data2d.store(sino_id, stack.data[:, :, i])
            astra.algorithm.run(alg, niterations)
            rec[i, :, :] = astra.data2d.get(rec_id)

    else:
        if ncores is None:
            ncores = min(nx, int(0.9 * mp.cpu_count()))

        proj_id = astra.create_projector('linear', proj_geom, vol_geom)

        if method.lower() == 'fbp':
            logger.info('Reconstructing with CPU-based FBP algorithm')
            cfg = astra.astra_dict('FBP')
            cfg['ProjectorId'] = proj_id
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = rec_id
            cfg['option'] = {}
            cfg['option']['FilterType'] = filter.lower()
            niterations = 1
        elif method.lower() == 'sirt':
            logger.info('Reconstructing with CPU-based FBP algorithm')
            cfg = astra.astra_dict('SIRT')
            cfg['ProjectorId'] = proj_id
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = rec_id
            if constrain:
                cfg['option'] = {}
                cfg['option']['MinConstraint'] = thresh

        alg = astra.algorithm.create(cfg)

        if ncores == 1:
            for i in tqdm.tqdm(range(0, nx)):
                rec[i] = run_alg(stack.data[:, :, i], niterations, sino_id, alg, rec_id)
        else:
            logger.info('Using %s CPU cores to reconstruct %s slices' % (ncores, nx))
            with mp.Pool(ncores) as pool:
                for i, result in enumerate(pool.starmap(run_alg, [(stack.data[:, :, i], niterations, sino_id, alg, rec_id) for i in range(0, nx)])):
                    rec[i] = result
    astra.clear()
    return rec


def astra_project(obj, angles, cuda=False):
    """
    Calculate projection of a 3D object using Astra-toolbox.

    Args
    ----------
    obj : NumPy array
        Either a 2D or 3D array containing the object to project.
        If 2D, the structure is of the form [z, x] where z is the
        projection axis.  If 3D, the strucutre is [y, z, x] where y is
        the tilt axis.
    angles : list or NumPy array
        The projection angles in degrees
    cuda : boolean
        If True, perform reconstruction using the GPU-accelerated algorithm.

    Returns
    ----------
    sino : Numpy array
        3D array of the form [y, angle, x] containing the projection of
        the input object

    """
    if len(obj.shape) == 2:
        obj = np.expand_dims(obj, 0)
    thetas = np.pi * angles / 180.
    y_pix, thickness, x_pix = obj.shape
    if cuda:
        vol_geom = astra.create_vol_geom(thickness, x_pix, y_pix)
        proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0,
                                           y_pix, x_pix, thetas)
        sino_id, sino = astra.create_sino3d_gpu(obj,
                                                proj_geom,
                                                vol_geom)
        astra.data3d.delete(sino_id)
    else:
        sino = np.zeros([y_pix, len(angles), x_pix], np.float32)
        vol_geom = astra.create_vol_geom(thickness, x_pix)
        proj_geom = astra.create_proj_geom('parallel', 1.0, x_pix, thetas)
        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
        for i in range(0, y_pix):
            sino_id, sino[i, :, :] = astra.create_sino(obj[i, :, :],
                                                       proj_id,
                                                       vol_geom)
        astra.data2d.delete(sino_id)

    sino = np.rollaxis(sino, 1)
    return sino


def astra_sirt_error(sinogram, angles, iterations=50,
                     constrain=True, thresh=0, cuda=False):
    """
    Perform SIRT reconstruction using the Astra toolbox algorithms.

    Args
    ----------
    sinogram : NumPy array
       Tilt series data either of the form [angles, x] or [angles, y, x] where
       y is the tilt axis and x is the projection axis.
    angles : list or NumPy array
        Projection angles in degrees.
    iterations : integer
        Number of iterations for the SIRT reconstruction.
    constrain : boolean
        If True, output reconstruction is constrained above value given by
        'thresh'. Default is True.
    thresh : integer or float
        Value above which to constrain the reconstructed data if 'constrain'
        is True.
    cuda : boolean
        If True, use the CUDA-accelerated Astra algorithms. Otherwise,
        use the CPU-based algorithms
    Returns
    ----------
    rec : Numpy array
        3D array of the form [y, z, x] containing the reconstructed object.

    """
    thetas = angles * np.pi / 180

    nangles, ny = sinogram.shape

    proj_geom = astra.create_proj_geom('parallel', 1.0, ny, thetas)
    vol_geom = astra.create_vol_geom((ny, ny))
    rec_id = astra.data2d.create('-vol', vol_geom)
    sino_id = astra.data2d.create('-sino', proj_geom, np.zeros([nangles, ny]))

    if cuda:
        alg_name = 'SIRT_CUDA'
        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    else:
        alg_name = 'SIRT'
        proj_id = astra.create_projector('linear', proj_geom, vol_geom)

    astra.data2d.store(sino_id, sinogram)

    cfg = astra.astra_dict(alg_name)
    cfg['ProjectionDataId'] = sino_id
    cfg['ProjectorId'] = proj_id
    cfg['ReconstructionDataId'] = rec_id
    if constrain:
        cfg['option'] = {}
        cfg['option']['MinConstraint'] = thresh

    alg = astra.algorithm.create(cfg)

    rec = np.zeros([iterations, ny, ny], np.float32)
    residual_error = np.zeros(iterations)

    for i in tqdm.tqdm(range(iterations)):
        astra.algorithm.run(alg, 1)
        rec[i] = astra.data2d.get(rec_id)
        if cuda:
            residual_error[i] = astra.algorithm.get_res_norm(alg)
        else:
            curr_id, curr = astra.create_sino(rec[i], proj_id)
            residual_error[i] = np.linalg.norm((sinogram - curr))
    astra.clear()
    return rec, residual_error
