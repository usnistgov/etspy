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
import copy
from scipy.ndimage import gaussian_filter, convolve

ncpus = mp.cpu_count()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_alg(sino, iters, sino_id, alg_id, rec_id):
    """
    Run FBP or SIRT reconstruction algorithm.

    Args
    ----------
    sino : NumPy array
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


def run_dart(sino, iters, dart_iters, p,
             alg_id, proj_id, mask_id, rec_id, sino_id,
             thresholds, gray_levels):
    """
    Run DART reconstruction algorithm.

    Adapted from pseudo-code published in:
    K. J. Batenburg and J. Sijbers, "DART: A Practical Reconstruction
    Algorithm for Discrete Tomography," doi: 10.1109/TIP.2011.2131661.

    Args
    ----------
    sino : NumPy array
       Sinogram of shape (nangles, ny)
    iters : int
        Number of iterations for the SART reconstruction
    dart_iters : int
        Number of iterations for the DART reconstruction
    p : float
        Probability for free pixel determination
    alg_id : int
        ASTRA algorithm identity
    proj_id : int
        ASTRA projector identity
    mask_id : boolean
        ASTRA mask identity
    rec_id : boolean
        ASTRA reconstruction identity
    sino_id : int
        ASTRA sinogram identity
    thresholds : list or NumPy array
        Thresholds for DART reconstruction
    gray_levels : list or NumPy array
        Gray levels for DART reconstruction


    Returns
    ----------
    Numpy array
        Reconstruction of input sinogram

    """
    thickness, ny = astra.data2d.get(rec_id).shape
    astra.data2d.store(sino_id, sino)
    astra.data2d.store(rec_id, np.zeros([thickness, ny]))
    astra.data2d.store(mask_id, np.ones([thickness, ny]))
    astra.algorithm.run(alg_id, iters)
    curr_rec = astra.data2d.get(rec_id)
    dart_rec = copy.deepcopy(curr_rec)
    for j in range(dart_iters):
        segmented = dart_segment(dart_rec, thresholds, gray_levels)
        boundary = get_dart_boundaries(segmented)

        # Define free and fixed pixels
        free = np.random.rand(*dart_rec.shape)
        free = free < 1 - p
        free = np.logical_or(boundary, free)
        fixed = ~free
        free_idx = np.where(free)
        fixed_idx = np.where(fixed)

        # Set fixed pixels to segmented values
        dart_rec[fixed_idx[0], fixed_idx[1]] = segmented[fixed_idx[0], fixed_idx[1]]

        # Calculate sinogram of free pixels
        fixed_rec = copy.deepcopy(dart_rec)
        fixed_rec[free_idx[0], free_idx[1]] = 0
        _, fixed_sino = astra.creators.create_sino(fixed_rec, proj_id)
        free_sino = sino - fixed_sino

        # Run SART reconstruction on free sinogram with free pixel mask
        astra.data2d.store(rec_id, dart_rec)
        astra.data2d.store(mask_id, free)
        astra.data2d.store(sino_id, free_sino)
        astra.algorithm.run(alg_id, iters)
        dart_rec = astra.data2d.get(rec_id)

        # Smooth reconstruction
        if j < dart_iters - 1:
            smooth = gaussian_filter(dart_rec, sigma=1)
            curr_rec[free_idx[0], free_idx[1]] = smooth[free_idx[0], free_idx[1]]
        else:
            curr_rec = dart_rec
    return curr_rec


def run(stack, method, niterations=20, constrain=None, thresh=0, cuda=None, thickness=None, ncores=None,
        filter="shepp-logan", gray_levels=None, dart_iterations=None, p=0.99,):
    """
    Perform reconstruction of input tilt series.

    Args
    ----------
    stack :TomoStack object
       TomoStack containing the input tilt series
    method : string
        Reconstruction algorithm to use.  Must be either 'FBP' (default) or
        'SIRT'
    niterations : integer (only required for SIRT)
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
    thickness : int
        Limit for the height of the reconstruction
    ncores : int
        Number of cores to use for multithreaded CPU-based reconstructions
    filter : str
        Filter to use for filtered backprojection
    gray_levels : list or NumPy array
        Gray levels for DART reconstruction
    dart_iterations : int
        Number of DART iterations
    p : float
        Probability for setting free pixels in DART reconstruction

    Returns
    ----------
    rec : Numpy array
        Containing the reconstructed volume

    """
    if len(stack.data.shape) == 2:
        nangles, ny = stack.data.shape
        stack.data = stack.data[:, :, np.newaxis]
        nx = 1
    else:
        nangles, ny, nx = stack.data.shape

    thetas = np.pi * stack.metadata.Tomography.tilts / 180.0

    if thickness is None:
        thickness = ny

    rec = np.zeros([nx, thickness, ny], np.float32)

    proj_geom = astra.create_proj_geom("parallel", 1.0, ny, thetas)
    vol_geom = astra.create_vol_geom((thickness, ny))
    rec_id = astra.data2d.create("-vol", vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, np.zeros([nangles, ny]))

    if cuda:
        proj_id = astra.create_projector("cuda", proj_geom, vol_geom)

        if method.lower() == "fbp":
            logger.info("Reconstructing with CUDA-accelerated FBP algorithm")
            cfg = astra.astra_dict("FBP_CUDA")
            cfg["ProjectorId"] = proj_id
            cfg["ProjectionDataId"] = sino_id
            cfg["ReconstructionDataId"] = rec_id
            cfg["option"] = {}
            cfg["option"]["FilterType"] = filter.lower()
            niterations = 1

            alg = astra.algorithm.create(cfg)

            for i in tqdm.tqdm(range(0, nx)):
                astra.data2d.store(sino_id, stack.data[:, :, i])
                astra.algorithm.run(alg, niterations)
                rec[i, :, :] = astra.data2d.get(rec_id)
        elif method.lower() == "sirt":
            logger.info(
                "Reconstructing with CUDA-accelerated SIRT algorithm (%s iterations)"
                % niterations
            )
            cfg = astra.astra_dict("SIRT_CUDA")
            cfg["ProjectorId"] = proj_id
            cfg["ProjectionDataId"] = sino_id
            cfg["ReconstructionDataId"] = rec_id
            if constrain:
                cfg["option"] = {}
                cfg["option"]["MinConstraint"] = thresh
            alg = astra.algorithm.create(cfg)

            for i in tqdm.tqdm(range(0, nx)):
                astra.data2d.store(sino_id, stack.data[:, :, i])
                astra.algorithm.run(alg, niterations)
                rec[i, :, :] = astra.data2d.get(rec_id)
        elif method.lower() == "dart":
            thresholds = [(gray_levels[i] + gray_levels[i + 1]) // 2 for i in range(len(gray_levels) - 1)]
            mask = np.ones([thickness, ny])
            mask_id = astra.data2d.create('-vol', vol_geom, mask)
            cfg = astra.astra_dict('SART_CUDA')
            cfg["ProjectorId"] = proj_id
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = rec_id
            cfg['option'] = {}
            cfg['option']['MinConstraint'] = 0
            cfg['option']['MaxConstraint'] = 255
            cfg['option']['ReconstructionMaskId'] = mask_id
            alg = astra.algorithm.create(cfg)

            for i in tqdm.tqdm(range(0, nx)):
                sinogram = stack.data[:, :, i]
                astra.data2d.store(sino_id, sinogram)
                astra.data2d.store(rec_id, np.zeros([thickness, ny]))
                astra.data2d.store(mask_id, np.ones([thickness, ny]))
                rec[i, :, :] = run_dart(sinogram, niterations, dart_iterations, p,
                                        alg, proj_id, mask_id, rec_id, sino_id, thresholds, gray_levels)
    else:
        if ncores is None:
            ncores = min(nx, int(0.9 * mp.cpu_count()))

        proj_id = astra.create_projector("linear", proj_geom, vol_geom)

        if method.lower() == "fbp":
            logger.info("Reconstructing with CPU-based FBP algorithm")
            cfg = astra.astra_dict("FBP")
            cfg["ProjectorId"] = proj_id
            cfg["ProjectionDataId"] = sino_id
            cfg["ReconstructionDataId"] = rec_id
            cfg["option"] = {}
            cfg["option"]["FilterType"] = filter.lower()
            niterations = 1
        elif method.lower() == "sirt":
            logger.info("Reconstructing with CPU-based SIRT algorithm")
            cfg = astra.astra_dict("SIRT")
            cfg["ProjectorId"] = proj_id
            cfg["ProjectionDataId"] = sino_id
            cfg["ReconstructionDataId"] = rec_id
            if constrain:
                cfg["option"] = {}
                cfg["option"]["MinConstraint"] = thresh
        elif method.lower() == "dart":
            thresholds = [(gray_levels[i] + gray_levels[i + 1]) // 2 for i in range(len(gray_levels) - 1)]
            mask = np.ones([thickness, ny])
            mask_id = astra.data2d.create('-vol', vol_geom, mask)
            cfg = astra.astra_dict('SART')
            cfg["ProjectorId"] = proj_id
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = rec_id
            cfg['option'] = {}
            cfg['option']['MinConstraint'] = 0
            cfg['option']['MaxConstraint'] = 255
            cfg['option']['ReconstructionMaskId'] = mask_id

        alg = astra.algorithm.create(cfg)

        if method.lower() in ['fbp', 'sirt', ]:
            if ncores == 1:
                for i in tqdm.tqdm(range(0, nx)):
                    rec[i] = run_alg(stack.data[:, :, i], niterations, sino_id, alg, rec_id)
            else:
                logger.info("Using %s CPU cores to reconstruct %s slices" % (ncores, nx))
                with mp.Pool(ncores) as pool:
                    for i, result in enumerate(
                        pool.starmap(run_alg,
                                     [(stack.data[:, :, i], niterations, sino_id, alg, rec_id) for i in range(0, nx)],)):
                        rec[i] = result
        elif method.lower() == 'dart':
            if ncores == 1:
                for i in tqdm.tqdm(range(0, nx)):
                    rec[i] = run_dart(stack.data[:, :, i], niterations, dart_iterations, p,
                                      alg, proj_id, mask_id, rec_id, sino_id, thresholds, gray_levels)
            else:
                logger.info("Using %s CPU cores to reconstruct %s slices" % (ncores, nx))
                with mp.Pool(ncores) as pool:
                    for i, result in enumerate(
                        pool.starmap(run_dart,
                                     [(stack.data[:, :, i], niterations, dart_iterations, p,
                                       alg, proj_id, mask_id, rec_id, sino_id, thresholds, gray_levels)
                                      for i in range(0, nx)],)):
                        rec[i] = result
    astra.clear()
    return rec


def dart_segment(rec, thresholds, gray_vals):
    bins = np.digitize(rec, bins=thresholds, right=False)
    segmented = np.array(gray_vals)[bins]
    return segmented


def get_dart_boundaries(segmented):
    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]])
    edges = convolve(segmented.astype(np.int32), kernel, mode='constant', cval=0)
    boundaries = edges != 0
    return boundaries


def astra_sirt_error(sinogram, angles, iterations=50, constrain=True, thresh=0, cuda=False):
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

    proj_geom = astra.create_proj_geom("parallel", 1.0, ny, thetas)
    vol_geom = astra.create_vol_geom((ny, ny))
    rec_id = astra.data2d.create("-vol", vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, np.zeros([nangles, ny]))

    if cuda:
        alg_name = "SIRT_CUDA"
        proj_id = astra.create_projector("cuda", proj_geom, vol_geom)
    else:
        alg_name = "SIRT"
        proj_id = astra.create_projector("linear", proj_geom, vol_geom)

    astra.data2d.store(sino_id, sinogram)

    cfg = astra.astra_dict(alg_name)
    cfg["ProjectionDataId"] = sino_id
    cfg["ProjectorId"] = proj_id
    cfg["ReconstructionDataId"] = rec_id
    if constrain:
        cfg["option"] = {}
        cfg["option"]["MinConstraint"] = thresh

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
