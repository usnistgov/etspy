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


try:
    import genfire
except ImportError:
    genfire_installed = False
else:
    genfire_installed = True

if genfire_installed:
    import genfire as gf


ncpus = mp.cpu_count()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run(stack, method, iterations=20, constrain=None,
        thresh=None, cuda=None, thickness=None, **kwargs):
    """
    Perform reconstruction of input tilt series.

    Args
    ----------
    stack :TomoStack object
       TomoStack containing the input tilt series
    method : string
        Reconstruction algorithm to use.  Must be 'FBP' (default), 'SIRT',
        or 'GENFIRE'
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
    if stack.metadata.Tomography.tilts is None:
        raise ValueError("Tilts not defined")

    angles = stack.metadata.Tomography.tilts

    if method.lower() == 'genfire':
        print(True)
        if genfire_installed is False:
            raise ValueError('GENFIRE is not installed.')
        padded, npad = pad_square(stack)
        proj = np.transpose(padded.data, [2, 1, 0])

        tiltSeries = np.roll(np.rot90(proj, k=1, axes=(0, 1)), shift=1, axis=0)
        kwargs = {'griddingMethod': 0}
        tiltAngles = stack.metadata.Tomography.tilts
        # Convert to Euler representation assuming single-tilt axis
        eulerAngles = np.zeros((len(tiltAngles), 3))
        eulerAngles[:, 1] = tiltAngles

        # Convert enumerated type to that used by GENFIRE
        gridMap = {0: "FFT", 1: "DFT"}
        kwargs['griddingMethod'] = gridMap[kwargs['griddingMethod']]

        # Assemble rest of parameters
        pars = dict(projections=tiltSeries,
                    eulerAngles=eulerAngles,
                    resultsFilename="recon.mrc",
                    numIterations=10,
                    oversamplingRatio=1,
                    **kwargs)

        # Run the reconstruction
        GF = gf.reconstruct.GenfireReconstructor(**pars)
        results = GF.reconstruct()

        rec = np.rollaxis(results['reconstruction'], 1)
        rec = rec[npad:-npad, :, :]
        return rec

    elif method == 'FBP':
        if cuda is False:
            '''ASTRA weighted-backprojection reconstruction of single slice'''
            logger.info('Reconstructing volume using FBP')
            logger.info('Reconstruction using %s cores' % ncpus)
            pool = mp.Pool(ncpus)
            rec = pool.starmap(astra_fbp,
                               [(stack.data[:, :, i],
                                 angles,
                                 thickness)
                                for i in range(0, stack.data.shape[2])])
            pool.close()
            logger.info('Reconstruction complete')
            if type(rec) is list:
                if len(rec) > 1:
                    rec = np.vstack(rec)
                else:
                    rec = rec[0]
        elif cuda is True:
            '''ASTRA weighted-backprojection CUDA reconstruction of single
            slice'''
            logger.info('Reconstruction volume using FBP_CUDA')
            rec = astra_fbp(stack.data, angles, cuda=True)
            logger.info('Reconstruction complete')
        else:
            raise Exception('Unable to determine CUDA capability')
    elif method == 'SIRT':
        if constrain:
            if not thresh:
                thresh = 0
            logger.info("Reconstructing volume using SIRT w/ "
                        "minimum constraint. Minimum value: %s" % thresh)
        else:
            logger.info("Reconstructing volume using SIRT")
        if cuda is False:
            '''ASTRA SIRT reconstruction'''
            logger.info('Reconstructing volume using %s SIRT iterations'
                        % iterations)
            logger.info('Reconstruction using %s cores' % ncpus)
            pool = mp.Pool(ncpus)
            rec = pool.starmap(astra_sirt,
                               [(stack.data[:, :, i],
                                 angles,
                                 thickness,
                                 iterations,
                                 constrain,
                                 thresh)
                                for i in range(0, stack.data.shape[2])])
            pool.close()
            logger.info('Reconstruction complete')
            if type(rec) is list:
                if len(rec) > 1:
                    rec = np.vstack(rec)
                else:
                    rec = rec[0]
        elif cuda is True:
            '''ASTRA CUDA-accelerated SIRT reconstruction'''
            rec = astra_sirt(stack.data, angles, iterations=iterations,
                             constrain=constrain, thresh=thresh, cuda=True,
                             thickness=thickness)
            logger.info('Reconstruction complete')
        else:
            raise Exception('Unable to determine CUDA capability')
    else:
        raise ValueError('Unknown reconstruction algorithm:' + method)
    return rec


def astra_sirt(stack, angles, thickness=None, iterations=50,
               constrain=True, thresh=0, cuda=False):
    """
    Perform SIRT reconstruction using the Astra toolbox algorithms.

    Args
    ----------
    stack : NumPy array
       Tilt series data either of the form [angles, x] or [angles, y, x] where
       y is the tilt axis and x is the projection axis.
    angles : list or NumPy array
        Projection angles in degrees.
    thickness : int or float
        Number of pixels in the projection (through-thickness) direction
        for the reconstructed volume.  If None, thickness is set to be the
        same as that in the x-direction of the sinogram.
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

    if len(stack.shape) == 2:
        data = np.expand_dims(stack, 2)
    else:
        data = stack
    data = np.rollaxis(data, 2)
    y_pix, n_angles, x_pix = data.shape

    if thickness is None:
        thickness = data.shape[2]

    if cuda:
        chunksize = 128
        rec = np.zeros([y_pix, thickness, x_pix], data.dtype)
        nchunks = int(np.ceil(y_pix / chunksize))

        if nchunks == 1:
            chunksize = y_pix
            chunk_list = [[0, y_pix]]
        else:
            chunk_list = [None] * nchunks
            for i in range(0, int(y_pix / chunksize)):
                chunk_list[i] = [i * chunksize, (i + 1) * chunksize]
            if (np.mod(y_pix, chunksize) != 0) and (nchunks > 1):
                chunk_list[-1] = [chunk_list[-2][1], chunk_list[-2][1] + np.mod(y_pix, chunksize)]

        for i in range(0, len(chunk_list)):
            chunk = data[chunk_list[i][0]:chunk_list[i][1], :, :]
            vol_geom = astra.create_vol_geom(thickness, x_pix, chunk.shape[0])
            proj_geom = astra.create_proj_geom('parallel3d', 1, 1,
                                               chunk.shape[0], x_pix, thetas)
            data_id = astra.data3d.create('-proj3d', proj_geom, chunk)
            rec_id = astra.data3d.create('-vol', vol_geom)

            cfg = astra.astra_dict('SIRT3D_CUDA')
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = data_id
            if constrain:
                cfg['option'] = {}
                cfg['option']['MinConstraint'] = thresh

            alg_id = astra.algorithm.create(cfg)

            astra.algorithm.run(alg_id, iterations)

            rec[chunk_list[i][0]:chunk_list[i][1], :, :] =\
                astra.data3d.get(rec_id)
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(rec_id)
            astra.data3d.delete(data_id)

    else:
        rec = np.zeros([y_pix, thickness, x_pix], data.dtype)
        vol_geom = astra.create_vol_geom(thickness, x_pix)
        proj_geom = astra.create_proj_geom('parallel', 1.0,
                                           x_pix, thetas)
        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
        rec_id = astra.data2d.create('-vol', vol_geom)
        sinogram_id = astra.data2d.create('-sino', proj_geom, data[0, :, :])
        cfg = astra.astra_dict('SIRT')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ProjectorId'] = proj_id
        if constrain:
            cfg['option'] = {}
            cfg['option']['MinConstraint'] = thresh

        alg_id = astra.algorithm.create(cfg)
        for i in range(0, y_pix):
            astra.data2d.store(sinogram_id, data[i, :, :])
            astra.algorithm.run(alg_id, iterations)
            rec[i, :, :] = astra.data2d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
        astra.clear()
    return rec


def astra_fbp(stack, angles, thickness=None, cuda=None):
    """
    Perform SIRT reconstruction using the Astra toolbox algorithms.

    Args
    ----------
    stack : NumPy array
       Tilt series data either of the form [angles, x] or [angles, y, x] where
       y is the tilt axis and x is the projection axis.
    angles : list or NumPy array
        Projection angles in degrees.
    thickness : int or float
        Number of pixels in the projection (through-thickness) direction
        for the reconstructed volume.  If None, thickness is set to be the
        same as that in the x-direction of the sinogram.
    cuda : boolean
        If True, use the CUDA-accelerated Astra algorithms. Otherwise,
        use the CPU-based algorithms
    Returns
    ----------
    rec : Numpy array
        3D array of the form [y, z, x] containing the reconstructed object.

    """
    thetas = angles * np.pi / 180
    if len(stack.shape) == 2:
        data = np.expand_dims(stack, 2)
    else:
        data = stack
    data = np.rollaxis(data, 2)
    y_pix, n_angles, x_pix = data.shape

    if thickness is None:
        thickness = data.shape[2]

    if cuda:
        rec = np.zeros([y_pix, thickness, x_pix], data.dtype)

        vol_geom = astra.create_vol_geom(thickness, x_pix)
        proj_geom = astra.create_proj_geom('parallel', 1,
                                           x_pix, thetas)
        data_id = astra.data2d.create('-sino', proj_geom)
        rec_id = astra.data2d.create('-vol', vol_geom)

        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = data_id
        cfg['option'] = {}
        cfg['option']['FilterType'] = 'ram-lak'

        alg_id = astra.algorithm.create(cfg)

        for i in range(0, y_pix):
            astra.data2d.store(data_id, data[i, :, :])
            astra.algorithm.run(alg_id)
            rec[i, :, :] = astra.data2d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(data_id)

    else:
        rec = np.zeros([y_pix, thickness, x_pix], data.dtype)
        vol_geom = astra.create_vol_geom(thickness, x_pix)
        proj_geom = astra.create_proj_geom('parallel', 1.0,
                                           x_pix, thetas)
        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
        rec_id = astra.data2d.create('-vol', vol_geom)
        sinogram_id = astra.data2d.create('-sino', proj_geom, data[0, :, :])
        cfg = astra.astra_dict('FBP')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ProjectorId'] = proj_id
        cfg['option'] = {}
        cfg['option']['FilterType'] = 'ram-lak'

        alg_id = astra.algorithm.create(cfg)
        for i in range(0, y_pix):
            astra.data2d.store(sinogram_id, data[i, :, :])
            astra.algorithm.run(alg_id)
            rec[i, :, :] = astra.data2d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
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
    stack : NumPy array
       Tilt series data either of the form [angles, x] or [angles, y, x] where
       y is the tilt axis and x is the projection axis.
    angles : list or NumPy array
        Projection angles in degrees.
    thickness : int or float
        Number of pixels in the projection (through-thickness) direction
        for the reconstructed volume.  If None, thickness is set to be the
        same as that in the x-direction of the sinogram.
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
    if len(sinogram.data.shape) == 3:
        sino = sinogram.data[:, :, 0]
    else:
        sino = sinogram.data
    thetas = angles * np.pi / 180

    n_angles, x_pix = sino.shape

    thickness = x_pix

    rec = np.zeros([thickness, x_pix], sino.dtype)
    vol_geom = astra.create_vol_geom(thickness, x_pix)
    proj_geom = astra.create_proj_geom('parallel', 1, x_pix, thetas)
    data_id = astra.data2d.create('-sino', proj_geom, sino)
    rec_id = astra.data2d.create('-vol', vol_geom)

    if cuda:
        cfg = astra.astra_dict('SIRT_CUDA')
    else:
        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
        cfg = astra.astra_dict('SIRT')
        cfg['ProjectorId'] = proj_id

    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = data_id
    if constrain:
        cfg['option'] = {}
        cfg['option']['MinConstraint'] = thresh

    alg_id = astra.algorithm.create(cfg)
    residual_error = np.zeros(iterations)
    rec_stack = np.zeros([iterations, thickness, x_pix])

    for i in range(iterations):
        astra.algorithm.run(alg_id, 1)
        rec = astra.data2d.get(rec_id)
        rec_stack[i] = rec

        forward_project = astra_project(rec, angles=angles, cuda=cuda)[:, :, 0]
        residual_error[i] = np.sqrt(np.square(forward_project - sino).sum())

    astra.clear()
    return rec_stack, residual_error


def pad_square(stack):
    """
    Pad a tilt series to square dimensions for GENFIRE reconstruction.

    Args
    ----------
    stack : TomoStack
       Tilt series to be padded.

    Returns
    ----------
    padded : TomoStack
        Padded version of tilt series.

    row_pad : int
        Number of pixels added to each side of the stack.

    """
    padded = stack.deepcopy()
    ntilts, nrows, ncols = stack.data.shape
    if nrows > ncols:
        if np.mod(nrows, 2) != 0:
            row_pad = 1
            nrows += 1
        else:
            row_pad = 0
        col_pad = round((nrows - ncols) / 2)
        padded.data = np.pad(padded.data, [[0, 0], [row_pad, 0], [col_pad, col_pad]])
        return padded, col_pad
    elif ncols > ncols:
        if np.mod(ncols, 2) != 0:
            col_pad = 1
            ncols += 1
        else:
            col_pad = 0
        row_pad = round((ncols - nrows) / 2)
        padded.data = np.pad(padded.data, [[0, 0], [row_pad, row_pad], [col_pad, 0]])
        return padded, row_pad
