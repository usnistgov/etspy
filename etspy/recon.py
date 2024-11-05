# ruff: noqa: UP007
"""Reconstruction module for ETSpy package."""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
from typing import Literal, Optional, Union, cast

import astra
import numpy as np
import tqdm
from dask.base import compute as dask_compute
from dask.delayed import delayed as dask_delayed
from dask.diagnostics.progress import ProgressBar
from scipy.ndimage import convolve, gaussian_filter

ncpus = mp.cpu_count()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_alg(
    sino: np.ndarray,
    iters: int,
    cfg: dict,
    vol_geom: dict,
    proj_geom: dict,
) -> np.ndarray:
    """
    Run CPU-based FBP, SIRT, or SART reconstruction algorithm using dask.

    Parameters
    ----------
    sino
       Sinogram of shape (nangles, ny)
    iters
        Number of iterations for the reconstruction
    cfg
        ASTRA algorithm configuration, as described for each of the algorithms present
        in the :external+astra:doc:`ASTRA docs (Algorithms)<docs/algs/index>`
    vol_geom
        ASTRA volume geometry, as described in the
        :external+astra:doc:`ASTRA docs (Toolbox concepts)<docs/concepts>`
    proj_geom
        ASTRA projection geometry, as described in the
        :external+astra:doc:`ASTRA docs (Toolbox concepts)<docs/concepts>`

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Reconstruction of input sinogram

    Group
    -----
    recon
    """
    proj_id = astra.create_projector("strip", proj_geom, vol_geom)
    rec_id = astra.data2d.create("-vol", vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, sino)
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectorId"] = proj_id
    cfg["ProjectionDataId"] = sino_id
    cfg["ReconstructionDataId"] = rec_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, iters)
    return astra.data2d.get(rec_id)


def run_dart(
    sino: np.ndarray,
    iters: int,
    dart_iters: int,
    p: float,
    thresholds: Union[list, np.ndarray],
    gray_levels: Union[list, np.ndarray],
    cfg: dict,
    vol_geom: dict,
    proj_geom: dict,
) -> np.ndarray:
    """
    Run discrete algebraic reoncsturction technique (DART) algorithm.

    Adapted from pseudo-code published in:
    K. J. Batenburg and J. Sijbers, "DART: A Practical Reconstruction
    Algorithm for Discrete Tomography," doi: 10.1109/TIP.2011.2131661.

    Parameters
    ----------
    sino
       Sinogram of shape (nangles, ny)
    iters
        Number of iterations for the SART reconstruction
    dart_iters
        Number of iterations for the DART reconstruction
    p
        Probability for free pixel determination
    thresholds
        Thresholds for DART reconstruction
    gray_levels
        Gray levels for DART reconstruction
    cfg
        ASTRA algorithm configuration
    vol_geom
        ASTRA volume geometry
    proj_geom
        ASTRA projection geometry

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Reconstruction of input sinogram

    Group
    -----
    recon
    """
    proj_id = astra.create_projector("strip", proj_geom, vol_geom)
    rec_id = astra.data2d.create("-vol", vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, sino)
    mask_id = astra.data2d.create("-vol", vol_geom, 1)
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectorId"] = proj_id
    cfg["ProjectionDataId"] = sino_id
    cfg["ReconstructionDataId"] = rec_id
    alg_id = astra.algorithm.create(cfg)
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


def run(  # noqa: PLR0912, PLR0913, PLR0915
    stack: np.ndarray,
    tilts: np.ndarray,
    method: Literal["FBP", "SIRT", "SART", "DART"],
    niterations: int = 20,
    constrain: bool = False,
    thresh: float = 0,
    cuda: Optional[bool] = None,
    thickness: Optional[int] = None,
    ncores: Optional[int] = None,
    bp_filter: Literal[
        "ram-lak",
        "shepp-logan",
        "cosine",
        "hamming",
        "hann",
        "none",
        "tukey",
        "lanczos",
        "triangular",
        "gaussian",
        "barlett-hann",
        "blackman",
        "nuttall",
        "blackman-harris",
        "blackman-nuttall",
        "flat-top",
        "kaiser",
        "parzen",
        "projection",
        "sinogram",
        "rprojection",
        "rsinogram",
    ] = "shepp-logan",
    gray_levels: Optional[Union[list, np.ndarray]] = None,
    dart_iterations: int = 2,
    p: float = 0.99,
    show_progressbar: bool = True,
) -> np.ndarray:
    """
    Perform reconstruction of input tilt series.

    Parameters
    ----------
    stack
       NumPy array containing the input tilt series for a
       :py:class:`~etspy.base.TomoStack`
    tilts
        The tilt angles for the tilt series (usually found in the
        ``TomoStack.tilts.data`` property). Should be a one-dimensional array,
        so it may be necessary to use the :py:meth:`~numpy.ndarray.squeeze` method
        (`e.g.` ``tilts=stack.tilts.data.squeeze()``).
    method
        Reconstruction algorithm to use.  Must be either 'FBP' (default), 'SIRT',
        'SART', or 'DART
    niterations
        Number of iterations for reconstruction (used with ``SIRT``, ``SART``, and
        ``DART`` methods)
    constrain
        If True, output reconstruction is constrained above value given by
        'thresh'
    thresh
        Value above which to constrain the reconstructed data
    cuda
        If True, use the CUDA-accelerated Astra algorithms. Otherwise,
        use the CPU-based algorithms
    thickness
        Limit for the height of the reconstruction. If ``None``, the y-size
        of the stack is used.
    ncores
        Number of cores to use for multithreaded CPU-based reconstructions
    bp_filter
        Filter to use for filtered backprojection
    gray_levels
        Gray levels for DART reconstruction
    dart_iterations
        Number of DART iterations
    p
        Probability for setting free pixels in DART reconstruction
    show_progressbar
        If True, show a progress bar for the reconstruction. Default is True.

    Returns
    -------
    rec : :py:class:`~numpy.ndarray`
        Containing the reconstructed volume

    Group
    -----
    recon
    """
    if len(stack.shape) == 2:  # noqa: PLR2004
        nangles, ny = stack.shape
        stack = stack[:, :, np.newaxis]
        nx = 1
    else:
        nangles, ny, nx = stack.shape

    thetas = np.pi * tilts / 180.0
    mask_id = None
    thresholds = []

    if thickness is None:
        thickness = ny
    thickness = cast(int, thickness)

    rec = np.zeros((nx, thickness, ny), np.float32)

    proj_geom = astra.create_proj_geom("parallel", 1.0, ny, thetas)
    vol_geom = astra.create_vol_geom((thickness, ny))
    cfg = {}
    cfg["option"] = {}

    if cuda:  # coverage: nocuda
        if method.lower() == "fbp":
            logger.info("Reconstructing with CUDA-accelerated FBP algorithm")
            cfg["type"] = "FBP_CUDA"
            cfg["option"]["FilterType"] = bp_filter.lower()
            niterations = 1
        elif method.lower() == "sirt":
            logger.info(
                "Reconstructing with CUDA-accelerated SIRT algorithm (%s iterations)",
                niterations,
            )
            cfg["type"] = "SIRT_CUDA"
            if constrain:
                cfg["option"]["MinConstraint"] = thresh
        elif method.lower() == "sart":
            logger.info(
                "Reconstructing with CUDA-accelerated SART algorithm (%s iterations)",
                niterations,
            )
            cfg["type"] = "SART_CUDA"
            if constrain:
                cfg["option"]["MinConstraint"] = thresh

        elif method.lower() == "dart":
            logger.info(
                "Reconstructing with CUDA-accelerated DART algorithm (%s iterations)",
                niterations,
            )
            cfg["type"] = "SART_CUDA"
            if gray_levels is None:
                msg = "gray_levels must be provided for DART"
                raise ValueError(msg)
            gray_levels = cast(Union[list, np.ndarray], gray_levels)
            thresholds = [
                (gray_levels[i] + gray_levels[i + 1]) // 2
                for i in range(len(gray_levels) - 1)
            ]
            mask = np.ones([thickness, ny])
            mask_id = astra.data2d.create("-vol", vol_geom, mask)
            cfg["option"]["MinConstraint"] = 0
            cfg["option"]["MaxConstraint"] = 255
            cfg["option"]["ReconstructionMaskId"] = mask_id

        proj_id = astra.create_projector("cuda", proj_geom, vol_geom)
        rec_id = astra.data2d.create("-vol", vol_geom)
        sino_id = astra.data2d.create("-sino", proj_geom, np.zeros([nangles, ny]))
        proj_id = astra.create_projector("cuda", proj_geom, vol_geom)
        cfg["ReconstructionDataId"] = rec_id
        cfg["ProjectorId"] = proj_id
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = rec_id
        alg = astra.algorithm.create(cfg)

        for i in tqdm.tqdm(range(nx), disable=not (show_progressbar)):
            astra.data2d.store(sino_id, stack[:, :, i])
            astra.data2d.store(rec_id, np.zeros([thickness, ny]))
            if method.lower() == "dart":
                astra.data2d.store(mask_id, np.ones([thickness, ny]))
                rec[i, :, :] = run_dart(
                    stack[:, :, i],
                    niterations,
                    dart_iterations,
                    p,
                    thresholds,
                    cast(Union[list, np.ndarray], gray_levels),
                    cfg,
                    vol_geom,
                    proj_geom,
                )
            else:
                astra.algorithm.run(alg, niterations)
                rec[i, :, :] = astra.data2d.get(rec_id)
    else:
        if ncores is None:
            ncores = min(nx, int(0.9 * mp.cpu_count()))

        if method.lower() == "fbp":
            logger.info("Reconstructing with CPU-based FBP algorithm")
            cfg["type"] = "FBP"
            cfg["option"]["FilterType"] = bp_filter.lower()
            niterations = 1
        elif method.lower() == "sirt":
            logger.info("Reconstructing with CPU-based SIRT algorithm")
            cfg["type"] = "SIRT"
            if constrain:
                cfg["option"]["MinConstraint"] = thresh
        elif method.lower() == "sart":
            logger.info("Reconstructing with CPU-based SART algorithm")
            cfg["type"] = "SART"
            if constrain:
                cfg["option"]["MinConstraint"] = thresh
        elif method.lower() == "dart":
            logger.info("Reconstructing with CPU-based DART algorithm")
            cfg["type"] = "SART"
            if gray_levels is None:
                msg = "gray_levels must be provided for DART"
                raise ValueError(msg)
            gray_levels = cast(np.ndarray, gray_levels)  # explicit type-checking cast
            thresholds = [
                (gray_levels[i] + gray_levels[i + 1]) // 2
                for i in range(len(gray_levels) - 1)
            ]
            mask = np.ones([thickness, ny])
            mask_id = astra.data2d.create("-vol", vol_geom, mask)
            cfg["option"]["MinConstraint"] = 0
            cfg["option"]["MaxConstraint"] = 255
            cfg["option"]["ReconstructionMaskId"] = mask_id

        if method.lower() in ["fbp", "sirt", "sart"]:
            tasks = [
                dask_delayed(run_alg)(
                    stack[:, :, i],
                    niterations,
                    cfg,
                    vol_geom,
                    proj_geom,
                )
                for i in range(nx)
            ]
            if show_progressbar:
                with ProgressBar():
                    results = dask_compute(*tasks, num_workers=ncores)
            else:
                results = dask_compute(*tasks, num_workers=ncores)

            for i, result in enumerate(results):
                rec[i] = result
        elif method.lower() == "dart":
            tasks = [
                dask_delayed(run_dart)(
                    stack[:, :, i],
                    niterations,
                    dart_iterations,
                    p,
                    thresholds,
                    gray_levels,
                    cfg,
                    vol_geom,
                    proj_geom,
                )
                for i in range(nx)
            ]
            if show_progressbar:
                with ProgressBar():
                    results = dask_compute(*tasks, num_workers=ncores)
            else:
                results = dask_compute(*tasks, num_workers=ncores)

            for i, result in enumerate(results):
                rec[i] = result

    astra.clear()
    return rec


def dart_segment(
    rec: np.ndarray,
    thresholds: Union[list, np.ndarray],
    gray_vals: Union[list, np.ndarray],
) -> np.ndarray:
    """
    Segmentation step for DART Reconstruction.

    Parameters
    ----------
    rec
       Tomographic reconstruction.
    thresholds
        Threshold values for segmentation.
    gray_vals
        Grayscale values to assign the segmented regions.

    Returns
    -------
    segmented : :py:class:`~numpy.ndarray`
        Segmented version of the reconstruction.

    Group
    -----
    recon
    """
    bins = np.digitize(rec, bins=thresholds, right=False)
    segmented = np.array(gray_vals)[bins]
    return segmented


def get_dart_boundaries(segmented: np.ndarray) -> np.ndarray:
    """
    Boundary step for DART Reconstruction.

    Parameters
    ----------
    segmented
        Segmented reconstruction.

    Returns
    -------
    boundaries : :py:class:`~numpy.ndarray`
        Boundaries of the segmented reconstruction.

    Group
    -----
    recon
    """
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    edges = convolve(segmented.astype(np.int32), kernel, mode="constant", cval=0)
    boundaries = edges != 0
    return boundaries


def astra_error(
    sinogram: np.ndarray,
    angles: np.ndarray,
    method: Literal["SIRT", "SART"] = "SIRT",
    iterations: int = 50,
    constrain: bool = True,
    thresh: float = 0,
    cuda=False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform SIRT reconstruction using the Astra toolbox algorithms.

    Parameters
    ----------
    sinogram
       Tilt series data of the shape ``(n_angles, n_y)``, where
       y is the axis perpendicular to the tilt axis.
    angles
        Projection angles in degrees.
    method
        Reconstruction algorithm use.  Must be 'SIRT' or 'SART'.
    iterations
        Number of iterations for the SIRT reconstruction.
    constrain
        If True, output reconstruction is constrained above value given by
        'thresh'. Default is True.
    thresh
        Value above which to constrain the reconstructed data if ``constrain``
        is ``True``.
    cuda
        If True, use the CUDA-accelerated Astra algorithms. Otherwise,
        use the CPU-based algorithms

    Returns
    -------
    rec : :py:class:`~numpy.ndarray`
        3D array of the form [y, z, x] containing the reconstructed object.
    residual_error : :py:class:`~numpy.ndarray`
        A 1D array of the residual error after each iteration

    Group
    -----
    recon
    """
    thetas = angles * np.pi / 180

    if len(sinogram.shape) != 2:  # noqa: PLR2004
        msg = (
            "Sinogram must be two-dimensional (ntilts, y). Provided shape "
            f"was {sinogram.shape}."
        )
        raise ValueError(msg)

    nangles, ny = sinogram.shape

    if nangles != len(angles):
        msg = (
            "Number of angles must match size of the first dimension of "
            f"the sinogram. [len(angles) was {len(angles)}; sinogram.shape was "
            f"{sinogram.shape}] ({len(angles)} != {nangles})"
        )
        raise ValueError(msg)

    proj_geom = astra.create_proj_geom("parallel", 1.0, ny, thetas)
    vol_geom = astra.create_vol_geom((ny, ny))
    rec_id = astra.data2d.create("-vol", vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, np.zeros([nangles, ny]))

    if cuda:  # coverage: nocuda
        alg_name = method.upper() + "_CUDA"
        proj_id = astra.create_projector("cuda", proj_geom, vol_geom)
    else:
        alg_name = method.upper()
        proj_id = astra.create_projector("strip", proj_geom, vol_geom)

    astra.data2d.store(sino_id, sinogram)

    cfg = astra.astra_dict(alg_name)
    cfg["ProjectionDataId"] = sino_id
    cfg["ProjectorId"] = proj_id
    cfg["ReconstructionDataId"] = rec_id
    if constrain:
        cfg["option"] = {}  # pyright: ignore[reportArgumentType]
        cfg["option"]["MinConstraint"] = thresh

    alg = astra.algorithm.create(cfg)

    rec = np.zeros([iterations, ny, ny], np.float32)
    residual_error = np.zeros(iterations)

    for i in tqdm.tqdm(range(iterations)):
        astra.algorithm.run(alg, 1)
        rec[i] = astra.data2d.get(rec_id)
        if cuda:  # coverage: nocuda
            residual_error[i] = astra.algorithm.get_res_norm(alg)
        else:
            curr_id, curr = astra.create_sino(rec[i], proj_id)
            residual_error[i] = np.linalg.norm(sinogram - curr)
    astra.clear()

    return rec, residual_error
