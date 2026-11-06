# ruff: noqa: UP007
"""Reconstruction module for ETSpy package."""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
from typing import Literal, Union, cast

import astra
import numpy as np
import tqdm
from dask.base import compute as dask_compute
from dask.delayed import delayed as dask_delayed
from dask.diagnostics.progress import ProgressBar
from scipy.ndimage import convolve, gaussian_filter

ncpus = mp.cpu_count()
logger = logging.getLogger(__name__)


def create_projector(proj_geom: dict, vol_geom: dict, cuda: bool = False) -> int:
    """
    Create the ASTRA projector used for all ETSpy projections.

    ETSpy uses the same forward model everywhere: the Joseph (linear
    interpolation) projector, which is ASTRA's ``"linear"`` projector on the CPU
    and its ``"cuda"`` projector on the GPU. Using one model keeps
    reconstructions, forward projections, and simulations consistent with each
    other and between CPU and GPU.

    Parameters
    ----------
    proj_geom
        ASTRA projection geometry
    vol_geom
        ASTRA volume geometry
    cuda
        If True, create the GPU projector. Otherwise, create the CPU projector.

    Returns
    -------
    proj_id : int
        ID of the ASTRA projector. It should be deleted with
        ``astra.projector.delete`` when no longer needed.

    Group
    -----
    recon
    """
    return astra.create_projector("cuda" if cuda else "linear", proj_geom, vol_geom)


def resolve_cuda(cuda: bool | None) -> bool:
    """
    Decide whether to use CUDA (GPU) reconstruction and projection.

    Parameters
    ----------
    cuda
        ``True`` to use CUDA, ``False`` to use the CPU, or ``None`` to use CUDA if
        :py:func:`astra.astra.use_cuda` detects a CUDA GPU.

    Returns
    -------
    cuda : bool
        Whether to use CUDA

    Raises
    ------
    RuntimeError
        If ``cuda`` is ``True`` but ASTRA does not detect a CUDA GPU (ASTRA would
        otherwise fail with a low-level error, or for 3D operations crash the
        Python process)

    Group
    -----
    recon
    """
    if cuda is None:
        cuda = bool(astra.use_cuda())
        logger.info("CUDA %s with Astra", "detected" if cuda else "not detected")
        return cuda
    if cuda and not astra.use_cuda():
        msg = (
            "CUDA was requested (cuda=True) but ASTRA did not detect a CUDA GPU. "
            "Use cuda=False to run on the CPU."
        )
        raise RuntimeError(msg)
    return bool(cuda)


# Default number of iterations for each iterative algorithm. Iterations are full
# passes through all of the projections; for SART (and the SART steps of DART), ETSpy
# converts them to ASTRA iterations, each of which uses a single projection.
DEFAULT_ITERATIONS = {"SIRT": 50, "SART": 5, "DART": 5}


def astra_iterations(
    method: str,
    iterations: int | None,
    nangles: int,
) -> tuple[int, int]:
    """
    Convert a number of passes through the projections to ASTRA iterations.

    In ASTRA, an iteration of SIRT uses all of the projections, but an iteration of
    SART uses a single projection. ETSpy always counts iterations as full passes
    through the projections, so for SART (and the SART steps of DART) the number
    of passes is multiplied by the number of projections.

    Parameters
    ----------
    method
        Reconstruction algorithm (``"FBP"``, ``"SIRT"``, ``"SART"``, or
        ``"DART"``)
    iterations
        Number of passes through the projections. If ``None``, the default for
        the algorithm (:py:data:`DEFAULT_ITERATIONS`) is used.
    nangles
        Number of projections

    Returns
    -------
    passes : int
        Number of passes through the projections
    n_astra : int
        Number of ASTRA iterations to run

    Group
    -----
    recon
    """
    method = method.upper()
    if method == "FBP":
        return 1, 1
    passes = DEFAULT_ITERATIONS[method] if iterations is None else int(iterations)
    if method in ("SART", "DART"):
        if iterations is not None and passes >= nangles:
            logger.warning(
                "%s iterations are counted as full passes through all %d "
                "projections, so %d iterations will run %d single-projection ASTRA "
                "SART iterations. (Earlier versions of ETSpy counted single "
                "projections; SART typically needs only a few passes.)",
                method,
                nangles,
                passes,
                passes * nangles,
            )
        return passes, passes * nangles
    return passes, passes


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
        in the :external+astra:doc:`ASTRA docs (Algorithms)<docs/algs/index>`. The
        data and projector IDs are filled in by this function, and the dictionary
        passed in is not modified.
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
    proj_id = create_projector(proj_geom, vol_geom)
    rec_id = astra.data2d.create("-vol", vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, sino)
    # Copy so that the caller's configuration, which is shared between the dask
    # threads reconstructing different slices, is never mutated
    cfg = copy.deepcopy(cfg)
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectorId"] = proj_id
    cfg["ProjectionDataId"] = sino_id
    alg_id = astra.algorithm.create(cfg)
    try:
        astra.algorithm.run(alg_id, iters)
        return astra.data2d.get(rec_id)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data2d.delete([rec_id, sino_id])
        astra.projector.delete(proj_id)


def _forward_project(rec: np.ndarray, proj_id: int) -> np.ndarray:
    """Forward project a slice, deleting the temporary ASTRA sinogram object."""
    sino_id, sino = astra.create_sino(rec, proj_id)
    astra.data2d.delete(sino_id)
    return sino


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
    cuda: bool = False,
) -> np.ndarray:
    """
    Run discrete algebraic reconstruction technique (DART) algorithm.

    Adapted from pseudo-code published in:
    K. J. Batenburg and J. Sijbers, "DART: A Practical Reconstruction
    Algorithm for Discrete Tomography," doi: 10.1109/TIP.2011.2131661.

    Each DART iteration segments the current reconstruction, holds the
    non-boundary pixels (minus a random subset) fixed at their segmented
    gray level, and re-reconstructs only the remaining free pixels from the
    residual sinogram (measured sinogram minus the forward projection of the
    fixed pixels). The free pixels are then smoothed before the next iteration.

    Parameters
    ----------
    sino
       Sinogram of shape (nangles, ny)
    iters
        Number of ASTRA SART iterations for each SART reconstruction (each uses a
        single projection; see :py:func:`astra_iterations`)
    dart_iters
        Number of iterations for the DART reconstruction
    p
        Probability for free pixel determination
    thresholds
        Thresholds for DART reconstruction
    gray_levels
        Gray levels for DART reconstruction
    cfg
        ASTRA algorithm configuration (a SART or SART_CUDA configuration). The
        data, projector, and mask IDs are filled in by this function, and the
        dictionary passed in is not modified.
    vol_geom
        ASTRA volume geometry
    proj_geom
        ASTRA projection geometry
    cuda
        If True, use the GPU projector for forward projection. Otherwise, use the
        CPU projector (see :py:func:`create_projector`).

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Reconstruction of input sinogram

    Group
    -----
    recon
    """
    proj_id = create_projector(proj_geom, vol_geom, cuda=cuda)
    rec_id = astra.data2d.create("-vol", vol_geom, 0)
    sino_id = astra.data2d.create("-sino", proj_geom, sino)
    mask_id = astra.data2d.create("-vol", vol_geom, 1)

    # Copy so that the caller's configuration (which may be shared between
    # threads) is never mutated
    cfg = copy.deepcopy(cfg)
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectorId"] = proj_id
    cfg["ProjectionDataId"] = sino_id
    cfg["option"]["ReconstructionMaskId"] = mask_id
    alg_id = astra.algorithm.create(cfg)

    try:
        # Initial continuous reconstruction of all pixels
        astra.algorithm.run(alg_id, iters)
        dart_rec = astra.data2d.get(rec_id)

        for j in range(dart_iters):
            segmented = dart_segment(dart_rec, thresholds, gray_levels)
            boundary = get_dart_boundaries(segmented)

            # Free pixels are the boundary pixels plus a random subset of the
            # rest (each non-boundary pixel is free with probability 1 - p)
            free = np.logical_or(boundary, np.random.rand(*dart_rec.shape) >= p)

            # Fixed pixels take their segmented values; free pixels start from
            # zero so the forward projection of the volume only contains the
            # fixed pixels
            fixed_rec = np.where(free, 0, segmented).astype(np.float32)
            fixed_sino = _forward_project(fixed_rec, proj_id)

            # Reconstruct only the free pixels from the residual sinogram
            astra.data2d.store(rec_id, np.where(free, dart_rec, 0))
            astra.data2d.store(mask_id, free.astype(np.float32))
            astra.data2d.store(sino_id, sino - fixed_sino)
            astra.algorithm.run(alg_id, iters)
            free_rec = astra.data2d.get(rec_id)

            dart_rec = np.where(free, free_rec, fixed_rec)

            # Smooth the free pixels before the next iteration
            if j < dart_iters - 1:
                smooth = gaussian_filter(dart_rec, sigma=1)
                dart_rec[free] = smooth[free]
    finally:
        astra.algorithm.delete(alg_id)
        astra.data2d.delete([rec_id, sino_id, mask_id])
        astra.projector.delete(proj_id)
    return dart_rec


def run(  # noqa: PLR0912, PLR0913, PLR0915
    stack: np.ndarray,
    tilts: np.ndarray,
    method: Literal["FBP", "SIRT", "SART", "DART"],
    niterations: int | None = None,
    constrain: bool = False,
    thresh: float = 0,
    cuda: bool | None = None,
    thickness: int | None = None,
    ncores: int | None = None,
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
    gray_levels: Union[list, np.ndarray] | None = None,
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
        ``DART`` methods). Each iteration is a full pass through all of the
        projections, for SART as well as SIRT (see :py:func:`astra_iterations`).
        For DART, this is the number of SART passes in each DART step. If
        ``None``, the default for the algorithm is used (SIRT: 50, SART: 5, DART:
        5).
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
    thresholds = np.array([])
    passes, niterations = astra_iterations(method, niterations, nangles)

    if thickness is None:
        thickness = ny
    thickness = cast("int", thickness)

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
                passes,
            )
            cfg["type"] = "SIRT_CUDA"
            if constrain:
                cfg["option"]["MinConstraint"] = thresh
        elif method.lower() == "sart":
            logger.info(
                "Reconstructing with CUDA-accelerated SART algorithm (%s iterations)",
                passes,
            )
            cfg["type"] = "SART_CUDA"
            if constrain:
                cfg["option"]["MinConstraint"] = thresh

        elif method.lower() == "dart":
            logger.info(
                "Reconstructing with CUDA-accelerated DART algorithm (%s iterations)",
                passes,
            )
            cfg["type"] = "SART_CUDA"
            gray_levels, thresholds = _dart_setup(gray_levels, cfg)

        proj_id = create_projector(proj_geom, vol_geom, cuda=True)
        rec_id = astra.data2d.create("-vol", vol_geom)
        sino_id = astra.data2d.create("-sino", proj_geom, np.zeros([nangles, ny]))
        cfg["ReconstructionDataId"] = rec_id
        cfg["ProjectorId"] = proj_id
        cfg["ProjectionDataId"] = sino_id
        alg = astra.algorithm.create(cfg)

        try:
            for i in tqdm.tqdm(range(nx), disable=not (show_progressbar)):
                astra.data2d.store(sino_id, stack[:, :, i])
                astra.data2d.store(rec_id, np.zeros([thickness, ny]))
                if method.lower() == "dart":
                    rec[i, :, :] = run_dart(
                        stack[:, :, i],
                        niterations,
                        dart_iterations,
                        p,
                        thresholds,
                        gray_levels,
                        cfg,
                        vol_geom,
                        proj_geom,
                        cuda=True,
                    )
                else:
                    astra.algorithm.run(alg, niterations)
                    rec[i, :, :] = astra.data2d.get(rec_id)
        finally:
            astra.algorithm.delete(alg)
            astra.data2d.delete([rec_id, sino_id])
            astra.projector.delete(proj_id)
    else:
        if ncores is None:
            ncores = min(nx, int(0.9 * mp.cpu_count()))

        if method.lower() == "fbp":
            logger.info("Reconstructing with CPU-based FBP algorithm")
            cfg["type"] = "FBP"
            cfg["option"]["FilterType"] = bp_filter.lower()
            niterations = 1
        elif method.lower() == "sirt":
            logger.info(
                "Reconstructing with CPU-based SIRT algorithm (%s iterations)",
                passes,
            )
            cfg["type"] = "SIRT"
            if constrain:
                cfg["option"]["MinConstraint"] = thresh
        elif method.lower() == "sart":
            logger.info(
                "Reconstructing with CPU-based SART algorithm (%s iterations)",
                passes,
            )
            cfg["type"] = "SART"
            if constrain:
                cfg["option"]["MinConstraint"] = thresh
        elif method.lower() == "dart":
            logger.info(
                "Reconstructing with CPU-based DART algorithm (%s iterations)",
                passes,
            )
            cfg["type"] = "SART"
            gray_levels, thresholds = _dart_setup(gray_levels, cfg)

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

    return rec


def _dart_setup(
    gray_levels: Union[list, np.ndarray] | None,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate DART gray levels, calculate thresholds, and set SART constraints.

    Thresholds are the midpoints between adjacent (sorted) gray levels, and the
    SART reconstruction is constrained to the range spanned by the gray levels.
    """
    if gray_levels is None:
        msg = "gray_levels must be provided for DART"
        raise ValueError(msg)
    gray_levels = np.sort(np.asarray(gray_levels, dtype=np.float32))
    thresholds = (gray_levels[:-1] + gray_levels[1:]) / 2
    cfg["option"]["MinConstraint"] = float(gray_levels[0])
    cfg["option"]["MaxConstraint"] = float(gray_levels[-1])
    return gray_levels, thresholds


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
    iterations: int | None = None,
    constrain: bool = True,
    thresh: float = 0,
    cuda=False,
    show_progressbar: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform a SIRT or SART reconstruction, recording the result after each iteration.

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
        Number of iterations, each a full pass through all of the projections
        (see :py:func:`astra_iterations`). If ``None``, the default for the
        algorithm is used (SIRT: 50, SART: 5).
    constrain
        If True, output reconstruction is constrained above value given by
        'thresh'. Default is True.
    thresh
        Value above which to constrain the reconstructed data if ``constrain``
        is ``True``.
    cuda
        If True, use the CUDA-accelerated Astra algorithms. Otherwise,
        use the CPU-based algorithms
    show_progressbar
        If True (default), show a progress bar over the iterations.

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

    alg_name = method.upper() + "_CUDA" if cuda else method.upper()
    proj_id = create_projector(proj_geom, vol_geom, cuda=cuda)

    astra.data2d.store(sino_id, sinogram)

    cfg = astra.astra_dict(alg_name)
    cfg["ProjectionDataId"] = sino_id
    cfg["ProjectorId"] = proj_id
    cfg["ReconstructionDataId"] = rec_id
    if constrain:
        cfg["option"] = {}  # pyright: ignore[reportArgumentType]
        cfg["option"]["MinConstraint"] = thresh

    alg = astra.algorithm.create(cfg)

    passes, n_astra = astra_iterations(method, iterations, nangles)
    per_pass = n_astra // passes
    rec = np.zeros([passes, ny, ny], np.float32)
    residual_error = np.zeros(passes)

    try:
        for i in tqdm.tqdm(range(passes), disable=not show_progressbar):
            astra.algorithm.run(alg, per_pass)
            rec[i] = astra.data2d.get(rec_id)
            if cuda:  # coverage: nocuda
                residual_error[i] = astra.algorithm.get_res_norm(alg)
            else:
                curr = _forward_project(rec[i], proj_id)
                residual_error[i] = np.linalg.norm(sinogram - curr)
    finally:
        astra.algorithm.delete(alg)
        astra.data2d.delete([rec_id, sino_id])
        astra.projector.delete(proj_id)

    return rec, residual_error
