# ruff: noqa: UP007
"""Reconstruction module for ETSpy package."""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Literal, Union

import astra
import numpy as np
import tqdm
from dask.base import compute as dask_compute
from dask.delayed import delayed as dask_delayed
from dask.diagnostics.progress import ProgressBar
from scipy.ndimage import convolve, gaussian_filter

from etspy import FbpMethodType, RngType, _as_rng

ncpus = mp.cpu_count()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _AstraObjects:
    """Create ASTRA objects and delete all of them when closed."""

    def __init__(self):
        self._algorithms: list[int] = []
        self._data: list[int] = []
        self._projectors: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        """Delete every object created through this instance."""
        for alg_id in self._algorithms:
            astra.algorithm.delete(alg_id)
        if self._data:
            astra.data2d.delete(self._data)
        for proj_id in self._projectors:
            astra.projector.delete(proj_id)
        self._algorithms, self._data, self._projectors = [], [], []

    def projector(self, kind: str, proj_geom: dict, vol_geom: dict) -> int:
        """Create a 2D projector."""
        proj_id = astra.create_projector(kind, proj_geom, vol_geom)
        self._projectors.append(proj_id)
        return proj_id

    def data2d(self, kind: str, geometry: dict, data=None) -> int:
        """Create a 2D data object."""
        data_id = astra.data2d.create(kind, geometry, data)
        self._data.append(data_id)
        return data_id

    def algorithm(self, cfg: dict) -> int:
        """Create an algorithm."""
        alg_id = astra.algorithm.create(cfg)
        self._algorithms.append(alg_id)
        return alg_id

    def forward_project(self, image: np.ndarray, proj_id: int) -> np.ndarray:
        """Forward project an image, without keeping the sinogram object."""
        sino_id, sino = astra.create_sino(image, proj_id)
        astra.data2d.delete(sino_id)
        return sino


def _resolve_cuda(cuda: bool | None) -> bool:
    """Return ``cuda``, or whether ASTRA can use CUDA if ``cuda`` is ``None``."""
    if cuda is not None:
        return cuda
    if astra.use_cuda():  # coverage: nocuda
        logger.info("CUDA detected with Astra")
        return True
    logger.info("CUDA not detected with Astra")
    return False


def _copy_cfg(cfg: dict) -> dict:
    """Copy an algorithm configuration so per-slice IDs don't leak between tasks."""
    return {**cfg, "option": dict(cfg.get("option", {}))}


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
    cfg = _copy_cfg(cfg)
    with _AstraObjects() as astra_objects:
        rec_id = astra_objects.data2d("-vol", vol_geom)
        cfg["ProjectorId"] = astra_objects.projector("strip", proj_geom, vol_geom)
        cfg["ProjectionDataId"] = astra_objects.data2d("-sino", proj_geom, sino)
        cfg["ReconstructionDataId"] = rec_id
        astra.algorithm.run(astra_objects.algorithm(cfg), iters)
        return astra.data2d.get(rec_id)


def run_dart(  # noqa: PLR0913
    sino: np.ndarray,
    iters: int,
    dart_iters: int,
    p: float,
    thresholds: Union[list, np.ndarray],
    gray_levels: Union[list, np.ndarray],
    cfg: dict,
    vol_geom: dict,
    proj_geom: dict,
    rng: RngType = None,
    projector: Literal["strip", "cuda"] = "strip",
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
    rng
        Random number generator or seed used to select free pixels. If ``None``, a
        generator is seeded from NumPy's global random state.
    projector
        ASTRA projector type; ``"cuda"`` for CUDA algorithms.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Reconstruction of input sinogram

    Group
    -----
    recon
    """
    rng = _as_rng(rng)
    cfg = _copy_cfg(cfg)
    with _AstraObjects() as astra_objects:
        rec_id = astra_objects.data2d("-vol", vol_geom)
        mask_id = astra_objects.data2d("-vol", vol_geom, 1)
        proj_id = astra_objects.projector(projector, proj_geom, vol_geom)
        sino_id = astra_objects.data2d("-sino", proj_geom, sino)
        cfg["ProjectorId"] = proj_id
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = rec_id
        cfg["option"]["ReconstructionMaskId"] = mask_id
        alg_id = astra_objects.algorithm(cfg)
        astra.algorithm.run(alg_id, iters)
        dart_rec = astra.data2d.get(rec_id)

        for j in range(dart_iters):
            segmented = dart_segment(dart_rec, thresholds, gray_levels)
            boundary = get_dart_boundaries(segmented)
            free = np.logical_or(boundary, rng.random(dart_rec.shape) < 1 - p)

            # fix non-free pixels at their gray level and reconstruct the free
            # pixels from the sinogram remaining after removing the fixed ones
            dart_rec[~free] = segmented[~free]
            fixed_rec = np.where(free, 0, dart_rec)
            free_sino = sino - astra_objects.forward_project(fixed_rec, proj_id)
            astra.data2d.store(rec_id, dart_rec)
            astra.data2d.store(mask_id, free.astype(np.float32))
            astra.data2d.store(sino_id, free_sino)
            astra.algorithm.run(alg_id, iters)
            dart_rec = astra.data2d.get(rec_id)

            if j < dart_iters - 1:
                smooth = gaussian_filter(dart_rec, sigma=1)
                dart_rec[free] = smooth[free]
    return dart_rec


def run(  # noqa: PLR0912, PLR0913, PLR0915
    stack: np.ndarray,
    tilts: np.ndarray,
    method: Literal["FBP", "SIRT", "SART", "DART"],
    niterations: int = 20,
    constrain: bool = False,
    thresh: float = 0,
    cuda: bool | None = None,
    thickness: int | None = None,
    ncores: int | None = None,
    bp_filter: FbpMethodType = "shepp-logan",
    gray_levels: Union[list, np.ndarray] | None = None,
    dart_iterations: int = 2,
    p: float = 0.99,
    show_progressbar: bool = True,
    rng: RngType = None,
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
        ``TomoStack.tilts.data`` property).
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
    rng
        Random number generator or seed used by DART to select free pixels. If
        ``None``, a generator is seeded from NumPy's global random state.

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

    thetas = np.deg2rad(np.asarray(tilts, dtype=float).ravel())
    method = method.lower()
    device = "CUDA-accelerated" if cuda else "CPU-based"
    suffix = "_CUDA" if cuda else ""
    thresholds = np.array([])

    if thickness is None:
        thickness = ny

    rec = np.zeros((nx, thickness, ny), np.float32)

    proj_geom = astra.create_proj_geom("parallel", 1.0, ny, thetas)
    vol_geom = astra.create_vol_geom((thickness, ny))
    cfg = {"option": {}}

    if method == "fbp":
        logger.info("Reconstructing with %s FBP algorithm", device)
        cfg["type"] = "FBP" + suffix
        cfg["option"]["FilterType"] = bp_filter.lower()
        niterations = 1
    elif method in ("sirt", "sart"):
        logger.info(
            "Reconstructing with %s %s algorithm (%s iterations)",
            device,
            method.upper(),
            niterations,
        )
        cfg["type"] = method.upper() + suffix
        if constrain:
            cfg["option"]["MinConstraint"] = thresh
    elif method == "dart":
        logger.info(
            "Reconstructing with %s DART algorithm (%s iterations)",
            device,
            niterations,
        )
        if gray_levels is None:
            msg = "gray_levels must be provided for DART"
            raise ValueError(msg)
        cfg["type"] = "SART" + suffix
        gray_levels = np.sort(np.asarray(gray_levels, dtype=float))
        thresholds = (gray_levels[:-1] + gray_levels[1:]) / 2
        cfg["option"]["MinConstraint"] = float(gray_levels[0])
        cfg["option"]["MaxConstraint"] = float(gray_levels[-1])
    else:
        msg = f'Invalid reconstruction method "{method}"'
        raise ValueError(msg)

    # independent streams so parallel slices never share a generator
    rngs = _as_rng(rng).spawn(nx) if method == "dart" else [None] * nx

    if cuda:  # coverage: nocuda
        with _AstraObjects() as astra_objects:
            if method != "dart":
                rec_id = astra_objects.data2d("-vol", vol_geom)
                sino_id = astra_objects.data2d(
                    "-sino",
                    proj_geom,
                    np.zeros([nangles, ny]),
                )
                cfg["ProjectorId"] = astra_objects.projector(
                    "cuda",
                    proj_geom,
                    vol_geom,
                )
                cfg["ProjectionDataId"] = sino_id
                cfg["ReconstructionDataId"] = rec_id
                alg_id = astra_objects.algorithm(cfg)

            for i in tqdm.tqdm(range(nx), disable=not show_progressbar):
                if method == "dart":
                    rec[i] = run_dart(
                        stack[:, :, i],
                        niterations,
                        dart_iterations,
                        p,
                        thresholds,
                        gray_levels,
                        cfg,
                        vol_geom,
                        proj_geom,
                        rng=rngs[i],
                        projector="cuda",
                    )
                else:
                    astra.data2d.store(sino_id, stack[:, :, i])
                    astra.data2d.store(rec_id, np.zeros([thickness, ny]))
                    astra.algorithm.run(alg_id, niterations)
                    rec[i] = astra.data2d.get(rec_id)
    else:
        if ncores is None:
            ncores = max(1, min(nx, int(0.9 * mp.cpu_count())))

        if method == "dart":
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
                    rng=rngs[i],
                )
                for i in range(nx)
            ]
        else:
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
    rec = np.zeros([iterations, ny, ny], np.float32)
    residual_error = np.zeros(iterations)

    with _AstraObjects() as astra_objects:
        rec_id = astra_objects.data2d("-vol", vol_geom)
        sino_id = astra_objects.data2d("-sino", proj_geom, sinogram)

        if cuda:  # coverage: nocuda
            alg_name = method.upper() + "_CUDA"
            proj_id = astra_objects.projector("cuda", proj_geom, vol_geom)
        else:
            alg_name = method.upper()
            proj_id = astra_objects.projector("strip", proj_geom, vol_geom)

        cfg = astra.astra_dict(alg_name)
        cfg["ProjectionDataId"] = sino_id
        cfg["ProjectorId"] = proj_id
        cfg["ReconstructionDataId"] = rec_id
        if constrain:
            cfg["option"] = {}  # pyright: ignore[reportArgumentType]
            cfg["option"]["MinConstraint"] = thresh

        alg = astra_objects.algorithm(cfg)

        for i in tqdm.tqdm(range(iterations)):
            astra.algorithm.run(alg, 1)
            rec[i] = astra.data2d.get(rec_id)
            if cuda:  # coverage: nocuda
                residual_error[i] = astra.algorithm.get_res_norm(alg)
            else:
                curr = astra_objects.forward_project(rec[i], proj_id)
                residual_error[i] = np.linalg.norm(sinogram - curr)

    return rec, residual_error
