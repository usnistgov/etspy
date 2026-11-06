"""
Total variation (TV) regularized reconstruction for ETSpy.

This module contains the TV operators used by the TV-regularized reconstruction
algorithms. Volumes are arrays of shape ``(nx, nz, ny)`` (the
:py:class:`~etspy.base.RecStack` order), where ``x`` is parallel to the tilt
axis. With ``tv_dims=3`` the TV couples neighbouring slices (differences along
all three axes); with ``tv_dims=2`` only the in-slice axes (``z`` and ``y``) are
used, so each slice is treated independently.

The operators work with NumPy arrays, or with CuPy arrays on the GPU.

The TV methods are implemented from the published algorithms:

- E. Y. Sidky and X. Pan, "Image reconstruction in circular cone-beam computed
  tomography by constrained, total-variation minimization", Physics in Medicine
  and Biology 53 (2008) 4777. https://doi.org/10.1088/0031-9155/53/17/021
- A. Beck and M. Teboulle, "Fast gradient-based algorithms for constrained total
  variation image denoising and deblurring problems", IEEE Transactions on Image
  Processing 18 (2009) 2419. https://doi.org/10.1109/TIP.2009.2028250

"""

import logging
import multiprocessing as mp
from typing import Any, Literal

import astra
import numpy as np
import tqdm
from dask.base import compute as dask_compute
from dask.delayed import delayed as dask_delayed
from dask.diagnostics.progress import ProgressBar

from etspy.recon import create_projector

logger = logging.getLogger(__name__)

TVDims = Literal[2, 3]

# smoothing of the TV near zero gradient (for volumes scaled to a maximum of ~1)
DEFAULT_TV_EPS = 1e-8


def get_array_module(array) -> Any:
    """Return CuPy for CuPy arrays and NumPy otherwise."""
    try:
        import cupy as cp  # type: ignore  # noqa: PLC0415
    except ImportError:
        return np
    return cp.get_array_module(array)


def _tv_axes(tv_dims: TVDims) -> tuple[int, ...]:
    """Axes of a ``(nx, nz, ny)`` volume included in the TV."""
    if tv_dims == 3:  # noqa: PLR2004
        return (0, 1, 2)
    if tv_dims == 2:  # noqa: PLR2004
        return (1, 2)
    msg = f"tv_dims must be 2 or 3 (was {tv_dims})"
    raise ValueError(msg)


def gradient(volume, tv_dims: TVDims = 3):
    """
    Forward-difference gradient of a volume.

    The difference at the last index along each axis is zero (Neumann boundary
    conditions), so values do not wrap around the edges of the volume.

    Parameters
    ----------
    volume
        Array of shape ``(nx, nz, ny)``
    tv_dims
        3 to include all axes, or 2 to include only the in-slice axes

    Returns
    -------
    grad
        Array of shape ``(len(axes), nx, nz, ny)`` with one component per axis

    Group
    -----
    recon
    """
    xp = get_array_module(volume)
    axes = _tv_axes(tv_dims)
    grad = xp.zeros((len(axes), *volume.shape), dtype=volume.dtype)
    for i, axis in enumerate(axes):
        forward = [slice(None)] * 3
        forward[axis] = slice(0, -1)
        grad[(i, *forward)] = xp.diff(volume, axis=axis)
    return grad


def divergence(field, tv_dims: TVDims = 3):
    """
    Divergence of a vector field, the negative adjoint of :py:func:`gradient`.

    For any volume ``u`` and field ``p``,
    ``sum(gradient(u) * p) == -sum(u * divergence(p))``.

    Parameters
    ----------
    field
        Array of shape ``(len(axes), nx, nz, ny)``
    tv_dims
        3 or 2, as used to compute the field with :py:func:`gradient`

    Returns
    -------
    div
        Array of shape ``(nx, nz, ny)``

    Group
    -----
    recon
    """
    xp = get_array_module(field)
    axes = _tv_axes(tv_dims)
    div = xp.zeros(field.shape[1:], dtype=field.dtype)
    for i, axis in enumerate(axes):
        component = field[i]
        n = component.shape[axis]
        first, middle, last = ([slice(None)] * 3 for _ in range(3))
        first[axis], middle[axis], last[axis] = 0, slice(1, n - 1), n - 1
        before = [slice(None)] * 3
        before[axis] = slice(0, n - 2)
        div[tuple(first)] += component[tuple(first)]
        div[tuple(middle)] += component[tuple(middle)] - component[tuple(before)]
        last_before = [slice(None)] * 3
        last_before[axis] = n - 2
        div[tuple(last)] -= component[tuple(last_before)]
    return div


def total_variation(volume, tv_dims: TVDims = 3, eps: float = 0.0) -> float:
    """
    Isotropic total variation of a volume.

    ``TV(u) = sum(sqrt(|grad u|**2 + eps))``, using :py:func:`gradient`.

    Parameters
    ----------
    volume
        Array of shape ``(nx, nz, ny)``
    tv_dims
        3 to include all axes, or 2 to include only the in-slice axes
    eps
        Smoothing term (0 gives the exact TV)

    Returns
    -------
    tv : float

    Group
    -----
    recon
    """
    xp = get_array_module(volume)
    magnitude = xp.sqrt((gradient(volume, tv_dims) ** 2).sum(axis=0) + eps)
    return float(magnitude.sum())


def tv_gradient(volume, tv_dims: TVDims = 3, eps: float = DEFAULT_TV_EPS):
    """
    Gradient of the smoothed total variation with respect to the volume.

    This is ``-divergence(grad u / sqrt(|grad u|**2 + eps))``, the derivative of
    :py:func:`total_variation`.

    Parameters
    ----------
    volume
        Array of shape ``(nx, nz, ny)``
    tv_dims
        3 to include all axes, or 2 to include only the in-slice axes
    eps
        Smoothing term (must be positive)

    Returns
    -------
    dtv
        Array of shape ``(nx, nz, ny)``

    Group
    -----
    recon
    """
    xp = get_array_module(volume)
    grad = gradient(volume, tv_dims)
    magnitude = xp.sqrt((grad**2).sum(axis=0) + eps)
    return -divergence(grad / magnitude, tv_dims)


def tv_gradient_descent(
    volume,
    n_steps: int,
    step: float,
    tv_dims: TVDims = 3,
    eps: float = DEFAULT_TV_EPS,
    positivity: bool = True,
):
    """
    Reduce the total variation of a volume by normalized gradient descent.

    This is the TV step of ASD-POCS (Sidky and Pan, 2008): each of the
    ``n_steps`` steps moves the volume by a distance ``step`` (in the L2 norm)
    along the negative TV gradient. Negative values are set to zero at the end if
    ``positivity`` is True.

    Parameters
    ----------
    volume
        Array of shape ``(nx, nz, ny)``; it is not modified
    n_steps
        Number of gradient descent steps
    step
        Length (L2 norm) of each step
    tv_dims
        3 to include all axes, or 2 to include only the in-slice axes
    eps
        Smoothing of the TV near zero gradient
    positivity
        If True, set negative values to zero after the descent

    Returns
    -------
    result
        Array of shape ``(nx, nz, ny)``

    Group
    -----
    recon
    """
    xp = get_array_module(volume)
    result = volume.copy()
    for _ in range(n_steps):
        dtv = tv_gradient(result, tv_dims, eps)
        norm = float(xp.linalg.norm(dtv))
        if norm == 0:
            break
        result -= (step / norm) * dtv
    if positivity:
        xp.clip(result, 0, None, out=result)
    return result


def tv_fgp(
    volume,
    weight: float,
    n_iterations: int = 10,
    tv_dims: TVDims = 3,
    positivity: bool = True,
):
    """
    Total variation denoising with the fast gradient projection (FGP) method.

    Solves ``min_x 0.5 * |x - b|**2 + weight * TV(x)`` (with ``x >= 0`` if
    ``positivity``), where ``b`` is the input volume and TV is the isotropic
    total variation, using the accelerated dual projected gradient method of Beck
    and Teboulle (2009). This is the TV step of FISTA-TV.

    Parameters
    ----------
    volume
        Array of shape ``(nx, nz, ny)``; it is not modified
    weight
        TV regularization weight (``weight <= 0`` returns the input, clipped to
        non-negative values if ``positivity``)
    n_iterations
        Number of FGP iterations
    tv_dims
        3 to include all axes, or 2 to include only the in-slice axes
    positivity
        If True, constrain the result to be non-negative

    Returns
    -------
    result
        Denoised array of shape ``(nx, nz, ny)``

    Group
    -----
    recon
    """
    xp = get_array_module(volume)

    def project_constraint(x):
        return xp.clip(x, 0, None) if positivity else x

    if weight <= 0:
        return project_constraint(volume.copy())

    ndims = len(_tv_axes(tv_dims))
    step = 1.0 / (4.0 * ndims * weight)  # 1 / (weight * |gradient|**2)
    dual = xp.zeros((ndims, *volume.shape), dtype=volume.dtype)
    momentum_point = dual.copy()
    t = 1.0
    for _ in range(n_iterations):
        x = project_constraint(volume + weight * divergence(momentum_point, tv_dims))
        new_dual = momentum_point + step * gradient(x, tv_dims)
        # project onto the set of fields with |p| <= 1 at every voxel (isotropic)
        new_dual /= xp.maximum(1.0, xp.sqrt((new_dual**2).sum(axis=0)))
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        momentum_point = new_dual + ((t - 1.0) / t_new) * (new_dual - dual)
        dual, t = new_dual, t_new
    return project_constraint(volume + weight * divergence(dual, tv_dims))


def _sart_pass(
    sinogram: np.ndarray,
    initial: np.ndarray,
    relaxation: float,
    vol_geom: dict,
    proj_geom: dict,
    cuda: bool,
) -> tuple[np.ndarray, float]:
    """
    Run one SART pass (every projection once) on a slice, with positivity.

    Returns the updated slice and the squared norm of its residual
    ``|A x - b|**2``.
    """
    nangles = sinogram.shape[0]
    proj_id = create_projector(proj_geom, vol_geom, cuda=cuda)
    rec_id = astra.data2d.create("-vol", vol_geom, initial)
    sino_id = astra.data2d.create("-sino", proj_geom, sinogram)
    cfg = astra.astra_dict("SART_CUDA" if cuda else "SART")
    cfg["ProjectorId"] = proj_id
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectionDataId"] = sino_id
    cfg["option"] = {"MinConstraint": 0.0, "Relaxation": float(relaxation)}
    alg_id = astra.algorithm.create(cfg)
    try:
        astra.algorithm.run(alg_id, nangles)
        updated = astra.data2d.get(rec_id)
        fp_id, forward = astra.create_sino(updated, proj_id)
        astra.data2d.delete(fp_id)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data2d.delete([rec_id, sino_id])
        astra.projector.delete(proj_id)
    return updated, float(((forward - sinogram) ** 2).sum())


def _asd_pocs_volume(  # noqa: PLR0913
    sinograms: np.ndarray,
    thetas: np.ndarray,
    thickness: int,
    iterations: int,
    tv_dims: TVDims,
    beta: float,
    beta_reduction: float,
    tv_steps: int,
    alpha: float,
    alpha_reduction: float,
    r_max: float,
    eps: float,
    cuda: bool,
    ncores: int,
    show_progressbar: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Run ASD-POCS on a set of sinograms, sharing the step sizes between slices.

    ``sinograms`` has shape ``(nangles, ny, nslices)`` and is scaled to a maximum
    of about 1; the result has shape ``(nslices, thickness, ny)``.
    """
    _, ny, nslices = sinograms.shape
    vol_geom = astra.create_vol_geom((thickness, ny))
    proj_geom = astra.create_proj_geom("parallel", 1.0, ny, thetas)
    volume = np.zeros((nslices, thickness, ny), np.float32)
    data_norm = float(np.sqrt((sinograms.astype(np.float64) ** 2).sum()))
    history = {
        name: np.zeros(iterations)
        for name in ("data_distance", "tv", "tv_step", "relaxation")
    }
    xp = np
    if cuda:
        try:
            import cupy as cp  # type: ignore  # noqa: PLC0415

            xp = cp
        except ImportError:
            xp = np

    tv_step = 0.0
    for i in tqdm.tqdm(range(iterations), disable=not show_progressbar):
        # data consistency: one SART pass per slice, with positivity
        if cuda or nslices == 1:
            results = [
                _sart_pass(
                    sinograms[:, :, s],
                    volume[s],
                    beta,
                    vol_geom,
                    proj_geom,
                    cuda,
                )
                for s in range(nslices)
            ]
        else:
            tasks = [
                dask_delayed(_sart_pass)(
                    sinograms[:, :, s],
                    volume[s],
                    beta,
                    vol_geom,
                    proj_geom,
                    cuda,
                )
                for s in range(nslices)
            ]
            results = dask_compute(*tasks, num_workers=ncores, scheduler="threads")
        updated = np.stack([r[0] for r in results]).astype(np.float32)
        residual_norm = np.sqrt(sum(r[1] for r in results))
        data_distance = residual_norm / data_norm if data_norm > 0 else 0.0
        data_step = float(np.linalg.norm(updated - volume))
        if i == 0:
            tv_step = alpha * data_step

        # TV minimization
        tv_volume = tv_gradient_descent(
            xp.asarray(updated),
            tv_steps,
            tv_step,
            tv_dims,
        )
        new_volume = (
            xp.asnumpy(tv_volume) if xp is not np else tv_volume  # type: ignore[attr-defined]
        ).astype(np.float32)
        tv_change = float(np.linalg.norm(new_volume - updated))

        history["data_distance"][i] = data_distance
        history["tv"][i] = total_variation(new_volume, tv_dims)
        history["tv_step"][i] = tv_step
        history["relaxation"][i] = beta

        # adapt the step sizes (Sidky and Pan, 2008)
        if tv_change > r_max * data_step and data_distance > eps:
            tv_step *= alpha_reduction
        beta *= beta_reduction
        volume = new_volume
    return volume, history


def asd_pocs(  # noqa: PLR0913
    stack: np.ndarray,
    tilts: np.ndarray,
    iterations: int = 100,
    tv_dims: TVDims = 3,
    thickness: int | None = None,
    cuda: bool = False,
    ncores: int | None = None,
    sart_relaxation: float = 0.5,
    sart_relaxation_reduction: float = 0.985,
    tv_steps: int = 10,
    alpha: float = 0.2,
    alpha_reduction: float = 0.95,
    r_max: float = 0.95,
    eps: float = 0.02,
    show_progressbar: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Reconstruct a tilt series with ASD-POCS total variation minimization.

    Adaptive steepest descent projection onto convex sets (ASD-POCS; Sidky and Pan,
    2008) minimizes the total variation of the reconstruction subject to data
    consistency and positivity, which suppresses noise and missing-wedge
    artifacts in objects made of regions of nearly uniform density. Each
    iteration performs one SART pass through all of the projections (with
    relaxation ``sart_relaxation`` and positivity), followed by ``tv_steps`` steps
    of TV gradient descent; the TV step size is adapted so that the TV steps do
    not undo the data consistency steps while the relative data error
    ``|A x - b| / |b|`` is above ``eps``.

    The projections are scaled internally to a maximum of 1 (and the
    reconstruction scaled back), so ``eps`` is independent of the intensity
    scale. The projections should have their background removed (or be
    converted to absorbance for bright-field data) before reconstruction.

    ASD-POCS works best for data with few projections or a limited tilt range and
    low to moderate noise. With noisier data, the relative data error cannot fall
    much below the relative noise level, and as the SART relaxation and TV step
    sizes decrease, later iterations increasingly fit the noise (as SART alone
    does), so fewer iterations can give a better result; the returned
    ``history`` can help to judge this.

    Parameters
    ----------
    stack
        Tilt series of shape ``(nangles, ny, nx)`` (or ``(nangles, ny)`` for a
        single slice), with the tilt axis along ``x``
    tilts
        Tilt angles in degrees
    iterations
        Number of ASD-POCS iterations
    tv_dims
        3 (default) to minimize the TV of the 3D volume, coupling neighbouring
        slices, or 2 to reconstruct each slice independently (each with its own
        adaptive step sizes)
    thickness
        Size of the reconstruction perpendicular to the projections (``z``); if
        ``None``, ``ny`` is used
    cuda
        If True, use ASTRA's CUDA SART (and CuPy for the TV steps, if available)
    ncores
        Number of threads for the CPU data consistency steps (default: 90% of
        the CPUs)
    sart_relaxation
        Initial relaxation parameter of the SART steps
    sart_relaxation_reduction
        Factor by which the relaxation is reduced after each iteration
    tv_steps
        Number of TV gradient descent steps per iteration
    alpha
        Initial TV step size, as a fraction of the size of the first SART step
    alpha_reduction
        Factor by which the TV step size is reduced when it is too large
    r_max
        Maximum ratio of the change made by the TV steps to the change made by
        the SART step before the TV step size is reduced
    eps
        Relative data error ``|A x - b| / |b|`` below which the TV step size is no
        longer reduced (around the relative noise level of the data)
    show_progressbar
        If True, show a progress bar

    Returns
    -------
    rec : :py:class:`~numpy.ndarray`
        Reconstruction of shape ``(nx, thickness, ny)``
    history : dict
        Arrays with one value per iteration: ``"data_distance"`` (relative data
        error), ``"tv"`` (total variation, of the scaled reconstruction),
        ``"tv_step"`` and ``"relaxation"`` (the step sizes used; averaged over
        slices if ``tv_dims=2``)

    Group
    -----
    recon
    """
    if stack.ndim == 2:  # noqa: PLR2004
        stack = stack[:, :, np.newaxis]
    nangles, ny, nx = stack.shape
    tilts = np.asarray(tilts, dtype=float).ravel()
    if tilts.shape[0] != nangles:
        msg = f"Number of tilts ({tilts.shape[0]}) does not match the stack ({nangles})"
        raise ValueError(msg)
    _tv_axes(tv_dims)  # validate
    thickness = ny if thickness is None else int(thickness)
    if ncores is None:
        ncores = max(1, min(nx, int(0.9 * mp.cpu_count())))
    thetas = np.deg2rad(tilts)

    scale = float(np.abs(stack).max())
    scale = scale if scale > 0 else 1.0
    sinograms = (stack / scale).astype(np.float32)

    settings = {
        "thetas": thetas,
        "thickness": thickness,
        "iterations": iterations,
        "tv_dims": tv_dims,
        "beta": sart_relaxation,
        "beta_reduction": sart_relaxation_reduction,
        "tv_steps": tv_steps,
        "alpha": alpha,
        "alpha_reduction": alpha_reduction,
        "r_max": r_max,
        "eps": eps,
        "cuda": cuda,
    }
    if tv_dims == 3 or nx == 1:  # noqa: PLR2004
        rec, history = _asd_pocs_volume(
            sinograms,
            ncores=ncores,
            show_progressbar=show_progressbar,
            **settings,
        )
    else:
        # independent ASD-POCS reconstruction of each slice
        def one_slice(s):
            return _asd_pocs_volume(
                sinograms[:, :, s : s + 1],
                ncores=1,
                show_progressbar=False,
                **settings,
            )

        slices = range(nx)
        if cuda:
            results = [
                one_slice(s) for s in tqdm.tqdm(slices, disable=not show_progressbar)
            ]
        else:
            tasks = [dask_delayed(one_slice)(s) for s in slices]
            if show_progressbar:
                with ProgressBar():
                    results = dask_compute(*tasks, num_workers=ncores)
            else:
                results = dask_compute(*tasks, num_workers=ncores)
        rec = np.concatenate([r[0] for r in results])
        norms = np.sqrt((sinograms.astype(np.float64) ** 2).sum(axis=(0, 1)))
        residuals = np.array([r[1]["data_distance"] for r in results]) * norms[:, None]
        total_norm = np.sqrt((norms**2).sum())
        history = {
            "data_distance": np.sqrt((residuals**2).sum(0))
            / (total_norm if total_norm > 0 else 1.0),
            "tv": np.sum([r[1]["tv"] for r in results], axis=0),
            "tv_step": np.mean([r[1]["tv_step"] for r in results], axis=0),
            "relaxation": np.mean([r[1]["relaxation"] for r in results], axis=0),
        }
    logger.info(
        "ASD-POCS finished: relative data error %.4g, total variation %.4g",
        history["data_distance"][-1],
        history["tv"][-1],
    )
    return rec * scale, history
