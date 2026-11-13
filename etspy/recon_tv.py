"""
Total variation (TV) regularized reconstruction for ETSpy.

This module contains the TV operators used by the TV-regularized reconstruction
algorithms. Volumes are arrays of shape ``(nx, nz, ny)`` (the
:py:class:`~etspy.base.RecStack` order), where ``x`` is parallel to the tilt
axis. With ``tv_dims=3`` the TV couples neighbouring slices (differences along
all three axes); with ``tv_dims=2`` only the in-slice axes (``z`` and ``y``) are
used, so each slice is treated independently.

The operators work with NumPy arrays, or with CuPy arrays on the GPU.

The FISTA TV method is implemented from the published algorithm:

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

from etspy.recon import create_projector, resolve_cuda

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


def _prepare_tilt_series(
    stack: np.ndarray,
    tilts: np.ndarray,
    tv_dims: TVDims,
    thickness: int | None,
    ncores: int | None,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """
    Validate a tilt series and scale it to a maximum of 1.

    Returns the scaled stack (shape ``(nangles, ny, nx)``), the tilts, the scale
    factor, the reconstruction thickness, and the number of CPU threads.
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
    scale = float(np.abs(stack).max())
    scale = scale if scale > 0 else 1.0
    return (stack / scale).astype(np.float32), tilts, scale, thickness, ncores


def _tv_array_module(cuda: bool) -> Any:
    """Array module for the TV steps: CuPy if running on the GPU and available."""
    if cuda:
        try:
            import cupy as cp  # type: ignore  # noqa: PLC0415
        except ImportError:
            return np
        return cp
    return np


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


def _project_slices(
    arrays: np.ndarray,
    proj_id: int,
    *,
    backward: bool,
    cuda: bool,
    ncores: int,
) -> np.ndarray:
    """
    Forward or back project a stack of slices with an ASTRA projector.

    ``arrays`` has shape ``(nslices, thickness, ny)`` for forward projection or
    ``(nslices, nangles, ny)`` for back projection.
    """
    create = astra.create_backprojection if backward else astra.create_sino

    def project(array):
        data_id, result = create(array, proj_id)
        astra.data2d.delete(data_id)
        return result

    if cuda or arrays.shape[0] == 1:
        results = [project(a) for a in arrays]
    else:
        tasks = [dask_delayed(project)(a) for a in arrays]
        results = dask_compute(*tasks, num_workers=ncores, scheduler="threads")
    return np.stack(results).astype(np.float32)


def _operator_norm_squared(
    proj_id: int,
    shape: tuple[int, int],
    n_iterations: int = 30,
) -> float:
    """Estimate ``|A|**2`` (Lipschitz constant of the data term) by power iteration."""
    x = np.random.default_rng(0).random(shape).astype(np.float32)
    norm_squared = 0.0
    for _ in range(n_iterations):
        x /= np.linalg.norm(x)
        sino_id, ax = astra.create_sino(x, proj_id)
        bp_id, x = astra.create_backprojection(ax, proj_id)
        astra.data2d.delete([sino_id, bp_id])
        norm_squared = float(np.linalg.norm(x))
    return norm_squared


def fista_tv(
    stack: np.ndarray,
    tilts: np.ndarray,
    iterations: int = 100,
    tv_weight: float = 0.1,
    tv_dims: TVDims = 3,
    fgp_iterations: int = 10,
    thickness: int | None = None,
    cuda: bool = False,
    ncores: int | None = None,
    show_progressbar: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Reconstruct a tilt series with FISTA total variation regularization.

    Minimizes ``0.5 * |A x - b|**2 + tv_weight * TV(x)`` subject to ``x >= 0``,
    where ``A`` is the (ASTRA) projection operator, ``b`` the projections, and TV
    the isotropic total variation, using the fast iterative
    shrinkage-thresholding algorithm (FISTA) with an FGP total variation proximal
    step (Beck and Teboulle, 2009). Each iteration takes a gradient step on the
    data term (step ``1 / |A|**2``, estimated by power iteration), applies
    :py:func:`tv_fgp` with weight ``tv_weight / |A|**2``, and adds Nesterov
    momentum.

    ``tv_weight`` sets the balance between fitting the data and suppressing
    variation, which makes FISTA-TV well suited to noisy data: larger values give
    smoother reconstructions. The projections are scaled internally to a maximum
    of 1 (and the reconstruction scaled back), so ``tv_weight`` does not depend
    on the intensity scale of the data. Suitable values increase with the noise
    level (on test data, about 0.03-0.1 for low noise and 0.3 for noise of 5% of
    the maximum projection value); a well-chosen weight typically gives a final
    relative data error (``history["data_distance"]``) close to the relative noise
    level of the data. The projections should have their
    background removed (or be converted to absorbance for bright-field data)
    before reconstruction.

    Parameters
    ----------
    stack
        Tilt series of shape ``(nangles, ny, nx)`` (or ``(nangles, ny)`` for a
        single slice), with the tilt axis along ``x``
    tilts
        Tilt angles in degrees
    iterations
        Number of FISTA iterations
    tv_weight
        Weight of the total variation term
    tv_dims
        3 (default) to regularize the TV of the 3D volume, coupling neighbouring
        slices, or 2 to regularize each slice independently
    fgp_iterations
        Number of FGP iterations in each TV proximal step
    thickness
        Size of the reconstruction perpendicular to the projections (``z``); if
        ``None``, ``ny`` is used
    cuda
        If True, use ASTRA's GPU projector (and CuPy for the TV steps, if
        available)
    ncores
        Number of threads for the CPU projections (default: 90% of the CPUs)
    show_progressbar
        If True, show a progress bar

    Returns
    -------
    rec : :py:class:`~numpy.ndarray`
        Reconstruction of shape ``(nx, thickness, ny)``
    history : dict
        Arrays with one value per iteration: ``"objective"`` (of the scaled
        problem), ``"data_distance"`` (relative data error ``|A x - b| / |b|``),
        and ``"tv"`` (total variation of the scaled reconstruction)

    Group
    -----
    recon
    """
    cuda = resolve_cuda(cuda)
    scaled, tilts, scale, thickness, ncores = _prepare_tilt_series(
        stack,
        tilts,
        tv_dims,
        thickness,
        ncores,
    )
    _, ny, nx = scaled.shape
    # (nslices, nangles, ny), the layout of ASTRA sinograms
    sinograms = np.ascontiguousarray(np.transpose(scaled, (2, 0, 1)))
    data_norm = float(np.linalg.norm(sinograms))
    xp = _tv_array_module(cuda)

    vol_geom = astra.create_vol_geom((thickness, ny))
    proj_geom = astra.create_proj_geom("parallel", 1.0, ny, np.deg2rad(tilts))
    proj_id = create_projector(proj_geom, vol_geom, cuda=cuda)
    history = {
        name: np.zeros(iterations) for name in ("objective", "data_distance", "tv")
    }
    try:
        lipschitz = _operator_norm_squared(proj_id, (thickness, ny))

        def forward(volume):
            return _project_slices(
                volume,
                proj_id,
                backward=False,
                cuda=cuda,
                ncores=ncores,
            )

        def back(sinos):
            return _project_slices(
                sinos,
                proj_id,
                backward=True,
                cuda=cuda,
                ncores=ncores,
            )

        x = np.zeros((nx, thickness, ny), np.float32)
        y = x.copy()
        t = 1.0
        for i in tqdm.tqdm(range(iterations), disable=not show_progressbar):
            # gradient step on the data term, then the TV proximal step
            gradient_step = y - back(forward(y) - sinograms) / lipschitz
            x_new = tv_fgp(
                xp.asarray(gradient_step),
                tv_weight / lipschitz,
                fgp_iterations,
                tv_dims,
            )
            if xp is not np:
                x_new = xp.asnumpy(x_new)  # type: ignore[attr-defined]
            x_new = x_new.astype(np.float32)
            # Nesterov momentum
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            x, t = x_new, t_new

            data_error = float(np.linalg.norm(forward(x) - sinograms))
            tv_value = total_variation(x, tv_dims)
            history["objective"][i] = 0.5 * data_error**2 + tv_weight * tv_value
            history["data_distance"][i] = data_error / data_norm if data_norm > 0 else 0
            history["tv"][i] = tv_value
    finally:
        astra.projector.delete(proj_id)
    logger.info(
        "FISTA-TV finished: relative data error %.4g, total variation %.4g",
        history["data_distance"][-1],
        history["tv"][-1],
    )
    return x * scale, history
