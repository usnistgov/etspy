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

from typing import Any, Literal

import numpy as np

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
