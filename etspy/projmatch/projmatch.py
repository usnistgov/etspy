"""Projection matching alignment."""

import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import astra
import numpy as np
import tqdm
from hyperspy.signals import Signal1D, Signal2D
from scipy.fft import fft, fftfreq, ifft
from scipy.ndimage import fourier_shift, gaussian_filter
from scipy.signal import convolve

from etspy._background import background_mode
from etspy.align import apply_shifts as _apply_shifts
from etspy.align import fourier_shift as _fourier_shift
from etspy.recon import astra_iterations, create_projector

if TYPE_CHECKING:
    from etspy.base import TomoStack

logger = logging.getLogger(__name__)

DIM_2D = 2

# largest shift (pixels at the current binning) applied in one iteration
MAX_STEP = 0.5
# downsampling levels that would leave fewer pixels than this (perpendicular to the
# tilt axis) are skipped, since the alignment cannot be refined reliably on them
MIN_BINNED_SIZE = 16


class _SliceReconstructor:
    """ASTRA objects that reconstruct and reproject one slice at a time.

    Each thread of a :py:class:`ProjMatch` reconstruction uses its own instance, so
    that the ASTRA objects are never shared between threads.
    """

    def __init__(
        self,
        config: dict,
        thetas: np.ndarray,
        ny: int,
        iterations: int,
        cuda: bool,
    ):
        proj_geom = astra.create_proj_geom("parallel", 1.0, ny, thetas)
        vol_geom = astra.create_vol_geom((ny, ny))
        self.iterations = iterations
        self.proj_id = create_projector(proj_geom, vol_geom, cuda=cuda)
        self.rec_id = astra.data2d.create("-vol", vol_geom)
        self.sino_id = astra.data2d.create(
            "-sino",
            proj_geom,
            np.zeros([len(thetas), ny]),
        )
        config = {**config, "option": dict(config["option"])}
        config["ReconstructionDataId"] = self.rec_id
        config["ProjectorId"] = self.proj_id
        config["ProjectionDataId"] = self.sino_id
        self.alg_id = astra.algorithm.create(config)

    def __call__(
        self,
        sino: np.ndarray,
        initial: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct a sinogram (starting from ``initial``) and reproject it."""
        if initial is not None:
            astra.data2d.store(self.rec_id, initial)
        astra.data2d.store(self.sino_id, sino)
        astra.algorithm.run(self.alg_id, self.iterations)
        rec = astra.data2d.get(self.rec_id)
        sino_id, reproj = astra.create_sino(rec, self.proj_id)
        astra.data2d.delete(sino_id)
        return rec, reproj

    def delete(self):
        """Delete the ASTRA objects."""
        astra.algorithm.delete(self.alg_id)
        astra.data2d.delete([self.rec_id, self.sino_id])
        astra.projector.delete(self.proj_id)


def _shift(data: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Shift a background-subtracted stack by ``(Y, X)`` per image, filling zeros."""
    out = _fourier_shift(data, shifts[:, 0:1], axis=1)
    if np.any(shifts[:, 1]):
        out = _fourier_shift(out, shifts[:, 1:2], axis=2)
    return out


def _select_slices(nslice: int | None, params: dict, nx: int) -> tuple[int, int]:
    """Return the range ``(start, stop)`` of the slices used by ProjMatch."""
    if nslice is not None and "slices" in params:
        msg = 'Give either nslice or the "slices" parameter, not both'
        raise ValueError(msg)
    if nslice is not None:
        if not 0 <= nslice < nx:
            msg = f"nslice must be in the range [0, {nx - 1}]"
            raise ValueError(msg)
        return nslice, nslice + 1
    slices = params.get("slices")
    if slices is None:
        return 0, nx
    if isinstance(slices, slice):
        slices = slices.indices(nx)[:2]
    start, stop = int(slices[0]), int(slices[1])
    if not 0 <= start < stop <= nx:
        msg = f"Invalid slices {(start, stop)} for a stack with {nx} slices"
        raise ValueError(msg)
    return start, stop


class ProjMatch:
    """Class to perform alignment via projection matching method.

    Projection images are aligned by minimizing the difference between the input
    projections and the forward projections of the current reconstruction.  Python
    implementation of the method reported in:

    M. Odstrčil, Mirko Holler, Jörg Raabe, and Manuel Guizar-Sicairos. Alignment methods
    for nanotomography with deep subpixel accuracy, Optics Express Vol. 27 (2019)
    pp. 36637-36652.
    https://doi.org/10.1364/OE.27.036637

    A previous Python implementation based on the same Matlab source code and with
    additional functionality is also available:
    https://github.com/jtschwar/projection_refinement/

    By default, all of the slices of the stack are reconstructed, and the shifts
    perpendicular to the tilt axis (Y) and along it (X) are both refined. The
    shifts found are in ``total_shifts``, and are applied to the stack with
    :py:meth:`apply_shifts`.

    The direction of the tilt axis is not refined: correct it beforehand with
    :py:meth:`~etspy.base.TomoStack.tilt_align`. (With single-axis tilt series of
    limited range, the reconstruction absorbs almost all of a small rotation of the
    tilt axis, so that projection matching cannot measure it reliably.)

    Group
    -----
    align
    """

    def __init__(
        self,
        stack: "TomoStack",
        nslice: int | None = None,
        cuda: bool = False,
        params: dict | None = None,
    ):
        """Create a ProjMatch instance.

        Parameters
        ----------
        stack : TomoStack
            Stack to be aligned.
        nslice : int, optional
            If given, only this slice (along the tilt axis) is used, and only the
            Y-shifts are refined. By default, the slices given by the "slices"
            parameter (all of them unless given) are used.
        cuda : bool
            If True, use CUDA-acceleration. Default is False.
        params : Optional[Dict], optional
            Dictionary of parameters for alignment. Acceptable keys are:

            - "levels": downsampling factors, from coarse to fine (default
              ``[8, 4, 2, 1]``)
            - "iterations": maximum number of iterations per level (default 100)
            - "recon_algorithm": "FBP" (default), "SIRT" or "SART"
            - "recon_iterations": passes through all of the projections for each
              SIRT or SART reconstruction (default: 20 for SIRT, 5 for SART)
            - "relax": relaxation of the shift updates (default 0.5)
            - "minstep": the refinement at each level stops when the updates
              (before relaxation) are smaller than this (pixels at full
              resolution; default 0.01)
            - "slices": the slices (along the x-axis) to use, as a
              :py:class:`slice` or ``(start, stop)``. Default: all of them. Each
              iteration reconstructs every slice used, so fewer slices are faster
              but use less of the data
            - "align_x": whether to refine the X-shifts (along the tilt axis).
              Default: True if more than one slice is used
            - "ncores": number of threads reconstructing the slices on the CPU
              (default: 90% of the CPU cores)
            - "history": if True, the sinogram and reconstruction of the central
              slice at every iteration are kept (in ``sino_update`` and
              ``rec_update``; default False, since they can use a lot of memory)

        """
        self.stack = stack
        data = np.asarray(stack.data, dtype=np.float64)
        if data.ndim == DIM_2D:
            data = data[:, :, np.newaxis]
        self.tilts = stack.tilts.data.squeeze()
        self.thetas = np.deg2rad(self.tilts)
        self.nangles, self.ny, nx = data.shape
        self.cuda = cuda
        self.params = {} if params is None else params

        self.slices = slices = _select_slices(nslice, self.params, nx)
        self.nslices = slices[1] - slices[0]
        # with a single slice, only that slice is needed; otherwise the whole
        # width is kept, since X-shifts bring in content from outside
        # the slices used
        multi = self.nslices > 1
        if multi:
            self._data = data
        else:
            self._data = data[:, :, slices[0] : slices[1]]
        # sinogram of the (central) slice, as recorded in the history
        self.sino = data[:, :, (slices[0] + slices[1] - 1) // 2]
        # background (vacuum) level of each projection, used to fill the pixels
        # uncovered when the projections are shifted
        self.background = np.array(
            [background_mode(np.atleast_2d(im)) for im in stack.data],
        )

        self.align_x = bool(self.params.get("align_x", multi))
        if not multi and self.align_x:
            msg = (
                "Refining the X-shifts needs more than one slice "
                f"(slices {slices} were selected)"
            )
            raise ValueError(msg)

        self.levels = self.params.get("levels", [8, 4, 2, 1])
        self.iterations = self.params.get("iterations", 100)
        self.history = bool(self.params.get("history", False))
        ncores = self.params.get("ncores")
        if ncores is None:
            ncores = int(0.9 * mp.cpu_count())
        self.ncores = 1 if cuda else max(1, min(int(ncores), self.nslices))

        # the alignment: shifts (Y, X) of each projection
        self.total_shifts = np.zeros([self.nangles, 2])

        # per-level history, filled in by calculate_shifts
        self.error: list[Any] = []
        self.projection_error: list[Any] = []
        self.sino_update: list[Any] | None = None
        self.rec_update: list[Any] | None = None
        self.shift_update: Any = None

        self.recon_algorithm = self.params.get("recon_algorithm", "FBP")
        # recon_iterations are full passes through the projections (also for
        # SART), converted here to the number of ASTRA iterations to run
        default_passes = {"sirt": 20, "sart": 5}.get(self.recon_algorithm.lower())
        _, self.recon_iterations = astra_iterations(
            self.recon_algorithm,
            self.params.get("recon_iterations", default_passes),
            self.nangles,
        )
        # reconstructors (one per thread), created per level by update_geometries
        self._reconstructors: list[_SliceReconstructor] = []
        # reconstruction of each slice, the starting point of the next SIRT or
        # SART reconstruction (None for FBP)
        self._recs: np.ndarray | None = None
        self.create_recon_algorithm(self.recon_algorithm, self.cuda)
        self.relax = self.params.get("relax", 0.5)
        self.minstep = self.params.get("minstep", 0.01)

    def _shifts_2d(self) -> np.ndarray:
        """Return ``total_shifts`` as ``(Y, X)``; a 1D array gives Y-shifts only."""
        shifts = np.asarray(self.total_shifts, dtype=float)
        if shifts.ndim == 1:
            shifts = np.stack([shifts, np.zeros_like(shifts)], axis=1)
        if shifts.shape != (self.nangles, 2):
            msg = (
                f"Number of shifts ({len(shifts)}) is not consistent "
                f"with number of images in the stack ({self.nangles})"
            )
            raise ValueError(msg)
        return shifts

    def calculate_shifts(
        self,
        show_progressbar=False,
    ):
        """Calculate the shifts by projection matching.

        The alignment is refined at each downsampling level in ``levels``,
        starting from the current ``total_shifts`` (so calling this method again
        continues the refinement). The history of the
        most recent call is stored in:

        - ``error``: for each level, a :py:class:`~hyperspy.signals.Signal1D` of
          the normalized residual at each iteration
        - ``projection_error``: for each level, a
          :py:class:`~hyperspy.signals.Signal1D` of the normalized RMS residual of
          each projection (signal axis) at each iteration (navigation axis); see
          also :py:meth:`final_projection_error`
        - ``shift_update``: a :py:class:`~hyperspy.signals.Signal1D` of the total
          shifts (Y, X) after each level
        - ``sino_update`` and ``rec_update`` (only if the "history" parameter is
          True; otherwise ``None``): for each level, a
          :py:class:`~hyperspy.signals.Signal2D` of the shifted sinogram and of
          the reconstruction of the central slice at each iteration

        Parameters
        ----------
        show_progressbar : bool, optional
            If True, show a progress bar for the iterations at each level.
            Default is False.
        """
        self.total_shifts = self._shifts_2d()
        errors: list[list[float]] = []
        projection_errors: list[list[np.ndarray]] = []
        sinos: list[list[np.ndarray]] = []
        recs: list[list[np.ndarray]] = []
        total_shifts_history = []
        levels = [j for j in self.levels if j == 1 or self.ny / j >= MIN_BINNED_SIZE]
        if len(levels) < len(self.levels):
            logger.warning(
                "Skipping downsampling levels %s, which would leave fewer than %i "
                "pixels",
                sorted(set(self.levels) - set(levels), reverse=True),
                MIN_BINNED_SIZE,
            )
        executor = ThreadPoolExecutor(max_workers=self.ncores)
        try:
            for j in levels:
                level = self._refine_level(j, show_progressbar, executor)
                errors.append(level[0])
                projection_errors.append(level[1])
                sinos.append(level[2])
                recs.append(level[3])
                total_shifts_history.append(self.total_shifts.copy())
        finally:
            executor.shutdown()
            self.delete_astra_objects()

        self.error = [Signal1D(np.array(e)) for e in errors]
        self.projection_error = []
        for e in projection_errors:
            signal = Signal1D(np.array(e))
            signal.axes_manager[0].name = "Iteration"
            signal.axes_manager[1].name = "Projection"
            self.projection_error.append(signal)
        self.shift_update = Signal1D(np.array(total_shifts_history))
        if self.history:
            self.sino_update = [Signal2D(np.stack(s)) for s in sinos]
            self.rec_update = [Signal2D(np.stack(r)) for r in recs]
        else:
            self.sino_update = None
            self.rec_update = None

    def final_projection_error(self) -> np.ndarray:
        """
        Return the residual of each projection at the end of the refinement.

        This is the normalized RMS difference between each (high-pass filtered)
        projection and the reprojection of the reconstruction, at the last
        iteration of the last level of :py:meth:`calculate_shifts`. Projections
        with a much larger error than the others are not consistent with the rest
        of the tilt series (for example because of damage, contamination or a
        large misalignment).

        Returns
        -------
        error : :py:class:`~numpy.ndarray`
            Error of each projection, in the order of the tilts
        """
        if not self.projection_error:
            msg = "Run calculate_shifts first"
            raise ValueError(msg)
        return self.projection_error[-1].data[-1].copy()

    def _downsample(self, data: np.ndarray, j: int) -> np.ndarray:
        """Blur and downsample a stack by ``j`` along y (and x for several slices)."""
        if j == 1:
            return data.copy()
        axes = (1, 2) if self.nslices > 1 else (1,)
        for axis in axes:
            data = blur_convolve(data, j, axis=axis)
        for axis in axes:
            data = interpolate_ft(data, j, axis=axis)
        return data

    def _refine_level(  # noqa: PLR0915
        self,
        j: int,
        show_progressbar: bool,
        executor: ThreadPoolExecutor,
    ) -> tuple[list[float], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Refine the alignment at downsampling level ``j``.

        Returns the normalized residual and the residual of each projection at
        each iteration and, if the history is kept, the shifted sinogram and
        reconstruction of the central slice at each iteration.
        """
        logger.info("Binning %i", j)
        background = self.background[:, np.newaxis, np.newaxis]
        # apply the current alignment at full resolution, then downsample
        corrected = _shift(self._data - background, self.total_shifts)
        binned = self._downsample(corrected + background, j)
        # downsampling scales the intensities slightly; scale the background too
        scale = float(self._downsample(np.ones((1, *self._data.shape[1:])), j).mean())
        background = background * scale
        binned = binned - background

        _, ny, nx = binned.shape
        if self.nslices > 1:
            nx_full = self._data.shape[2]
            x0 = int(np.floor(self.slices[0] * nx / nx_full))
            x1 = max(x0 + 1, int(np.ceil(self.slices[1] * nx / nx_full)))
            x1 = min(x1, nx)
        else:
            x0, x1 = 0, nx
        nslices = x1 - x0
        mid = nslices // 2
        self.update_geometries(ny)
        iterative = self.recon_algorithm.lower() != "fbp"
        self._recs = np.zeros((nslices, ny, ny), np.float32) if iterative else None

        # alignment found at this level, in pixels at this binning
        shifts = np.zeros((self.nangles, 2))
        mass = 0
        errors: list[float] = []
        projection_errors: list[np.ndarray] = []
        sinos: list[np.ndarray] = []
        recs: list[np.ndarray] = []

        for i in tqdm.tqdm(range(self.iterations), disable=not (show_progressbar)):
            # the background is subtracted before reconstruction, since a constant
            # offset is not consistent with an object inside the field of view
            current = _shift(binned, shifts)[:, :, x0:x1]
            rec, reproj = self.reconstruct(current, executor)

            resid = high_pass_filter(reproj - current, 5)
            grad_y = sino_gradient(reproj)
            grad_y_hp = high_pass_filter(grad_y, 5)
            update = np.zeros((self.nangles, 2))
            update[:, 0] = -np.sum(grad_y_hp * resid, axis=(1, 2)) / np.sum(
                grad_y_hp**2,
                axis=(1, 2),
            )
            if self.align_x:
                grad_x = _gradient(reproj, axis=2)
                resid_x = high_pass_filter(reproj - current, 5, axis=2)
                grad_x_hp = high_pass_filter(grad_x, 5, axis=2)
                update[:, 1] = -np.sum(grad_x_hp * resid_x, axis=(1, 2)) / np.sum(
                    grad_x_hp**2,
                    axis=(1, 2),
                )
            update = np.minimum(MAX_STEP, np.abs(update)) * np.sign(update) * self.relax
            update = update - np.median(update, axis=0)
            max_step = np.minimum(np.quantile(np.abs(update), 0.99, axis=0), MAX_STEP)
            update = np.minimum(max_step, np.abs(update)) * np.sign(update)
            shifts += update

            if i == 0:
                mass = np.median(np.mean(np.abs(current), axis=(1, 2)))
            if self.history:
                sinos.append(current[:, :, mid] + background[:, :, 0])
                recs.append(rec[mid])
            errors.append(float(np.linalg.norm(resid) / mass))
            projection_errors.append(
                np.sqrt(np.mean(resid**2, axis=(1, 2))) / mass,
            )
            # converged when the (unrelaxed) steps are smaller than minstep
            max_update = np.max(np.quantile(np.abs(update), 0.995, axis=0)) / self.relax
            if max_update * j < self.minstep:
                logger.info("Converged after %i iterations", i)
                break

        shifts = shifts - np.median(shifts, axis=0)
        self.total_shifts = self.total_shifts + shifts * j
        return errors, projection_errors, sinos, recs

    def create_recon_algorithm(
        self,
        method: str,
        cuda: bool,
    ):
        """Create a configuration dictionary for the reconstruction.

        Parameters
        ----------
        method
            Reconstruction method
        cuda
            If True, use CUDA acceleration for reconstruction

        """
        self.recon_config = {"option": {}, "type": ""}
        if cuda:
            self.recon_config["type"] = method.upper() + "_CUDA"
        else:
            self.recon_config["type"] = method.upper()

        if method.lower() in [
            "sirt",
            "sart",
        ]:
            self.recon_config["option"]["MinConstraint"] = 0.0

    def update_geometries(
        self,
        current_ny: int,
    ):
        """Create the reconstructors for the current downsampling.

        Parameters
        ----------
        current_ny
            Number of pixels in the Y-dimension in the current downsampled sinogram.

        """
        self.delete_astra_objects()
        for _ in range(self.ncores):
            self._reconstructors.append(
                _SliceReconstructor(
                    self.recon_config,
                    self.thetas,
                    current_ny,
                    self.recon_iterations,
                    self.cuda,
                ),
            )

    def delete_astra_objects(self):
        """Delete the ASTRA objects created for the current downsampling level."""
        while self._reconstructors:
            self._reconstructors.pop().delete()

    def reconstruct(
        self,
        sinograms: np.ndarray,
        executor: ThreadPoolExecutor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct each slice and reproject it.

        Parameters
        ----------
        sinograms
            Projections of the slices to reconstruct, of shape
            ``(nangles, ny, nslices)``
        executor
            Thread pool used to reconstruct the slices in parallel

        Returns
        -------
        recs, reprojections
            The reconstruction of each slice, of shape ``(nslices, ny, ny)``, and
            their reprojections, of the same shape as ``sinograms``
        """
        nslices = sinograms.shape[2]
        reprojections = np.empty(sinograms.shape)
        recs = np.empty((nslices, sinograms.shape[1], sinograms.shape[1]))

        def run(worker: int):
            reconstructor = self._reconstructors[worker]
            for k in range(worker, nslices, len(self._reconstructors)):
                initial = None if self._recs is None else self._recs[k]
                recs[k], reprojections[:, :, k] = reconstructor(
                    sinograms[:, :, k],
                    initial,
                )
                if self._recs is not None:
                    self._recs[k] = recs[k]

        nworkers = min(len(self._reconstructors), nslices)
        if nworkers == 1:
            run(0)
        else:
            for future in [executor.submit(run, w) for w in range(nworkers)]:
                future.result()
        return recs, reprojections

    def apply_shifts(
        self,
    ) -> "TomoStack":
        """
        Apply the calculated shifts to the input stack.

        Pixels uncovered by the shifts are filled with the background level of each
        projection (see :py:func:`etspy.align.apply_shifts`). The shifts are added
        to ``metadata.Tomography.shifts``.

        Returns
        -------
        shifted : TomoStack
            Copy of input stack after the shifts are applied
        """
        return _apply_shifts(self.stack, self._shifts_2d(), "fourier")


def shift_sinogram(
    sino: np.ndarray,
    shifts: np.ndarray,
    background: float | np.ndarray | None = None,
) -> np.ndarray:
    """Apply shifts to a sinogram using Fourier shifts.

    Parameters
    ----------
    sino:
        Sinogram to shift, of shape ``(nangles, ny)``
    shifts:
        Shift of each projection (along ``ny``)
    background:
        Background level used to fill the pixels uncovered by the shifts, either a
        single value or one value per projection. If ``None``, it is estimated
        from the sinogram (see :py:func:`etspy._background.background_mode`).

    Returns
    -------
    shifted : :py:class:`~numpy.ndarray`
        Shifted sinogram (float64)
    """
    sino = np.asarray(sino, dtype=np.float64)
    if background is None:
        background = background_mode(sino)
    fill = np.broadcast_to(
        np.asarray(background, dtype=np.float64).reshape(-1, 1),
        (sino.shape[0], 1),
    )
    return _fourier_shift(sino - fill, np.asarray(shifts), axis=1) + fill


def _along_axis(func, data: np.ndarray, axis: int, *args) -> np.ndarray:
    """Apply a function of ``(nlines, n)`` arrays along one axis of ``data``."""
    moved = np.moveaxis(data, axis, -1)
    result = func(moved.reshape(-1, moved.shape[-1]), *args)
    result = result.reshape(*moved.shape[:-1], result.shape[-1])
    return np.moveaxis(result, -1, axis)


def blur_convolve(
    sino: np.ndarray,
    factor: int,
    axis: int = 1,
) -> np.ndarray:
    """Convolve a sinogram or image stack to blur by a given factor.

    Parameters
    ----------
    sino : np.ndarray
        Sinogram or stack to blur
    factor : int
        Factor by which to blur
    axis : int
        Axis along which to blur. Default is 1 (y).

    Returns
    -------
    np.ndarray
        Blurred version of input data

    """
    grid = np.arange(-np.ceil(2 * factor), np.ceil(2 * factor) + 1) / factor
    kernel = np.exp(-(grid**2))
    kernel /= np.sum(kernel)
    shape = kernel.shape[0]

    def blur(lines: np.ndarray) -> np.ndarray:
        blurred = convolve(lines, kernel.reshape(1, shape), mode="same")
        corr = convolve(np.ones([1, lines.shape[1]]), kernel.reshape(1, shape), "same")
        return blurred / corr

    return _along_axis(blur, sino, axis)


def blur_edges(
    sino: np.ndarray,
    window: int = 5,
    sigma: float = 5,
    axis: int = 1,
) -> np.ndarray:
    """Blur the edges on both sides of a sinogram within a user defind window.

    Parameters
    ----------
    sino : np.ndarray
        Sinogram or stack to blur
    window : int
        Number of pixels to blur on both sides of the sinogram.
    sigma : float
        Standard deviation for a Gaussian kernel which determines the amount of
        blurring. Default is 5.
    axis : int
        Axis along which to blur the edges. Default is 1 (y).

    Returns
    -------
    sino_filtered : np.ndarray
        Filtered version of input data

    """
    window = max(window, 3)
    filtered = gaussian_filter(sino, sigma, axes=(axis,))
    inner = [slice(None)] * sino.ndim
    inner[axis] = slice(window, -window)
    filtered[tuple(inner)] = sino[tuple(inner)]
    return filtered


def high_pass_filter(
    sino: np.ndarray,
    sigma: float = 5,
    axis: int = 1,
) -> np.ndarray:
    """High pass filter a sinogram or sinogram-like array using Gaussian convolution.

    Parameters
    ----------
    sino : np.ndarray
        Sinogram or stack to filter
    sigma : int
        Standard deviation of the Gaussian filter which controls the degree of
        filtering. Default is 5.
    axis : int
        Axis along which to filter. Default is 1 (y).

    Returns
    -------
    sino_filtered : np.ndarray
        Filtered version of input data

    """
    filtered = sino - gaussian_filter(sino, sigma, axes=(axis,))
    return filtered


def high_pass_fourier_filter(
    sino: np.ndarray,
    sigma: float = 0.01,
    apply_fft: bool = True,
) -> np.ndarray:
    """Fourier high pass filter a sinogram or sinogram-like array.

    Parameters
    ----------
    sino : np.ndarray
        Sinogram or stack to filter
    sigma : int
        Standard deviation of the Gaussian filter which controls the degree of
        filtering. Default is 0.01.
    apply_fft : bool
        If True, input is real and an FFT is applied prior to applying the filter.
        An IFFT is then applied to the filtered array.  Default is True.

    Returns
    -------
    sino_filtered : np.ndarray
        Filtered version of input data

    """
    is_real = np.all(np.isreal(sino))
    _, ny = sino.shape

    if apply_fft:
        sino = cast("np.ndarray", fft(sino))

    freq = fftfreq(ny)
    sigma = 256 / (ny) * sigma

    if sigma == 0:
        high_pass_filter = 2j * np.pi * freq
    else:
        high_pass_filter = 1 - np.exp(-0.5 * (freq / sigma) ** 2)

    sino_filtered = sino * high_pass_filter

    if apply_fft:
        sino_filtered = np.fft.ifft(sino_filtered)
    if is_real:
        sino_filtered = np.real(sino_filtered)
    return sino_filtered


def _interpolate_ft_lines(
    sino: np.ndarray,
    blur_factor: int,
) -> np.ndarray:
    """Downsample each line of a ``(nlines, n)`` array (see interpolate_ft)."""
    _, ny = sino.shape
    ny_new = int(np.ceil(ny / blur_factor / 2) * 2)
    ny_new = 2 + ny_new

    scale = ny_new / ny
    pad_pix = int(np.ceil(np.sqrt(1 / scale)))

    # Pad sinogram in y-dimension
    sino = np.pad(
        sino,
        (
            (0, 0),
            (pad_pix, pad_pix + 1),
        ),
        "symmetric",
    )
    sino_fft = fft(sino, axis=1)

    # Apply +0.5 shift in Fourier space
    sino_fft = fourier_shift(sino_fft, 0.5)

    # Apply FFT shift
    _, ny_fft = sino_fft.shape
    crop_pix = int(np.ceil(ny_fft / 2))
    idx = np.concatenate((np.arange(crop_pix, ny_fft), np.arange(crop_pix)))
    sino_fft = sino_fft[:, idx]

    # Crop in Fourier space while preserving center position
    center = np.floor(ny_new / 2) - np.floor(ny_fft / 2)
    y_crop = np.arange(
        np.maximum(-center, 0),
        np.minimum(-center + ny_new, ny_fft),
        dtype=int,
    )
    sino_fft = sino_fft[:, y_crop]

    # Reverse FFT shift
    offset = int(np.floor(ny_new / 2))
    idx = np.concatenate((np.arange(offset, ny_new), np.arange(offset)))
    sino_fft = sino_fft[:, idx]

    # Apply -0.5 shift in cropped Fourier space
    sino_fft = fourier_shift(sino_fft, -0.5)

    # Return to the Real Space
    sino_centered = ifft(sino_fft, axis=1)

    # Scale intensities to maintain average value between input and output
    sino_centered = sino_centered * scale
    # Remove the Padding
    sino_centered = sino_centered[:, 1:-1]

    sino_centered = np.real(sino_centered)
    return sino_centered


def interpolate_ft(
    sino: np.ndarray,
    blur_factor: int,
    axis: int = 1,
) -> np.ndarray:
    """Interpolate the FT of sinogram after convolution to maintain cente of mass.

    Parameters
    ----------
    sino : np.ndarray
        Input sinogram
    blur_factor : int
        Factor used for blurring
    axis : int
        Axis along which to downsample. Default is 1 (y).

    Returns
    -------
    sino_centered : np.ndarray
        Interpolated version of blurred sinogram

    """
    return _along_axis(_interpolate_ft_lines, sino, axis, blur_factor)


def sino_gradient(
    sino: np.ndarray,
    blur_window: int = 5,
    blur_sigma: float = 5.0,
) -> np.ndarray:
    """Calculate the gradient of a sinogram along Y axis.

    Parameters
    ----------
    sino : np.ndarray
        Input sinogram, or stack of shape ``(nangles, ny, nslices)``
    smooth_window : int
        Window for blurring edges. Default is 5.
    blur_sigma : float
        Sigma for Gaussian blur. Default is 5.0.

    Returns
    -------
    sino_grad : np.ndarray
        Gradient of input sinogram along the Y-axis

    """
    ny = sino.shape[1]
    sino = blur_edges(sino, blur_window, blur_sigma)
    x = 2j * np.pi * fftfreq(ny)
    x = x.reshape((1, ny) + (1,) * (sino.ndim - 2))

    sino_fft = cast("np.ndarray", fft(sino, axis=1))
    sino_fft = sino_fft * x

    sino_grad = ifft(sino_fft, axis=1)
    sino_grad = np.real(cast("np.ndarray", sino_grad))
    return sino_grad


def _gradient(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Calculate the derivative along an axis by central differences.

    Unlike a Fourier derivative, this is not affected by the discontinuity
    between the two edges of the (non-periodic) data.
    """
    return np.gradient(data, axis=axis)
