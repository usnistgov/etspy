"""Alignment module for ETSpy package."""

# pyright: reportPossiblyUnboundVariable=false

import logging
from abc import ABC, abstractmethod
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal, Union, cast

import matplotlib.pylab as plt
import numpy as np
import tqdm
from hyperspy.signal import BaseSignal
from pystackreg import StackReg
from scipy import ndimage
from scipy.fft import next_fast_len
from skimage.feature import canny
from skimage.filters import sobel
from skimage.registration import phase_cross_correlation as pcc
from skimage.transform import hough_line, hough_line_peaks

from etspy._background import background_mode, projected_mass

if TYPE_CHECKING:
    from hyperspy.misc.utils import DictionaryTreeBrowser as Dtb

    from etspy.base import TomoShifts, TomoStack  # pragma: no cover

has_cupy = True
try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import shift as shift_gpu

    has_gpu = cp.cuda.runtime.getDeviceCount() > 0

except Exception:
    has_cupy = False

logger = logging.getLogger(__name__)

CL_RES_THRESHOLD = 0.5  # threshold for common line registration method


def get_best_slices(stack: "TomoStack", nslices: int) -> np.ndarray:
    """
    Get best nslices for center of mass analysis.

    Slices which have the highest ratio of total mass to mass variance
    and their location are returned. Mass is measured relative to the
    background of each projection (see :py:func:`etspy._background.projected_mass`).


    Parameters
    ----------
    stack
        Tilt series from which to select the best slices
    nslices
        Number of slices to return

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Location along the x-axis of the best slices

    Group
    -----
    align
    """
    return _best_slices_from_mass(projected_mass(stack.data), nslices)


def _best_slices_from_mass(mass: np.ndarray, nslices: int) -> np.ndarray:
    """Get the best slices for CoM analysis from projected mass data."""
    total_mass = mass.sum((0, 1))
    mass_std = mass.sum(1).std(0)
    mass_std[mass_std == 0] = 1e-5
    mass_ratio = total_mass / mass_std
    return mass_ratio.argsort()[::-1][0:nslices]


def get_coms(stack: "TomoStack", slices: np.ndarray) -> np.ndarray:
    """
    Calculate the center of mass for indicated slices.

    Parameters
    ----------
    stack
        Tilt series from which to calculate the centers of mass.
    slices
        Location of slices to use for center of mass calculation.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Center of mass as a function of tilt for each slice [ntilts, nslices].

    Group
    -----
    align
    """
    return _coms_from_mass(projected_mass(stack.data)[:, :, slices])


def _coms_from_mass(sinos: np.ndarray) -> np.ndarray:
    """Calculate the CoM (in pixels from the y-center) of projected mass data."""
    # pixel coordinates relative to the tilt axis
    y_coordinates = np.arange(sinos.shape[1]) - (sinos.shape[1] - 1) / 2
    total_mass = sinos.sum(1)
    coms = np.sum(np.transpose(sinos, [0, 2, 1]) * y_coordinates, 2) / total_mass
    return coms


# Fourier rotations are applied as three shears; limit them to where this is accurate
MAX_FOURIER_ROTATION = 90.0


def fourier_shift(
    data: np.ndarray,
    shifts: np.ndarray | float,
    axis: int,
    xp: Any = np,
) -> np.ndarray:
    """
    Shift an array along one axis using the Fourier shift theorem.

    Each line of ``data`` along ``axis`` is shifted by its own amount: ``shifts``
    must broadcast to the shape of ``data`` without ``axis`` (for example one
    value per image of a stack, with shape ``(nimages, 1)``, or one value per
    line). Positive shifts move the content towards higher indices. The data are
    zero-padded by more than the largest shift, so that content does not wrap
    around, and pixels uncovered by the shift are filled with zeros (subtract the
    background first to fill with the background level).

    Parameters
    ----------
    data
        Real array to shift
    shifts
        Shift in pixels of each line along ``axis``
    axis
        Axis along which to shift
    xp
        Array module, :py:mod:`numpy` (default) or :py:mod:`cupy`

    Returns
    -------
    shifted : :py:class:`~numpy.ndarray`
        Shifted copy of ``data`` (floating point)

    Group
    -----
    align
    """
    float_dtype = xp.float64 if data.dtype == xp.float64 else xp.float32
    work = xp.moveaxis(xp.asarray(data, dtype=float_dtype), axis, -1)
    shifts = xp.broadcast_to(
        xp.asarray(shifts, dtype=float_dtype),
        work.shape[:-1],
    )
    if not xp.any(shifts):
        return xp.moveaxis(work, -1, axis).copy()
    n = work.shape[-1]
    pad = int(np.ceil(float(xp.abs(shifts).max()))) + 1
    length = next_fast_len(n + 2 * pad, real=True)
    padding = [(0, 0)] * (work.ndim - 1) + [(pad, length - n - pad)]
    padded = xp.pad(work, padding)
    freqs = xp.fft.rfftfreq(length).astype(float_dtype)
    phase = xp.exp(-2j * float_dtype(np.pi) * shifts[..., np.newaxis] * freqs)
    shifted = xp.fft.irfft(xp.fft.rfft(padded, axis=-1) * phase, n=length, axis=-1)
    shifted = shifted[..., pad : pad + n].astype(float_dtype)
    return xp.moveaxis(shifted, -1, axis)


def fourier_rigid_transform(
    data: np.ndarray,
    angle: float,
    translations: np.ndarray,
) -> np.ndarray:
    """
    Rotate and translate each image of a stack using Fourier methods.

    Each image is rotated by ``angle`` (degrees) about its center (``nx / 2``,
    ``ny / 2``, as in :py:meth:`~etspy.base.TomoStack.trans_stack`) and then
    translated, so that content at ``(x, y)`` moves to ``R (x - c) + c + t``.
    The rotation is decomposed into three shears, each applied as one-dimensional
    Fourier shifts (with the translation included in the last two), which
    avoids the loss of resolution caused by interpolation. Pixels uncovered by the
    transformation are filled with the background level of each image.

    Parameters
    ----------
    data
        Stack of images of shape ``(nimages, ny, nx)``
    angle
        Rotation angle in degrees (at most 90 in magnitude)
    translations
        Translation ``(tx, ty)`` of each image in pixels (``x`` along columns and
        ``y`` along rows), shape ``(nimages, 2)``, or a single ``(tx, ty)``

    Returns
    -------
    transformed : :py:class:`~numpy.ndarray`
        Transformed stack (floating point)

    Group
    -----
    align
    """
    if abs(angle) > MAX_FOURIER_ROTATION:
        msg = (
            f"Fourier rotation is limited to angles of at most {MAX_FOURIER_ROTATION} "
            f"degrees (was {angle})"
        )
        raise ValueError(msg)
    nimages, ny, nx = data.shape
    translations = np.broadcast_to(
        np.asarray(translations, dtype=float),
        (nimages, 2),
    )
    theta = np.deg2rad(angle)
    a, b = -np.tan(theta / 2), np.sin(theta)
    x_rel = np.arange(nx) - nx / 2
    y_rel = np.arange(ny) - ny / 2
    float_dtype = np.float64 if data.dtype == np.float64 else np.float32
    out = np.empty(data.shape, dtype=float_dtype)
    for i in range(nimages):
        background = background_mode(data[i])
        image = data[i].astype(np.float64) - background
        tx, ty = translations[i]
        # rows (fixed y) shift along x, then columns along y, then rows along x
        image = fourier_shift(image, a * y_rel, axis=1)
        image = fourier_shift(image, b * x_rel + ty, axis=0)
        image = fourier_shift(image, a * y_rel + tx - a * ty, axis=1)
        out[i] = image + background
    return out


def apply_shifts(
    stack: "TomoStack",
    shifts: Union["TomoShifts", np.ndarray],
    method: Literal["interp", "fourier"] = "fourier",
    cuda: bool = False,
    **kwargs,
) -> "TomoStack":
    """
    Apply a series of shifts to a TomoStack.

    Shifts are applied to the data using either interpolation or Fourier shift methods.
    These operations are carried out using either CPU or GPU resources depending on the
    value of `cuda`. If `cuda` is True, the shifts will be applied using
    GPU-acceleration via CuPy. The shifts are stored in
    ``shifted.metadata.Tomography.shifts``. With either method, pixels uncovered by
    a shift are filled with the background (vacuum) level of their projection,
    estimated from its intensity histogram (zero if the background has been removed
    with :py:meth:`~etspy.base.TomoStack.remove_background`). The result is
    floating point.

    Parameters
    ----------
    stack : :py:class:`~TomoStack`
        The image series to be aligned
    shifts : Union[:py:class:`~TomoShifts`, :py:class:`~numpy.ndarray`]
        The X- (tilt parallel) and Y-shifts (tilt perpendicular) to be applied to
        each image. Should be of size
        ``(*stack.axes_manager.navigation_shape[::-1], 2)``,
        with Y-shifts in the ``shifts[:, 0]`` position and X-shifts in ``shifts[:, 1]``
        position (if ``shifts`` is a :py:class:`~numpy.ndarray`).
    method : :py:class:`~str`
        Image shifts can be applied using either interpolation via scipy.ndimage.shift
        or via Fourier shift as implemented in scipy.ndimage.fourier_shift.  Must be
        either 'interp' or 'fourier'.
    cuda : :py:class:`~bool`
        Enable/disable the use of GPU-accelerated processes using CUDA. If True, shifts
        will be applied using CuPy. If False, shifts will be applied using NumPy and
        SciPy.

    Returns
    -------
    shifted : TomoStack
        Copy of input stack after shifts are applied

    Group
    -----
    align
    """
    if cuda and not has_cupy:
        msg = (
            "CUDA was requested (cuda=True), but CuPy is not "
            "available. Use cuda=False to apply the shifts on the CPU."
        )
        raise ValueError(msg)
    shifted = stack.deepcopy()
    xp = cp if cuda else np

    if isinstance(shifts, BaseSignal):
        shifts = shifts.data
    shifts = cast("np.ndarray", shifts)

    if len(shifts) != stack.data.shape[0]:
        msg = (
            f"Number of shifts ({len(shifts)}) is not consistent "
            f"with number of images in the stack ({stack.data.shape[0]})"
        )
        raise ValueError(msg)

    # Shift relative to the background of each projection, so that pixels uncovered
    # by a shift are filled with the background level
    float_dtype = np.float64 if shifted.data.dtype == np.float64 else np.float32
    background = np.array(
        [background_mode(im) for im in shifted.data],
        dtype=float_dtype,
    )[:, np.newaxis, np.newaxis]
    shifts = xp.array(shifts)
    data = xp.array(shifted.data.astype(float_dtype) - background)

    if method.lower() == "interp":
        order = kwargs.pop("order", 3)
        shift_func = shift_gpu if cuda else ndimage.shift
        for i in range(data.shape[0]):
            data[i, :, :] = shift_func(
                data[i, :, :],
                shift=[shifts[i, 0], shifts[i, 1]],
                order=order,
            )
    elif method.lower() == "fourier":
        # shift along y (rows) and then x (columns); with a single shift per
        # image this is the same as one two-dimensional Fourier shift
        data = fourier_shift(data, shifts[:, 0:1].astype(float_dtype), axis=1, xp=xp)
        data = fourier_shift(data, shifts[:, 1:2].astype(float_dtype), axis=2, xp=xp)
    else:
        msg = f"Invalid shift application method {method}."
        raise ValueError(msg)

    shifted.data = (cp.asnumpy(data) if cuda else data) + background
    shifts = cp.asnumpy(shifts) if cuda else shifts
    shifted.shifts.data = shifted.shifts.data + shifts
    return shifted


class StackAligner(ABC):
    """Abstract Base class for Alignment methods."""

    def __init__(
        self,
        stack: "TomoStack",
        start: int | None = None,
        use_cuda: bool | None = False,
        show_progressbar: bool = False,
        **kwargs,
    ):
        use_cuda = cast("bool", use_cuda)
        self.stack = stack
        self.shifts = np.zeros([stack.data.shape[0], 2])
        if start is None:
            start = stack.data.shape[0] // 2
        start = cast("int", start)
        self.start = start
        self.use_cuda = use_cuda
        self.show_progressbar = show_progressbar
        self.kwargs = kwargs

    def align(
        self,
        shift_method: Literal["interp", "fourier"] = "fourier",
    ) -> "TomoStack":
        """
        Perform stack alignment.

        Shifts are calculated using the provided strategy and then applied using the
        specified shift method.
        """
        self.shifts = self.calculate_shifts()

        # Pass the strategy's CUDA preference to the applicator
        return apply_shifts(
            self.stack,
            self.shifts,
            method=shift_method,
            cuda=self.use_cuda,
        )

    @abstractmethod
    def calculate_shifts(self) -> np.ndarray:
        """
        Calculate the alignment shifts using the selected strategy.

        Returns
        -------
        shifts : :py:class:`~numpy.ndarray`
            An array of shape ``(n_images, 2)`` containing the calculated
            shifts. The first column ``shifts[:, 0]`` should contain
            Y-shifts (perpendicular to the tilt axis) and the second column
            ``shifts[:, 1]`` should contain X-shifts (parallel to the tilt axis).

        """


class PhaseCorrelationAligner(StackAligner):
    """
    Aligner class for phase correlation (PC) strategy.

    If `use_cuda` is False, shifts are determined using PC as implemented in
    scikit-image. Based on:
    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup.
    Efficient subpixel image registration algorithms, Optics Letters vol. 33
    (2008) pp. 156-158.
    https://doi.org/10.1364/OL.33.000156

    If `use_cuda` is True, shifts are determined using a custom implementation of the
    same PC algorithm which is optimized for GPU-acceleration using CuPy and CUDA.

    Initialiazer for phase correlation alignment strategy.

    Atrributes
    ----------
    start : py:class:`~int`
        Position in tilt series to use as starting point for the alignment
    upsample_factor : py:class:`~int`
        Factor by which to resample the data for phase correlation
    use_cuda : py:class:`~bool`
        Enable/disable the use of GPU-accelerated processes using CUDA
    show_progressbar : py:class:`~bool`
        Enable/disable progress bar

    """

    def __init__(
        self,
        stack: "TomoStack",
        start: int | None = 0,
        use_cuda: bool = False,
        show_progressbar: bool = True,
        **kwargs,
    ):
        super().__init__(
            stack=stack,
            start=start,
            use_cuda=use_cuda,
            show_progressbar=show_progressbar,
            **kwargs,
        )
        self.upsample_factor = kwargs.get("upsample_factor", 3)

    def calculate_shifts(self) -> np.ndarray:
        """
        Calculate shifts using the phase correlation algorithm.

        Parameters
        ----------
        stack : :py:class:`~TomoStack`
            The image stack to be aligned

        Returns
        -------
        shifts : :py:class:`~numpy.ndarray`
            The X- and Y-shifts to be applied to each image

        Group
        -----
        align
        """
        if has_cupy and self.use_cuda:
            shifts = self._cupy_calculate_shifts()

        else:
            shifts = np.zeros((self.stack.data.shape[0], 2))
            with tqdm.tqdm(
                total=self.stack.data.shape[0] - 1,
                desc="Calculating shifts",
                disable=not self.show_progressbar,
            ) as pbar:
                for i in range(self.start, 0, -1):
                    shift = pcc(
                        self.stack.data[i],
                        self.stack.data[i - 1],
                        upsample_factor=self.upsample_factor,
                    )[0]
                    shifts[i - 1] = shifts[i] + shift
                    pbar.update(1)

                for i in range(self.start, self.stack.data.shape[0] - 1):
                    shift = pcc(
                        self.stack.data[i],
                        self.stack.data[i + 1],
                        upsample_factor=self.upsample_factor,
                    )[0]
                    shifts[i + 1] = shifts[i] + shift
                    pbar.update(1)
        return shifts

    def _cupy_calculate_shifts(self):
        """Calculate shifts of stack using CUDA-implementation of phase correlation."""
        stack_cp = cp.array(self.stack.data)
        shifts = cp.zeros([stack_cp.shape[0], 2])
        ref_cp = stack_cp[0]
        ref_fft = cp.fft.fftn(ref_cp)
        shape = ref_fft.shape
        with tqdm.tqdm(
            total=self.stack.data.shape[0] - 1,
            desc="Calculating shifts",
            disable=not self.show_progressbar,
        ) as pbar:
            for i in range(self.start, 0, -1):
                shift = self._cupy_phase_correlate(
                    stack_cp[i],
                    stack_cp[i - 1],
                    shape=shape,
                )
                shifts[i - 1] = shifts[i] + shift
                pbar.update(1)
            for i in range(self.start, self.stack.data.shape[0] - 1):
                shift = self._cupy_phase_correlate(
                    stack_cp[i],
                    stack_cp[i + 1],
                    shape=shape,
                )
                shifts[i + 1] = shifts[i] + shift
                pbar.update(1)
        shifts = shifts.get()
        return shifts

    def _cupy_phase_correlate(self, ref_cp, mov_cp, shape):
        """CUDA-implementation of phase correlation."""
        # missing coverage b/c of CUDA
        ref_fft = cp.fft.fftn(ref_cp)
        mov_fft = cp.fft.fftn(mov_cp)

        cross_power_spectrum = ref_fft * mov_fft.conj()
        eps = cp.finfo(cross_power_spectrum.real.dtype).eps
        cross_power_spectrum /= cp.maximum(cp.abs(cross_power_spectrum), 100 * eps)
        phase_correlation = cp.fft.ifft2(cross_power_spectrum)

        maxima = cp.unravel_index(
            cp.argmax(cp.abs(phase_correlation)),
            phase_correlation.shape,
        )
        midpoint = cp.array([cp.fix(axis_size / 2) for axis_size in shape])

        float_dtype = cross_power_spectrum.real.dtype

        shift = cp.stack(maxima).astype(float_dtype, copy=False)
        shift[shift > midpoint] -= cp.array(shape)[shift > midpoint]

        if self.upsample_factor > 1:
            upsample_factor = cp.array(self.upsample_factor, dtype=float_dtype)
            upsampled_region_size = cp.ceil(upsample_factor * 1.5)
            dftshift = cp.fix(upsampled_region_size / 2.0)

            shift = cp.round(shift * upsample_factor) / upsample_factor

            sample_region_offset = dftshift - shift * upsample_factor
            phase_correlation = self._upsampled_dft(
                cross_power_spectrum.conj(),
                upsampled_region_size,
                upsample_factor,
                sample_region_offset,
            ).conj()
            maxima = np.unravel_index(
                cp.argmax(np.abs(phase_correlation)),
                phase_correlation.shape,
            )

            maxima = cp.stack(maxima).astype(float_dtype, copy=False)
            maxima -= dftshift

            shift += maxima / upsample_factor
        return shift

    def _upsampled_dft(
        self,
        data,
        upsampled_region_size,
        upsample_factor,
        axis_offsets,
    ):
        """
        CuPy-implmentation of DFT upsampling algorithm.

        This code is a CuPy adaptation of the algorithm used in:

        scikit-image.registration._phase_cross_correlation.
        https://github.com/scikit-image/scikit-image/blob/v0.26.0/src/skimage/registration/_phase_cross_correlation.py

        It is intended to achieve sub-pixel precision while avoiding padding of the
        dataset and the related risk of memory limitations. See scikit-image source for
        more detail.

        Parameters
        ----------
        data : :py:class:`~numpy.ndarray`
            DFT of original data to upsample.
        upsampled_region_size : :py:class:`~int`
            The size of the region to be sampled.
        upsample_factor : :py:class:`~int`
            Factor by which to upsample the DFT.
        axis_offsets : :py:class:`~list` of :py:class:`~int`
            The offsets of the region to be sampled.

        Returns
        -------
        output : :py:class:`~numpy.ndarray`
            The upsampled DFT of the specified region.
        """
        # missing coverage because of CUDA
        upsampled_region_size = [
            upsampled_region_size,
        ] * data.ndim

        im2pi = 1j * 2 * cp.pi

        dim_properties = list(
            zip(
                data.shape,
                upsampled_region_size,
                axis_offsets,
                strict=False,
            ),
        )

        for n_items, ups_size, ax_offset in dim_properties[::-1]:
            kernel = (cp.arange(ups_size) - ax_offset)[:, None] * cp.fft.fftfreq(
                n_items,
                upsample_factor,
            )
            kernel = cp.exp(-im2pi * kernel)
            # use kernel with same precision as the data
            kernel = kernel.astype(data.dtype, copy=False)
            data = cp.tensordot(kernel, data, axes=(1, -1))  # type: ignore
        return data


class StackRegAligner(StackAligner):
    """
    Aligner class for StackReg strategy.

    Calculated rigid translational shifts using PyStackReg. PyStackReg is a Python port
    of the StackReg plugin for ImageJ which uses a pyramidal approach to minimize the
    least-squares difference in image intensity between a source and target image.
    StackReg is described in:
    P. Thevenaz, U.E. Ruttimann, M. Unser. A Pyramid Approach to
    Subpixel Registration Based on Intensity, IEEE Transactions
    on Image Processing vol. 7, no. 1, pp. 27-41, January 1998.
    https://doi.org/10.1109/83.650848

    Attributes
    ----------
    start
        Position in tilt series to use as starting point for the alignment.
        If ``None``, the slice closest to the midpoint will be used.
    show_progressbar
        Enable/disable progress bar

    Returns
    -------
    shifts : :py:class:`~numpy.ndarray`
        The X- and Y-shifts to be applied to each image

    Group
    -----
    align
    """

    def __init__(
        self,
        stack: "TomoStack",
        start: int | None = 0,
        use_cuda: bool = False,
        show_progressbar: bool = True,
        **kwargs,
    ):
        super().__init__(
            stack=stack,
            start=start,
            use_cuda=use_cuda,
            show_progressbar=show_progressbar,
            **kwargs,
        )

    def calculate_shifts(self) -> np.ndarray:
        """
        Calculate shifts using PyStackReg.

        Parameters
        ----------
        stack
            The image series to be aligned

        Returns
        -------
        shifts : :py:class:`~numpy.ndarray`
            The shifts to be applied to each image

        """
        shifts = np.zeros((self.stack.data.shape[0], 2))

        if self.start is None:
            self.start = (
                self.stack.data.shape[0] // 2
            )  # Use the midpoint if start is not provided
        self.start = cast("int", self.start)

        # Initialize pystackreg object with TranslationTransform2D
        reg = StackReg(StackReg.TRANSLATION)

        with tqdm.tqdm(
            total=self.stack.data.shape[0] - 1,
            desc="Calculating shifts",
            disable=not self.show_progressbar,
        ) as pbar:
            # Calculate shifts relative to the image at the 'start' index
            for i in range(self.start, 0, -1):
                transformation = reg.register(
                    self.stack.data[i],
                    self.stack.data[i - 1],
                )
                shift = -transformation[0:2, 2][::-1]
                shifts[i - 1] = shifts[i] + shift
                pbar.update(1)

            for i in range(self.start, self.stack.data.shape[0] - 1):
                transformation = reg.register(
                    self.stack.data[i],
                    self.stack.data[i + 1],
                )
                shift = -transformation[0:2, 2][::-1]
                shifts[i + 1] = shifts[i] + shift
                pbar.update(1)
        return shifts


class CoMAligner(StackAligner):
    """
    Center of mass (COM) tracking alignment strategy.

    A Python implementation of algorithms described in:
    T. Sanders. Physically motivated global alignment method for electron
    tomography, Advanced Structural and Chemical Imaging vol. 1 (2015) pp 1-11.
    https://doi.org/10.1186/s40679-015-0005-7

    Attributes
    ----------
    start
        Position in tilt series to use as starting point for the alignment.
        If ``None``, the slice closest to the midpoint will be used.
    show_progressbar
        Enable/disable progress bar

    Returns
    -------
    shifts : :py:class:`~numpy.ndarray`
        The X- and Y-shifts to be applied to each image

    Group
    -----
    align
    """

    def __init__(
        self,
        stack: "TomoStack",
        start: int | None = 0,
        use_cuda: bool = False,
        show_progressbar: bool = True,
        **kwargs,
    ):
        super().__init__(
            stack=stack,
            start=start,
            use_cuda=use_cuda,
            show_progressbar=show_progressbar,
            **kwargs,
        )
        self.start = start
        self.show_progressbar = show_progressbar
        self.xrange = kwargs.get("xrange")
        self.p = kwargs.get("p", 20)
        self.nslices = kwargs.get("nslices", 20)

    def calculate_shifts(self) -> np.ndarray:
        """Calculate shifts using center of mass tracking method."""
        logger.info("Performing stack registration using center of mass method")
        shifts = np.zeros([self.stack.data.shape[0], 2])
        self._mass = projected_mass(self.stack.data)
        # _calculate_shifts_conservation_of_mass: x-shifts (parallel to tilt axis)
        # _calculate_shifts_com: y-shifts (perpendicular to tilt axis)
        shifts[:, 1] = self._calculate_shifts_conservation_of_mass()
        shifts[:, 0] = self._calculate_shifts_com()
        return shifts

    def _calculate_shifts_conservation_of_mass(
        self,
    ) -> np.ndarray:
        """
        Calculate shifts parallel to the tilt axis using conservation of mass.

        The mass of each projection summed perpendicular to the tilt axis (its
        profile along the tilt axis) is shifted by up to ``p`` pixels in each
        direction to best match the mean profile within a window of the x-axis.
        The window is the central 60% of the x-axis, or ``xrange`` reduced by
        ``p`` at each end if ``xrange`` is given. ``p`` is reduced if needed so
        that the shifted window stays within the image.

        Returns
        -------
        xshifts : :py:class:`~numpy.ndarray`
            Calculated shifts parallel to tilt axis.

        Raises
        ------
        ValueError
            If the window is empty, or there is no room to shift it

        Group
        -----
        align
        """
        logger.info("Refinining X-shifts using conservation of mass method")
        nx = self.stack.data.shape[2]

        if self.xrange is None:
            x_start, x_stop = round(nx / 5), round(4 / 5 * nx)
            search = min(self.p, x_start, nx - x_stop)
        else:
            data_start = max(round(self.xrange[0]), 0)
            data_stop = min(round(self.xrange[1]), nx)
            search = min(self.p, (data_stop - data_start - 1) // 2)
            x_start, x_stop = data_start + search, data_stop - search
        if search < 1 or x_stop <= x_start:
            msg = (
                "Not enough pixels along the x-axis to calculate shifts parallel "
                f"to the tilt axis (x-axis size {nx}, xrange {self.xrange})"
            )
            raise ValueError(msg)
        if search < self.p:
            logger.info("Limiting the x-shift search range to %d pixels", search)

        profiles = self._mass.sum(1)
        mean_profile = profiles[:, x_start:x_stop].mean(0)
        offsets = np.arange(-search, search + 1)
        residuals = np.stack(
            [
                np.linalg.norm(
                    profiles[:, x_start + j : x_stop + j] - mean_profile,
                    axis=1,
                )
                for j in offsets
            ],
            axis=1,
        )
        return -offsets[residuals.argmin(1)].astype(float)

    def _calculate_shifts_com(
        self,
    ) -> np.ndarray:
        """
        Calculate shifts perpendicular to the tilt axis by center of mass analysis.

        In an aligned tilt series, the center of mass (CoM) of each slice
        perpendicular to the tilt axis follows ``x cos(theta) + z sin(theta)``,
        where ``(x, z)`` is the CoM of that slice of the object. The misalignment
        of each projection is the part of the measured CoM that does not follow
        this path, i.e. the residual of a least squares fit of the CoM to
        ``cos(theta)`` and ``sin(theta)``. The residuals of the ``nslices`` slices
        with the most consistent mass are averaged.

        Returns
        -------
        yshifts : :py:class:`~numpy.ndarray`
            Shifts perpendicular to the tilt axis for each projection

        Group
        -----
        align
        """
        logger.info("Refinining Y-shifts using center of mass method")
        slices = _best_slices_from_mass(self._mass, self.nslices)

        thetas = np.deg2rad(np.asarray(self.stack.tilts.data, dtype=float).squeeze())
        coms = _coms_from_mass(self._mass[:, :, slices])
        design = np.array([np.cos(thetas), np.sin(thetas)]).T
        fit = design @ np.linalg.lstsq(design, coms, rcond=None)[0]
        return -(coms - fit).mean(1)


class CommonLineAligner(StackAligner):
    """
    Aligner class for common line (CL) strategy.

    A combination of center of mass tracking for aligment of projections perpendicular
    to the tilt axis and common line alignment for parallel to the tilt axis. This is a
    Python implementation of Matlab code described in:
    M. C. Scott, et al. Electron tomography at 2.4-ångström resolution,
    Nature 483, 444-447 (2012).
    https://doi.org/10.1038/nature10934

    Attributes
    ----------
    start : py:class:`~int`
        Position in tilt series to use as starting point for the alignment. If ``None``,
        the slice closest to the midpoint will be used.
    com_ref_index : py:class:`~int`
        Reference slice for center of mass alignment.  All other slices
        will be aligned to this reference.
    cl_ref_index : py:class:`~int`
        Reference slice for common line alignment.  All other slices
        will be aligned to this reference. If not provided the projection
        closest to the middle of the stack will be chosen.
    cl_resolution : py:class:`~float`
        Resolution for subpixel common line alignment. Default is 0.05.
        Should be less than 0.5.
    cl_div_factor : py:class:`~int`
        Factor which determines the number of iterations of common line
        alignment to perform.  Default is 8.
    """

    def __init__(
        self,
        stack: "TomoStack",
        start: int | None = 0,
        use_cuda: bool = False,
        show_progressbar: bool = True,
        **kwargs,
    ):
        super().__init__(
            stack=stack,
            start=start,
            use_cuda=use_cuda,
            show_progressbar=show_progressbar,
            **kwargs,
        )
        if start is None:
            start = stack.data.shape[0] // 2
        self.com_ref_index = kwargs.get("com_ref_index")
        self.cl_ref_index = kwargs.get("cl_ref_index")
        self.cl_resolution = kwargs.get("cl_resolution", 0.05)
        self.cl_div_factor = kwargs.get("cl_div_factor", 8)
        if self.com_ref_index is None:
            self.com_ref_index = self.start
        if self.cl_ref_index is None:
            self.cl_ref_index = self.start

    def calculate_shifts(self) -> np.ndarray:
        """Calculate shifts using combined center of mass and common line methods."""
        logger.info(
            "Performing stack registration using combined "
            "center of mass and common line methods",
        )
        # explicit type casts for type checking
        self.com_ref_index = cast("int", self.com_ref_index)
        self.cl_ref_index = cast("int", self.cl_ref_index)

        shifts = self._calc_shifts_com_cl()
        return shifts

    def _calc_shifts_com_cl(self) -> np.ndarray:
        """
        Calculate shifts using combined center of mass and common line methods.

        Returns
        -------
        shifts : :py:class:`~numpy.ndarray`
            The calculated shifts to be applied to each image

        Group
        -----
        align
        """
        if self.cl_resolution >= CL_RES_THRESHOLD:
            msg = f"Resolution should be less than {CL_RES_THRESHOLD}"
            raise ValueError(msg)

        logger.info("Center of mass reference slice: %s", self.com_ref_index)
        logger.info("Common line reference slice: %s", self.cl_ref_index)
        xshifts = np.zeros(self.stack.data.shape[0])
        yshifts = np.zeros(self.stack.data.shape[0])
        com_ref_index = cast("int", self.com_ref_index)
        yshifts = self._calc_yshifts(com_ref_index)
        xshifts = self.calc_shifts_cl(
            self.cl_ref_index,
            self.cl_resolution,
            self.cl_div_factor,
        )
        shifts = np.stack([yshifts, xshifts], axis=1)
        return shifts

    def _calc_yshifts(self, com_ref_index: int) -> np.ndarray:
        """
        Calculate Y-shifts using center of mass tracking.

        Each projection is shifted so that its center of mass (perpendicular to
        the tilt axis) matches that of the reference projection.

        Parameters
        ----------
        com_ref_index
            Tilt index of the reference projection.

        Returns
        -------
        yshifts : :py:class:`~numpy.ndarray`
            Shifts perpendicular to the tilt axis for each projection
        """
        ntilts = self.stack.data.shape[0]
        coms = np.zeros(ntilts)
        mass = projected_mass(self.stack.data)

        for i in tqdm.tqdm(range(ntilts), disable=not self.show_progressbar):
            coms[i], _ = ndimage.center_of_mass(mass[i])
        yshifts = coms[com_ref_index] - coms
        return yshifts

    def calc_shifts_cl(
        self,
        cl_ref_index: int | None,
        cl_resolution: float,
        cl_div_factor: int,
    ) -> np.ndarray:
        """
        Calculate shifts using the common line method.

        Used to align stack in dimension parallel to the tilt axis

        Parameters
        ----------
        stack
            The stack on which to calculate shifts
        cl_ref_index
            Tilt index of reference projection. If not provided the projection
            closest to the middle of the stack will be chosen.
        cl_resolution
            Degree of sub-pixel analysis
        cl_div_factor
            Factor used to determine number of iterations of alignment.

        Returns
        -------
        yshifts : :py:class:`~numpy.ndarray`
            Shifts parallel to tilt axis for each projection

        Group
        -----
        align
        """

        def align_line(ref_line, line, cl_resolution, cl_div_factor):
            npad = len(ref_line) * 2 - 1

            # Pad with zeros while preserving the center location
            ref_line_pad = self.pad_line(ref_line, npad)
            line_pad = self.pad_line(line, npad)

            niters = int(
                np.abs(np.floor(np.log(cl_resolution) / np.log(cl_div_factor))),
            )
            start, end = -0.5, 0.5

            ref_line_pad_ft = np.fft.fftshift(
                np.fft.fft(np.fft.ifftshift(ref_line_pad)),
            )
            line_pad_ft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(line_pad)))

            midpoint = (npad - 1) / 2
            kx = np.arange(-midpoint, midpoint + 1)

            for _ in range(niters):
                boundary = np.linspace(start, end, cl_div_factor, endpoint=False)
                index = (boundary[:-1] + boundary[1:]) / 2

                max_vals = np.zeros(len(index))
                for j, idx in enumerate(index):
                    pfactor = np.exp(2 * np.pi * 1j * (idx * kx / npad))
                    conjugate = np.conj(ref_line_pad_ft) * line_pad_ft * pfactor
                    xcorr = np.abs(
                        np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(conjugate))),
                    )
                    max_vals[j] = np.max(xcorr)

                max_loc = np.argmax(max_vals)
                start, end = boundary[max_loc], boundary[max_loc + 1]

            subpixel_shift = index[max_loc]
            max_pfactor = np.exp(2 * np.pi * 1j * (subpixel_shift * kx / npad))

            # Determine integer shift via cross correlation
            conjugate = np.conj(ref_line_pad_ft) * line_pad_ft * max_pfactor
            xcorr = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(conjugate))))
            max_loc = np.argmax(xcorr)

            integer_shift = max_loc
            integer_shift = integer_shift - midpoint

            # Calculate full shift
            shift = integer_shift + subpixel_shift
            return -shift

        if cl_ref_index is None:
            cl_ref_index = self.stack.data.shape[0] // 2

        yshifts = np.zeros(self.stack.data.shape[0])
        ref_cm_line = self.stack.data[cl_ref_index].sum(0)

        for i in tqdm.tqdm(
            range(self.stack.data.shape[0]),
            disable=not self.show_progressbar,
        ):
            if i == cl_ref_index:
                continue
            curr_cm_line = self.stack.data[i].sum(0)
            yshifts[i] = align_line(
                ref_cm_line,
                curr_cm_line,
                cl_resolution,
                cl_div_factor,
            )
        return yshifts

    def pad_line(self, line: np.ndarray, paddedsize: int) -> np.ndarray:
        """
        Pad a 1D array for FFT treatment without altering center location.

        Parameters
        ----------
        line
            The data to be padded (should be 1D)
        paddedsize
            The size of the desired padded data.

        Returns
        -------
        padded : :py:class:`~numpy.ndarray`
            Padded version of input data (1 dimensional)

        Group
        -----
        align
        """
        npix = len(line)
        start_index = (paddedsize - npix) // 2
        end_index = start_index + npix
        padded_line = np.zeros(paddedsize)
        padded_line[start_index:end_index] = line
        return padded_line


class TiltAligner(ABC):
    """Abstract class for performing tilt axis alignment."""

    def __init__(self, stack, **kwargs):
        self.stack = stack
        self.kwargs = kwargs
        self.interpolation = kwargs.get("interpolation", "fourier")

    @abstractmethod
    def align_tilt_axis(self) -> "TomoStack":
        """Align the tilt axis using the selected strategy."""


class TiltCOMAligner(TiltAligner):
    """Tilt alignment class for center of mass method.

    For each of a number of slices, the CoM is measured at each tilt and
    fit to the path expected for rotation about an axis,
    ``r + x0 * cos(theta) + z0 * sin(theta)``, where ``r`` is the position of the
    tilt axis in that slice. A straight line fit of ``r`` along the tilt axis gives
    the shift and in-plane rotation needed to center the tilt axis and make it
    parallel to the x-axis; each slice is weighted by how closely its CoM follows
    the expected path.

    ----------
    slices
        Locations at which to perform the Center of Mass analysis. If not
        provided, an appropriate list of slices will be automatically determined.
    nslices
        Nubmer of slices to use for the analysis (only used if the ``slices``
        parameter is not specified). The slices are spread along the central 80% of
        the tilt axis. If ``None``, a value of 10% of the x-axis size will be used,
        clamped to the range [3, 50].
    iterations
        Number of times to refine the estimate (default: 3).

    Returns
    -------
    out : TomoStack
        Copy of the input stack after rotation and translation to center and
        make the tilt axis horizontal

    Group
    -----
    align
    """

    EDGE_FRACTION = 0.1  # fraction of the x-axis at each end excluded from slices

    def __init__(self, stack, **kwargs):
        super().__init__(stack, **kwargs)
        self.slices = kwargs.get("slices")
        self.nslices = kwargs.get("nslices")
        self.iterations = kwargs.get("iterations", 3)

    def _select_slices(self, stack: "TomoStack") -> np.ndarray:
        """
        Select slices spread along the tilt axis for the CoM analysis.

        The central part of the x-axis is divided into ``nslices`` equal bins and
        the slice with the highest ratio of total mass to mass variance is chosen
        from each, so that the slices span the tilt axis (needed to determine the
        rotation) rather than clustering in one region.
        """
        nx = stack.data.shape[2]
        data = stack.data  # projected mass (see align_tilt_axis)
        total_mass = data.sum((0, 1))
        ratio = total_mass / np.maximum(data.sum(1).std(0), 1e-5)
        ratio[total_mass <= 0] = -np.inf  # never choose empty slices
        nslices = cast("int", self.nslices)
        bin_edges = np.linspace(
            self.EDGE_FRACTION * nx,
            (1 - self.EDGE_FRACTION) * nx,
            nslices + 1,
        ).astype(int)
        return np.array(
            [lo + np.argmax(ratio[lo:hi]) for lo, hi in pairwise(bin_edges) if hi > lo],
        )

    @staticmethod
    def _estimate(stack: "TomoStack", slices: np.ndarray) -> tuple[float, float]:
        """
        Estimate the tilt axis shift and rotation from the CoM at given slices.

        Returns
        -------
        tilt_shift, tilt_rotation
            Shift (pixels) and rotation (degrees) to pass to ``trans_stack`` to
            center the tilt axis and make it parallel to the x-axis
        """
        nx = stack.data.shape[2]
        thetas = np.deg2rad(stack.tilts.data.squeeze())
        coms = _coms_from_mass(stack.data[:, :, slices])

        # the tilt axis position r in each slice is the constant term of a linear
        # least squares fit of the CoM to [1, cos(theta), sin(theta)]
        design = np.stack([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)], 1)
        coeffs = np.linalg.lstsq(design, coms, rcond=None)[0]
        r = coeffs[0]

        # weight each slice by how well its CoM follows the expected path, so that
        # slices affected by e.g. material entering the field of view count less
        fit_rms = np.sqrt(((design @ coeffs - coms) ** 2).mean(0))
        weights = 1 / np.maximum(fit_rms, 1e-3 * np.median(fit_rms) + 1e-12)

        # r varies linearly along the tilt axis; evaluate it at the rotation
        # center used by trans_stack
        slope, intercept = np.polyfit(slices, r, 1, w=weights)
        tilt_shift = slope * nx / 2 + intercept
        tilt_rotation = -np.degrees(np.arctan(slope))
        return float(tilt_shift), float(tilt_rotation)

    def align_tilt_axis(self) -> "TomoStack":
        """Align tilt axis with center of mass tracking."""
        nx = self.stack.data.shape[2]
        nx_threshold = 3

        if np.all(self.stack.tilts.data == 0):
            msg = (
                "Tilts are not defined in stack.tilts (values were all zeros). "
                "Please set tilt values before alignment."
            )
            raise ValueError(msg)

        if nx < nx_threshold:
            msg = (
                f"Dataset is only {self.stack.data.shape[2]} pixels in x dimension. "
                "This method cannot be used."
            )
            raise ValueError(msg)

        if self.slices is None:
            if self.nslices is None:
                self.nslices = int(0.1 * nx)
                self.nslices = max(min(self.nslices, 50), 3)  # clamp nslices to [3, 50]
            else:
                if self.nslices > nx:
                    msg = "nslices is greater than the X-dimension of the data."
                    raise ValueError(msg)
                if self.nslices > 0.3 * nx:
                    self.nslices = int(0.3 * nx)
                    msg = (
                        "nslices is greater than 30% of number of x pixels. "
                        f"Using {self.nslices} slices instead."
                    )
                    logger.warning(msg)
            logger.info("Performing alignments using %s slices", self.nslices)
        else:
            self.slices = np.sort(self.slices)

        # Measure on a copy converted to projected mass (background removed and
        # negative values set to zero), so that the zeros used to fill pixels
        # uncovered by trans_stack in later iterations add no mass
        measure = self.stack.deepcopy()
        measure.data = projected_mass(measure.data)

        tilt_shift, tilt_rotation = 0.0, 0.0
        current = measure
        for i in range(self.iterations):
            slices = (
                self._select_slices(current) if self.slices is None else self.slices
            )
            shift_update, rotation_update = self._estimate(current, slices)
            tilt_shift += shift_update
            tilt_rotation += rotation_update
            if i < self.iterations - 1:
                # always transform from the untransformed data
                current = cast(
                    "TomoStack",
                    measure.trans_stack(yshift=tilt_shift, angle=tilt_rotation),
                )

        logger.info("Calculated tilt-axis shift %.2f", tilt_shift)
        logger.info("Calculated tilt-axis rotation %.2f", tilt_rotation)

        return cast(
            "TomoStack",
            self.stack.trans_stack(
                yshift=tilt_shift,
                angle=tilt_rotation,
                interpolation=self.interpolation,
            ),
        )


class TiltMaxImageAligner(TiltAligner):
    """Tilt alignment class for maximum image method.

    Perform automated determination of the tilt axis of a TomoStack.

    The projected maximum image used to determine the tilt axis by a
    combination of Sobel filtering and Hough transform analysis.

    Parameters
    ----------
    stack
        TomoStack array containing the tilt series data
    limit
        Maximum rotation angle to use for calculation
    delta
        Angular increment for calculation
    plot_results
        If ``True``, plot the maximum image along with the lines determined
        by Hough analysis
    also_shift
        If ``True``, also calculate and apply the global shift perpendicular to the tilt
        by minimizing the sum of the reconstruction
    shift_limit
        The limit of shifts applied if ``also_shift`` is set to ``True``

    Returns
    -------
    rotated : TomoStack
        Rotated version of the input stack

    Group
    -----
    align
    """

    def __init__(self, stack, **kwargs):
        super().__init__(stack, **kwargs)
        self.limit = kwargs.get("limit", 10)
        self.delta = kwargs.get("delta", 0.1)
        self.plot_results = kwargs.get("plot_results", False)
        self.also_shift = kwargs.get("also_shift", False)
        self.shift_limit = kwargs.get("shift_limit", 20.0)

    def align_tilt_axis(self) -> "TomoStack":
        """Align tilt axis using the maximum image method."""
        image = self.stack.data.max(0)
        image = image.astype("float32")
        edges = sobel(image)

        # Apply Canny edge detector for further edge enhancement
        edges = canny(edges)

        # Perform Hough transform to detect lines
        angles = np.pi * np.arange(-self.limit, self.limit, self.delta) / 180.0
        h, theta, d = hough_line(edges, angles)

        # Find peaks in Hough space
        _, angles, dists = hough_line_peaks(h, theta, d, num_peaks=5)

        # Calculate average angle from detected lines
        if len(angles) == 0:
            logger.warning(
                "No lines were found in the maximum image; the tilt axis "
                "rotation could not be determined and no rotation is applied",
            )
            rotation_angle = 0.0
        else:
            rotation_angle = float(np.degrees(np.mean(angles)))

        if self.plot_results:
            _, ax = plt.subplots(1)
            ax.imshow(image, cmap="gray")

            for i in range(len(angles)):
                (x0, y0) = dists[i] * np.array([np.cos(angles[i]), np.sin(angles[i])])
                ax.axline((x0, y0), slope=np.tan(angles[i] + np.pi / 2))

            plt.tight_layout()

        logger.info("Calculated tilt-axis rotation %.2f", -rotation_angle)

        if self.also_shift:
            # measure the shift on the rotated stack
            rotated = cast("TomoStack", self.stack.trans_stack(angle=-rotation_angle))
            tilt_shift = self._find_tilt_shift(rotated)
            logger.info("Calculated tilt-axis shift %.2f", -tilt_shift)
        else:
            tilt_shift = 0.0

        # produce the aligned stack from the input in a single transformation
        # (rotation followed by the shift); trans_stack adds the rotation and
        # shift to the alignment recorded in the stack metadata
        return cast(
            "TomoStack",
            self.stack.trans_stack(
                yshift=-tilt_shift,
                angle=-rotation_angle,
                interpolation=self.interpolation,
            ),
        )

    def _find_tilt_shift(self, ali: "TomoStack") -> float:
        """
        Find the tilt axis shift that minimizes the sum of the reconstruction.

        The central sinogram of ``ali`` is shifted perpendicular to the tilt axis
        by each whole number of pixels within ``shift_limit``, filling the pixels
        uncovered by the shift with the background level of each projection, and
        each shifted sinogram is reconstructed.
        """
        sinogram = ali.data[:, :, ali.data.shape[2] // 2]
        fill = np.array([background_mode(projection) for projection in ali.data])
        shifts = np.arange(-self.shift_limit, self.shift_limit, 1)
        shifted = np.empty((*sinogram.shape, len(shifts)), dtype=float)
        for i, shift in enumerate(shifts.astype(int)):
            shifted[:, :, i] = fill[:, np.newaxis]
            if shift >= 0:
                shifted[:, shift:, i] = sinogram[:, : sinogram.shape[1] - shift]
            else:
                shifted[:, :shift, i] = sinogram[:, -shift:]
        shifted_stack = type(ali)(shifted, tilts=ali.tilts)
        shifted_rec = shifted_stack.reconstruct("SIRT", 100, constrain=True)
        image_sum = cast("BaseSignal", shifted_rec.sum(axis=(1, 2)))
        return float(shifts[image_sum.data.argmin()])


def align_to_other(
    stack: "TomoStack",
    other: "TomoStack",
    shift_type: Literal["fourier", "interp"] = "fourier",
    cuda: bool = False,
) -> "TomoStack":
    """
    Spatially register a TomoStack using previously calculated shifts.

    The alignment recorded in ``stack`` (its image shifts, any cropping, and the
    tilt axis shift and rotation) is applied to ``other``. With
    ``shift_type="fourier"`` (default), all of it is applied in a single Fourier
    resampling of ``other`` (see :py:func:`fourier_rigid_transform`), which avoids
    the loss of resolution caused by resampling the data at each alignment step.
    This can also be used to apply the alignment determined from a stack to the
    original, unaligned version of the same data, for example
    ``aligned = ali.align_other(raw)``.

    Parameters
    ----------
    stack
        TomoStack which was previously aligned
    other
        TomoStack to be aligned. Must be the same size as the original
        (unaligned and uncropped) data from which ``stack`` was aligned
    shift_type
        ``"fourier"`` (default) to apply the whole alignment in a single Fourier
        resampling (on the CPU), or ``"interp"`` to apply the image shifts by
        spline interpolation and then the tilt axis shift and rotation by linear
        interpolation, as separate steps
    cuda
        Whether to use CUDA when applying the image shifts (``"interp"`` only)

    Returns
    -------
    out : TomoStack
        Aligned copy of other TomoStack

    Group
    -----
    align
    """
    stack_tomo_meta = cast("Dtb", stack.metadata.Tomography)

    tiltaxis = float(cast("float", stack_tomo_meta.tiltaxis))
    xshift = float(cast("float", stack_tomo_meta.xshift))
    yshift = float(cast("float", stack_tomo_meta.yshift))
    cropped = bool(stack_tomo_meta.get_item("cropped", default=False))
    shifts = np.asarray(stack.shifts.data, dtype=float)

    if shift_type == "fourier":
        # Apply all of the alignment (the image shifts, then the rotation and shift
        # of the tilt axis) in a single Fourier resampling of the original data
        out = other.deepcopy()
        _, ny, nx = out.data.shape
        theta = np.deg2rad(tiltaxis)
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        )
        # content moved by the image shift s (x, y) and then rotated about the
        # center c and translated by t ends up at R (p + s - c) + c + t
        image_shifts = shifts[:, ::-1]  # (x, y) order
        translations = image_shifts @ rotation.T + np.array([xshift, -yshift])
        if cropped:
            # trans_stack rotated the cropped stack about its own center c'; that
            # is a rotation about c followed by a translation of (I - R)(c' - c)
            y_start, y_stop, x_start, x_stop = _crop_bounds(shifts, ny, nx)
            crop_center = np.array(
                [(x_start + x_stop) / 2 - nx / 2, (y_start + y_stop) / 2 - ny / 2],
            )
            translations = translations + (np.eye(2) - rotation) @ crop_center
        out.data = fourier_rigid_transform(out.data, tiltaxis, translations)
        out.shifts = shifts
        if cropped:
            out = out.isig[x_start:x_stop, y_start:y_stop].deepcopy()
    else:
        out = other.deepcopy()
        out.shifts = np.zeros([out.data.shape[0], 2])
        out = apply_shifts(out, shifts, shift_type, cuda=cuda)
        if cropped:
            out = shift_crop(out)
        out = cast("TomoStack", out.trans_stack(xshift, yshift, tiltaxis))

    # record the alignment that was applied (trans_stack above adds to the existing
    # values, so set them explicitly)
    out_tomo_meta = cast("Dtb", out.metadata.Tomography)
    out_tomo_meta.tiltaxis = tiltaxis
    out_tomo_meta.xshift = xshift
    out_tomo_meta.yshift = yshift
    out_tomo_meta.cropped = cropped

    logger.info("TomoStack alignment applied")
    logger.info("X-shift: %.1f", xshift)
    logger.info("Y-shift: %.1f", yshift)
    logger.info("Rotation: %.1f", tiltaxis)
    return out


def _crop_bounds(shifts: np.ndarray, ny: int, nx: int) -> tuple[int, int, int, int]:
    """
    Pixel bounds of the area containing data in every image after shifting.

    A positive shift moves the image content towards higher indices, leaving
    empty pixels at the start of that axis, and a negative shift leaves empty
    pixels at the end. ``shifts[:, 0]`` are Y-shifts (rows) and ``shifts[:, 1]``
    are X-shifts (columns).

    Returns
    -------
    y_start, y_stop, x_start, x_stop
    """
    y_shifts, x_shifts = shifts[:, 0], shifts[:, 1]
    y_start = int(np.ceil(max(y_shifts.max(), 0)))
    y_stop = ny - int(np.ceil(max(-y_shifts.min(), 0)))
    x_start = int(np.ceil(max(x_shifts.max(), 0)))
    x_stop = nx - int(np.ceil(max(-x_shifts.min(), 0)))
    if y_stop <= y_start or x_stop <= x_start:
        msg = (
            "Shifts are too large to crop the stack to a common area "
            f"(Y-shift range: [{y_shifts.min():.1f}, {y_shifts.max():.1f}], "
            f"X-shift range: [{x_shifts.min():.1f}, {x_shifts.max():.1f}])"
        )
        raise ValueError(msg)
    return y_start, y_stop, x_start, x_stop


def shift_crop(stack: "TomoStack") -> "TomoStack":
    """
    Crop shifted stack to common area.

    Parameters
    ----------
    stack
        TomoStack which was previously aligned

    Returns
    -------
    out : TomoStack
        Cropped copy of input TomoStack

    Raises
    ------
    ValueError
        If the shifts are so large that no common area remains

    Group
    -----
    align
    """
    y_start, y_stop, x_start, x_stop = _crop_bounds(
        stack.shifts.data,
        *stack.data.shape[1:],
    )

    cropped = stack.isig[x_start:x_stop, y_start:y_stop].deepcopy()
    cropped.metadata.set_item("Tomography.cropped", value=True)
    return cropped
