"""Alignment module for ETSpy package."""

# pyright: reportPossiblyUnboundVariable=false

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from hyperspy.signal import BaseSignal
from pystackreg import StackReg
from scipy import ndimage, optimize
from scipy.fft import fft2, next_fast_len
from skimage.feature import canny
from skimage.filters import sobel
from skimage.registration import phase_cross_correlation as pcc
from skimage.transform import hough_line, hough_line_peaks

from etspy._gpu import affine_transform_gpu, cp, has_cupy, shift_gpu

if TYPE_CHECKING:
    from hyperspy.misc.utils import DictionaryTreeBrowser as Dtb

    from etspy.base import TomoShifts, TomoStack  # pragma: no cover

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CL_RES_THRESHOLD = 0.5  # threshold for common line registration method
BACKGROUND_WARN_FRACTION = 0.05
MASS_WARN_VARIATION = 0.1


def _check_mass_assumptions(stack: "TomoStack"):
    """Warn when data violate the assumptions of center of mass methods."""
    data = stack.data
    border = np.concatenate(
        [data[:, 0, :], data[:, -1, :], data[:, :, 0], data[:, :, -1]],
        axis=1,
    )
    if np.median(border) > BACKGROUND_WARN_FRACTION * data.max():
        logger.warning(
            "Image borders are not near zero; center of mass alignment assumes zero "
            "background. Consider TomoStack.subtract_background().",
        )
    mass = data.sum(axis=(1, 2))
    if mass.std() > MASS_WARN_VARIATION * abs(mass.mean()):
        logger.warning(
            "Total intensity varies by %.0f%% across projections; center of mass "
            "alignment assumes the object stays in the field of view and that "
            "intensity is proportional to mass (use TomoStack.to_absorbance() for "
            "bright-field data).",
            100 * mass.std() / abs(mass.mean()),
        )


def get_best_slices(stack: "TomoStack", nslices: int) -> np.ndarray:
    """
    Get best nslices for center of mass analysis.

    Slices which have the highest ratio of total mass to mass variance
    and their location are returned.

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
    total_mass = stack.data.sum((0, 1))
    mass_std = stack.data.sum(1).std(0)
    mass_std[mass_std == 0] = 1e-5
    mass_ratio = total_mass / mass_std
    best_slice_locations = mass_ratio.argsort()[::-1][0:nslices]
    return best_slice_locations


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
    sinos = stack.data[:, :, slices]
    y_coordinates = np.arange(sinos.shape[1]) - (sinos.shape[1] - 1) / 2
    total_mass = sinos.sum(1)
    coms = np.sum(np.transpose(sinos, [0, 2, 1]) * y_coordinates, 2) / total_mass
    return coms


def _to_matrices(vectors: np.ndarray) -> np.ndarray:
    """Convert ``(..., 6)`` transform vectors to ``(..., 3, 3)`` matrices."""
    vectors = np.asarray(vectors, dtype=float)
    matrices = np.zeros((*vectors.shape[:-1], 3, 3))
    matrices[..., :2, :] = vectors.reshape(*vectors.shape[:-1], 2, 3)
    matrices[..., 2, 2] = 1.0
    return matrices


def _to_vectors(matrices: np.ndarray) -> np.ndarray:
    """Convert ``(..., 3, 3)`` matrices to ``(..., 6)`` transform vectors."""
    return matrices[..., :2, :].reshape(*matrices.shape[:-2], 6)


def translation_matrices(shifts: np.ndarray) -> np.ndarray:
    """
    Build forward transform matrices for per-projection translations.

    Parameters
    ----------
    shifts
        Array of shape ``(n, 2)`` of ``(dy, dx)`` content displacements in pixels.

    Returns
    -------
    matrices : :py:class:`~numpy.ndarray`
        Array of shape ``(n, 3, 3)``.

    Group
    -----
    align
    """
    shifts = np.asarray(shifts, dtype=float).reshape(-1, 2)
    matrices = np.tile(np.eye(3), (len(shifts), 1, 1))
    matrices[:, :2, 2] = shifts
    return matrices


def rigid_matrix(
    shape: tuple[int, int],
    angle: float = 0.0,
    yshift: float = 0.0,
    xshift: float = 0.0,
) -> np.ndarray:
    """
    Build the forward transform for a rotation about the image center and a shift.

    Parameters
    ----------
    shape
        ``(ny, nx)`` shape of the images.
    angle
        Rotation angle in degrees, applied about the pixel center of the image.
    yshift, xshift
        Content displacement in pixels, applied after the rotation. Positive values
        move content toward larger indices.

    Returns
    -------
    matrix : :py:class:`~numpy.ndarray`
        Array of shape ``(3, 3)`` acting on ``(y, x, 1)`` pixel coordinates.

    Group
    -----
    align
    """
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    theta = np.deg2rad(angle)
    rotation = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
    )
    to_center = np.array([[1, 0, cy], [0, 1, cx], [0, 0, 1]])
    from_center = np.array([[1, 0, -cy], [0, 1, -cx], [0, 0, 1]])
    shift = np.array([[1, 0, yshift], [0, 1, xshift], [0, 0, 1]])
    return shift @ to_center @ rotation @ from_center


def compose_transforms(stack: "TomoStack", matrices: np.ndarray):
    """
    Compose ``matrices`` after the existing transforms of ``stack``, in place.

    Parameters
    ----------
    stack
        Stack whose :py:attr:`~etspy.base.TomoStack.transforms` are updated.
    matrices
        A ``(3, 3)`` matrix applied to every projection, or ``(n, 3, 3)`` matrices
        for each projection.

    Group
    -----
    align
    """
    current = _to_matrices(stack.transforms.data)
    stack.transforms.data = _to_vectors(matrices @ current)


def _resample(data, matrices: np.ndarray, output_shape, order: int, cuda: bool):
    """Resample each image of ``data`` with its forward transform."""
    xp = cp if cuda else np
    affine = affine_transform_gpu if cuda else ndimage.affine_transform
    out = np.empty((data.shape[0], *output_shape), dtype=data.dtype)
    for i in range(data.shape[0]):
        inverse = np.linalg.inv(matrices[i])
        image = affine(
            xp.asarray(data[i]),
            xp.asarray(inverse[:2, :2]),
            offset=xp.asarray(inverse[:2, 2]),
            output_shape=tuple(output_shape),
            order=order,
            mode="nearest",
        )
        out[i] = cp.asnumpy(image) if cuda else image
    return out


def apply_transforms(
    reference: "TomoStack",
    source: "TomoStack",
    shift_type: Literal["fourier", "interp"] = "fourier",
    cuda: bool = False,
) -> "TomoStack":
    """
    Apply the composed alignment of one stack to another in a single resampling.

    Parameters
    ----------
    reference
        Aligned stack whose :py:attr:`~etspy.base.TomoStack.transforms` are
        applied.
    source
        Stack to align. It must have the same number of projections as
        ``reference``, and the same image size as the data ``reference`` was
        aligned from.
    shift_type
        If every transform is a pure translation, ``"fourier"`` applies it with a
        Fourier shift. Otherwise, or with ``"interp"``, cubic spline
        interpolation is used.
    cuda
        Whether to use CuPy for the resampling.

    Returns
    -------
    out : TomoStack
        Aligned copy of ``source``, with the same image size as ``reference``.

    Group
    -----
    align
    """
    if cuda and not has_cupy:
        logger.warning("CUDA requested but CuPy is unavailable; using CPU")
        cuda = False
    if source.data.shape[0] != reference.data.shape[0]:
        msg = (
            f"Number of projections in the stack to align ({source.data.shape[0]}) "
            f"does not match the reference ({reference.data.shape[0]})"
        )
        raise ValueError(msg)

    matrices = _to_matrices(reference.transforms.data).reshape(-1, 3, 3)
    output_shape = reference.data.shape[-2:]
    data = source.data
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32)

    translation_only = np.allclose(matrices[:, :2, :2], np.eye(2))
    fits = all(o <= n for o, n in zip(output_shape, data.shape[-2:], strict=True))
    if shift_type == "fourier" and translation_only and fits:
        xp = cp if cuda else np
        shifted = _fourier_shift(xp.asarray(data), xp.asarray(matrices[:, :2, 2]), xp)
        shifted = shifted[:, : output_shape[0], : output_shape[1]]
        data = cp.asnumpy(shifted) if cuda else shifted
    else:
        data = _resample(data, matrices, output_shape, order=3, cuda=cuda)

    out = source.deepcopy()
    out.data = data
    out.get_dimensions_from_data()
    compose_transforms(out, matrices)
    out.shifts.data = out.shifts.data + reference.shifts.data

    ref_meta = cast("Dtb", reference.metadata.Tomography)
    out_meta = cast("Dtb", out.metadata.Tomography)
    for key in ("xshift", "yshift", "tiltaxis"):
        out_meta.set_item(key, out_meta.get_item(key, 0) + ref_meta.get_item(key, 0))
    cropped = out_meta.get_item("cropped", default=False)
    out_meta.set_item("cropped", bool(cropped or ref_meta.get_item("cropped")))
    for out_axis, ref_axis in zip(
        out.axes_manager.signal_axes,
        reference.axes_manager.signal_axes,
        strict=True,
    ):
        if out_axis.scale == ref_axis.scale:
            out_axis.offset = ref_axis.offset
    return out


def convert_legacy_alignment(stack: "TomoStack"):
    """
    Rebuild transforms for a stack saved before transforms were recorded, in place.

    Older versions applied registration shifts, then an optional crop, then a
    rotation and shift, and stored a ``yshift`` with the opposite sign convention.

    Group
    -----
    align
    """
    meta = cast("Dtb", stack.metadata.Tomography)
    shifts = stack.shifts.data.reshape(-1, 2)
    matrices = translation_matrices(shifts)
    if meta.get_item("cropped", default=False):
        # older versions cropped x using the y-shift column and vice versa
        crop = [-np.ceil(shifts[:, 1].max()), -np.ceil(shifts[:, 0].max())]
        matrices = translation_matrices(np.tile(crop, (len(shifts), 1))) @ matrices
    yshift = -float(meta.get_item("yshift", 0))
    matrices = (
        rigid_matrix(
            stack.data.shape[-2:],
            float(meta.get_item("tiltaxis", 0)),
            yshift,
            float(meta.get_item("xshift", 0)),
        )
        @ matrices
    )
    stack.transforms = _to_vectors(matrices).reshape(stack.transforms.data.shape)
    meta.set_item("yshift", yshift)
    logger.info("Rebuilt alignment transforms from metadata of an older file")


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
    GPU-acceleration via CuPy (or on the CPU, with a warning, if CuPy is unavailable).
    Regions uncovered by a shift are filled by repeating the nearest edge pixels. The
    shifts are stored in ``shifted.metadata.Tomography.shifts``.

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
        logger.warning("CUDA requested but CuPy is unavailable; applying shifts on CPU")
        cuda = False

    if isinstance(shifts, BaseSignal):
        shifts = shifts.data
    shifts = cast("np.ndarray", shifts)

    if len(shifts) != stack.data.shape[0]:
        msg = (
            f"Number of shifts ({len(shifts)}) is not consistent "
            f"with number of images in the stack ({stack.data.shape[0]})"
        )
        raise ValueError(msg)

    shifts = np.asarray(shifts, dtype=float)
    xp = cp if cuda else np

    if method.lower() == "interp":
        data = _interp_shift(xp.array(stack.data), shifts, kwargs.pop("order", 3), cuda)
    elif method.lower() == "fourier":
        data = _fourier_shift(xp.asarray(stack.data), xp.asarray(shifts), xp)
    else:
        msg = f"Invalid shift application method {method}."
        raise ValueError(msg)

    shifted = stack.copy_with_data(cp.asnumpy(data) if cuda else data)
    shifted.shifts.data = shifted.shifts.data + shifts
    compose_transforms(shifted, translation_matrices(shifts))
    return shifted


def _interp_shift(data, shifts: np.ndarray, order: int, cuda: bool):
    """Shift each image in ``data`` by ``(dy, dx)`` using spline interpolation."""
    shift_func = shift_gpu if cuda else ndimage.shift
    for i in range(data.shape[0]):
        data[i, :, :] = shift_func(
            data[i, :, :],
            shift=[float(shifts[i, 0]), float(shifts[i, 1])],
            order=order,
            mode="nearest",
        )
    return data


def _fourier_shift(data, shifts, xp):
    """Shift each image in ``data`` by ``(dy, dx)`` with a Fourier phase ramp."""
    minvals = xp.min(data, axis=(1, 2), keepdims=True)
    ntilts, ny, nx = data.shape
    dtype = data.dtype
    float_dtype = np.float64 if dtype == np.float64 else np.float32
    complex_dtype = np.complex128 if dtype == np.float64 else np.complex64
    data = data.astype(float_dtype)

    # Ensure padding covers shift distance on each side (2 * max_shift)
    max_y_shift, max_x_shift = (
        float(xp.abs(shifts[:, 0]).max()),
        float(xp.abs(shifts[:, 1]).max()),
    )
    y_pad_min, x_pad_min = (
        int(np.ceil(ny + 2 * max_y_shift)),
        int(np.ceil(nx + 2 * max_x_shift)),
    )

    ny_pad, nx_pad = (
        next_fast_len(y_pad_min, real=True),
        next_fast_len(x_pad_min, real=True),
    )

    y_pad_before, x_pad_before = (ny_pad - ny) // 2, (nx_pad - nx) // 2
    y_pad_after, x_pad_after = (
        ny_pad - ny - y_pad_before,
        nx_pad - nx - x_pad_before,
    )

    # Pad original data
    padded_data = xp.pad(
        data,
        ((0, 0), (y_pad_before, y_pad_after), (x_pad_before, x_pad_after)),
        mode="edge",
    )

    # 2. Use Real FFT (rfft2) for 2x speedup and 50% memory savings
    data_fft = xp.fft.rfft2(padded_data, axes=(1, 2))

    # 3. Construct frequency grids with explicit data types
    v = xp.fft.fftfreq(ny_pad, d=1.0).astype(float_dtype).reshape(1, ny_pad, 1)
    u = (
        xp.fft.rfftfreq(nx_pad, d=1.0)
        .astype(float_dtype)
        .reshape(1, 1, nx_pad // 2 + 1)
    )

    sy, sx = (
        shifts[:, 0].astype(float_dtype).reshape(ntilts, 1, 1),
        shifts[:, 1].astype(float_dtype).reshape(ntilts, 1, 1),
    )

    # Phase ramp kernel calculation
    pi = float_dtype(np.pi)
    phi = -2j * pi * (v * sy + u * sx)
    kernel = xp.exp(phi).astype(complex_dtype)

    # 4. In-place frequency multiplication to conserve memory
    data_fft *= kernel

    # 5. Inverse Real FFT and unpad
    data = xp.fft.irfft2(data_fft, s=(ny_pad, nx_pad), axes=(1, 2))

    # Immutable slice definitions (prevents mutation bugs)
    slice_y, slice_x = (
        slice(y_pad_before, -y_pad_after if y_pad_after > 0 else None),
        slice(x_pad_before, -x_pad_after if x_pad_after > 0 else None),
    )
    data = xp.clip(data[:, slice_y, slice_x], minvals, None)
    return data


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
        self.upsample_factor = kwargs.get("upsample_factor", 10)

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
            ntilts = self.stack.data.shape[0]
            shifts = np.zeros((ntilts, 2))

            def image_fft(i):
                return fft2(self.stack.data[i], workers=-1)

            with tqdm.tqdm(
                total=ntilts - 1,
                desc="Calculating shifts",
                disable=not self.show_progressbar,
            ) as pbar:
                # walk outward from the start image, reusing each image's FFT as
                # the reference for the next pair
                for step in (-1, 1):
                    ref_fft = image_fft(self.start)
                    stop = -1 if step < 0 else ntilts
                    for i in range(self.start + step, stop, step):
                        mov_fft = image_fft(i)
                        shift = pcc(
                            ref_fft,
                            mov_fft,
                            upsample_factor=self.upsample_factor,
                            space="fourier",
                        )[0]
                        shifts[i] = shifts[i - step] + shift
                        ref_fft = mov_fft
                        pbar.update(1)
        return shifts

    def _cupy_calculate_shifts(self):
        """Calculate shifts of stack using CUDA-implementation of phase correlation."""
        stack_cp = cp.array(self.stack.data)
        shifts = cp.zeros([stack_cp.shape[0], 2])
        shape = stack_cp.shape[1:]
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
        _check_mass_assumptions(self.stack)
        shifts = np.zeros([self.stack.data.shape[0], 2])
        # _calculate_shifts_conservation_of_mass: x-shifts (parallel to tilt axis)
        # _calculate_shifts_com: y-shifts (perpendicular to tilt axis)
        shifts[:, 1] = self._calculate_shifts_conservation_of_mass()
        # x misalignment moves mass between the slices tracked for y, so remove it
        x_aligned = apply_shifts(self.stack, shifts, cuda=self.use_cuda)
        shifts[:, 0] = self._calculate_shifts_com(x_aligned)
        return shifts

    def _calculate_shifts_conservation_of_mass(
        self,
    ) -> np.ndarray:
        """
        Calculate shifts parallel to the tilt axis using conservation of mass.

        Slices which have the highest ratio of total mass to mass variance
        and their location are returned.

        Parameters
        ----------
        stack
            Tilt series to be aligned.
        xrange
            The range for performing alignment
        p
            Padding element

        Returns
        -------
        xshifts : :py:class:`~numpy.ndarray`
            Calculated shifts parallel to tilt axis.

        Group
        -----
        align
        """
        logger.info("Refinining X-shifts using conservation of mass method")
        ntilts, _, nx = self.stack.data.shape

        p = self.p
        if self.xrange is None:
            xrange = (round(nx / 5), round(4 / 5 * nx))
            # the padded window must fit inside the image
            p = min(p, xrange[0] - 1, nx - xrange[1])
            if p < 1:
                msg = f"Image is too narrow ({nx} pixels) for center of mass alignment"
                raise ValueError(msg)
            if p < self.p:
                logger.warning("Padding reduced from %i to %i to fit image", self.p, p)
        else:
            xrange = (round(self.xrange[0]) + p, round(self.xrange[1]) - p)

        xshifts = np.zeros([ntilts, 1])
        total_mass = np.zeros([ntilts, xrange[1] - xrange[0] + 2 * p + 1])

        for i in range(ntilts):
            total_mass[i, :] = np.sum(
                self.stack.data[i, :, xrange[0] - p - 1 : xrange[1] + p],
                0,
            )

        mean_mass = np.mean(total_mass[:, p:-p], 0)

        for i in range(ntilts):
            s = 0
            for j in range(-p, p):
                resid = np.linalg.norm(
                    mean_mass - total_mass[i, p + j : -p + j],
                )
                if resid < s or j == -p:
                    s = resid
                    xshifts[i] = -j
        return xshifts[:, 0]

    def _calculate_shifts_com(
        self,
        stack: "TomoStack | None" = None,
    ) -> np.ndarray:
        """
        Align stack using a center of mass method.

        The center of mass of the best slices is fit to the sinusoidal path expected
        for rigid rotation. Shifts perpendicular to the tilt axis are the fit
        residuals, averaged over all slices.

        Parameters
        ----------
        stack
            The image series to analyze. If ``None``, ``self.stack`` is used.

        Returns
        -------
        yshifts : :py:class:`~numpy.ndarray`
            The Y-shifts to be applied to each image

        Group
        -----
        align
        """
        logger.info("Refinining Y-shifts using center of mass method")
        stack = self.stack if stack is None else stack
        slices = get_best_slices(stack, self.nslices)

        angles = stack.tilts.data.squeeze()
        thetas = np.pi * cast("np.ndarray", angles) / 180

        coms = get_coms(stack, slices)
        gam = np.array([np.cos(thetas), np.sin(thetas)]).T
        residuals = coms - gam @ np.linalg.lstsq(gam, coms, rcond=None)[0]

        yshifts = -residuals.mean(axis=1)
        return yshifts


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
        Not used. Each projection is shifted so that its center of mass lies at
        the image center, which also centers the tilt axis. Retained for
        compatibility.
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
        self.com_ref_index = kwargs.get("com_ref_index")
        self.cl_ref_index = kwargs.get("cl_ref_index")
        self.cl_resolution = kwargs.get("cl_resolution", 0.05)
        self.cl_div_factor = kwargs.get("cl_div_factor", 8)

    def calculate_shifts(self) -> np.ndarray:
        """Calculate shifts using combined center of mass and common line methods."""
        logger.info(
            "Performing stack registration using combined "
            "center of mass and common line methods",
        )
        _check_mass_assumptions(self.stack)
        if self.com_ref_index is None:
            self.com_ref_index = self.start
        if self.cl_ref_index is None:
            self.cl_ref_index = self.start

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

        logger.info("Common line reference slice: %s", self.cl_ref_index)
        yshifts = self._calc_yshifts()
        xshifts = self.calc_shifts_cl(
            self.cl_ref_index,
            self.cl_resolution,
            self.cl_div_factor,
        )
        shifts = np.stack([yshifts, xshifts], axis=1)
        return shifts

    def _calc_yshifts(self):
        """Calculate shifts moving each center of mass to the image center."""
        ntilts, ny, _ = self.stack.data.shape
        coms = np.zeros(ntilts)

        for i in tqdm.tqdm(range(ntilts), disable=not self.show_progressbar):
            coms[i], _ = ndimage.center_of_mass(self.stack.data[i, :, :])
        return (ny - 1) / 2 - coms

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

    @abstractmethod
    def align_tilt_axis(self) -> "TomoStack":
        """Align the tilt axis using the selected strategy."""


class TiltCOMAligner(TiltAligner):
    """Tilt alignment class for center of mass method.

    Perform tilt axis alignment using center of mass (CoM) tracking.

    Compares path of specimen to the path expected for an ideal cylinder

    Parameters
    ----------
    slices
        Locations at which to perform the Center of Mass analysis. If not
        provided, an appropriate list of slices will be automatically determined.
    nslices
        Nubmer of slices to use for the analysis (only used if the ``slices``
        parameter is not specified). If ``None``, a value of 10% of the x-axis
        size will be used, clamped to the range [3, 50].

    Returns
    -------
    out : TomoStack
        Copy of the input stack after rotation and translation to center and
        make the tilt axis parallel to x

    Group
    -----
    align
    """

    def __init__(self, stack, **kwargs):
        super().__init__(stack, **kwargs)
        self.slices = kwargs.get("slices")
        self.nslices = kwargs.get("nslices")

    def align_tilt_axis(self) -> "TomoStack":
        """Align tilt axis with center of mass tracking."""

        def com_motion(theta, r, x0, z0):
            return r - x0 * np.cos(theta) - z0 * np.sin(theta)

        def fit_line(x, m, b):
            return m * x + b

        nx = self.stack.data.shape[2]
        nx_threshold = 3
        _check_mass_assumptions(self.stack)

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

        # Determine the best slice locations for the analysis
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

            self.slices = get_best_slices(self.stack, self.nslices)
            logger.info("Performing alignments using best %s slices", self.nslices)

        self.slices = np.sort(self.slices)

        coms = get_coms(self.stack, self.slices)
        thetas = (
            np.pi * self.stack.tilts.data.squeeze() / 180.0
        )  # remove length 1 dimension

        r, x0, z0 = (
            np.zeros(len(self.slices)),
            np.zeros(len(self.slices)),
            np.zeros(
                len(self.slices),
            ),
        )

        for idx, _ in enumerate(self.slices):
            r[idx], x0[idx], z0[idx] = optimize.curve_fit(
                com_motion,
                xdata=thetas,
                ydata=coms[:, idx],
                p0=[0, 0, 0],
            )[0]
        slope, intercept = optimize.curve_fit(
            fit_line,
            xdata=r,
            ydata=self.slices,
            p0=[0, 0],
        )[0]
        # content shift that moves the tilt axis to the image center
        tilt_shift = (intercept - (nx - 1) / 2) / slope
        tilt_rotation = -(180 * np.arctan(1 / slope) / np.pi)

        final = cast(
            "TomoStack",
            self.stack.trans_stack(
                yshift=tilt_shift,
                angle=tilt_rotation,
            ),
        )

        logger.info("Calculated tilt-axis shift %.2f", tilt_shift)
        logger.info("Calculated tilt-axis rotation %.2f", tilt_rotation)

        return final


class TiltMaxImageAligner(TiltAligner):
    """Tilt alignment class for maximum image method.

    Perform automated determination of the tilt axis of a TomoStack.

    The projected maximum image used to determine the tilt axis by a
    combination of Sobel filtering and Hough transform analysis.

    The rotation estimate relies on features tracing straight streaks perpendicular
    to the tilt axis in the maximum image, so its accuracy depends on the specimen.
    In tests it was within 0.1 degrees for an extended catalyst model but off by up
    to several degrees for sparse objects or objects elongated along the tilt axis.
    The ``"CoM"`` method of :py:meth:`~etspy.base.TomoStack.tilt_align` is
    generally more reliable; :py:meth:`~etspy.base.TomoStack.test_align` can be
    used to check the result.

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
        """Align tilt axis with center of mass tracking."""
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
        rotation_angle = np.degrees(np.mean(angles))

        if self.plot_results:
            _, ax = plt.subplots(1)
            ax.imshow(image, cmap="gray")

            for i in range(len(angles)):
                (x0, y0) = dists[i] * np.array([np.cos(angles[i]), np.sin(angles[i])])
                ax.axline((x0, y0), slope=np.tan(angles[i] + np.pi / 2))

            plt.tight_layout()

        ali = cast("TomoStack", self.stack.trans_stack(angle=-rotation_angle))

        logger.info("Calculated tilt-axis rotation %.2f", -rotation_angle)

        if self.also_shift:
            idx = ali.data.shape[2] // 2
            shifts = np.arange(-self.shift_limit, self.shift_limit + 1, 1)
            nshifts = shifts.shape[0]
            shifted = ali.isig[0:nshifts, :].deepcopy()
            for i in range(nshifts):
                shifted.data[:, :, i] = np.roll(
                    ali.isig[idx : idx + 1, :].data.squeeze(),
                    int(shifts[i]),
                    axis=1,
                )
            shifted_rec = shifted.reconstruct(
                "SIRT",
                100,
                constrain=True,
                show_progressbar=False,
            )
            image_sum = cast("BaseSignal", shifted_rec.sum(axis=(1, 2))).data
            best = int(image_sum.argmin())
            tilt_shift = float(shifts[best])
            if 0 < best < nshifts - 1:
                # parabolic interpolation of the minimum for subpixel precision
                left, center, right = image_sum[best - 1 : best + 2]
                curvature = left - 2 * center + right
                if curvature > 0:
                    tilt_shift += 0.5 * (left - right) / curvature
            ali = cast("TomoStack", ali.trans_stack(yshift=tilt_shift))

            logger.info("Calculated tilt-axis shift %.2f", tilt_shift)

        return ali


def align_to_other(
    stack: "TomoStack",
    other: "TomoStack",
    shift_type: Literal["fourier", "interp"] = "fourier",
    cuda: bool = False,
) -> "TomoStack":
    """
    Spatially register a TomoStack using the alignment of another.

    The composed transforms of ``stack`` (registration, cropping and tilt-axis
    alignment, in the order they were applied) are applied to ``other`` in a single
    resampling step. See :py:func:`~etspy.align.apply_transforms`.

    Parameters
    ----------
    stack
        TomoStack which was previously aligned
    other
        TomoStack to be aligned. Must be the same size as the data ``stack`` was
        aligned from
    shift_type
        If the alignment is a pure translation, ``'fourier'`` applies it with a
        Fourier shift; otherwise, or with ``'interp'``, cubic spline interpolation
        is used.
    cuda
        Whether to use CuPy for the resampling.

    Returns
    -------
    out : TomoStack
        Aligned copy of other TomoStack

    Group
    -----
    align
    """
    out = apply_transforms(stack, other, shift_type, cuda)
    tomo_meta = cast("Dtb", stack.metadata.Tomography)
    logger.info("TomoStack alignment applied")
    logger.info("X-shift: %.1f", tomo_meta.get_item("xshift", 0))
    logger.info("Y-shift: %.1f", tomo_meta.get_item("yshift", 0))
    logger.info("Rotation: %.1f", tomo_meta.get_item("tiltaxis", 0))
    return out


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
        Aligned copy of other TomoStack

    Group
    -----
    align
    """
    cropped = stack.deepcopy()
    ny, nx = stack.data.shape[-2:]
    shifts = stack.shifts.data.reshape(-1, 2)

    # a positive shift uncovers the start of an axis, a negative one its end
    def valid_range(axis_shifts, size):
        start = int(np.ceil(max(axis_shifts.max(), 0)))
        stop = size + int(np.floor(min(axis_shifts.min(), 0)))
        return start, stop

    y_min, y_max = valid_range(shifts[:, 0], ny)
    x_min, x_max = valid_range(shifts[:, 1], nx)
    cropped = stack.isig[x_min:x_max, y_min:y_max].deepcopy()
    compose_transforms(cropped, translation_matrices([[-y_min, -x_min]])[0])
    cropped.metadata.set_item("Tomography.cropped", value=True)
    return cropped
