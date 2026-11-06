"""Projection matching alignment."""

import logging
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
    from numpy.typing import NDArray

    from etspy.base import TomoStack

logger = logging.getLogger(__name__)

DIM_2D = 2

# TODO: Enable alignment along X axis


class ProjMatch:
    """Class to perform alignment via projection matching method.

    Projection images are aligned by minimizing the difference between the input
    sinogram(s) and the forward projections of the current reconstruction.  Python
    implementation of Matlab code reported in:

    M. Odstrčil, Mirko Holler, Jörg Raabe, and Manuel Guizar-Sicairos. Alignment methods
    for nanotomography with deep subpixel accuracy, Optics Express Vol. 27 (2019)
    pp. 36637-36652.
    https://doi.org/10.1364/OE.27.036637

    A previous Python implementation based on the same Matlab source code and with
    additional functionality is also available:
    https://github.com/jtschwar/projection_refinement/
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
        cuda : bool
            If True, use CUDA-acceleration. Default is False.
        params : Optional[Dict], optional
            Dictionary of parameters for alignment. Acceptable keys are:
            "levels", "iterations", "recon_algorithm", "recon_iterations", "relax", and
            "minstep". "recon_iterations" is the number of iterations for
            the reconstruction algorithm (default: 20 for SIRT and 5 for SART).

        """
        self.stack = stack
        if self.stack.data.squeeze().ndim == DIM_2D:
            self.sino = stack.data.squeeze()
        elif type(nslice) is int:
            self.sino = stack.data[:, :, nslice]
        else:
            nslice = stack.data.shape[2] // 2
            self.sino = stack.data[:, :, nslice]
        self.tilts = stack.tilts.data.squeeze()
        self.thetas = np.deg2rad(self.tilts)
        # background (vacuum) level of each projection, used to fill the pixels
        # uncovered when the sinogram is shifted
        self.background = np.array(
            [background_mode(np.atleast_2d(im)) for im in stack.data],
        )

        if len(self.sino.shape) == DIM_2D:
            self.nangles, self.ny = self.sino.shape
            self.nx = None
            self.total_shifts = np.zeros(self.nangles)
        else:
            msg = "Alignment of 3D stacks is not yet implemented"
            raise NotImplementedError(msg)

        self.cuda = cuda

        if params is None:
            self.params = {}
        else:
            self.params = params
        self.levels = self.params.get("levels", [8, 4, 2, 1])
        self.iterations = self.params.get("iterations", 100)
        self.error: list[NDArray[Any]] = [np.empty(0) for _ in range(len(self.levels))]

        self.sino_update: list[NDArray | None] = [None] * len(self.levels)
        self.rec_update: list[NDArray | None] = [None] * len(self.levels)

        self.recon_algorithm = self.params.get("recon_algorithm", "FBP")

        default_passes = {"sirt": 20, "sart": 5}.get(self.recon_algorithm.lower())
        _, self.recon_iterations = astra_iterations(
            self.recon_algorithm,
            self.params.get("recon_iterations", default_passes),
            self.nangles,
        )  # ASTRA objects, created per downsampling level by update_geometries
        self.rec_id: int | None = None
        self.sino_id: int | None = None
        self.proj_id: int | None = None
        self.algorithm: int | None = None
        self.create_recon_algorithm(self.recon_algorithm, self.cuda)
        self.relax = self.params.get("relax", 0.1)
        self.minstep = self.params.get("minstep", 0.01)

    def calculate_shifts(
        self,
        show_progressbar=False,
    ):
        """Calculate shifts.

        Parameters
        ----------
        downsample_method : str, optional
            Method to use for downsampling.  If "interp", downsampling is performed
            using Hyperspy which employs an interpolation method.  The resulting data
            will be smaller than the input by a factor of the current level.  If "blur",
            the data is blurred by convolution and the result will have the same size
            as the input, by default "interp"

        """
        for idx, j in enumerate(self.levels):
            logger.info("Binning %i", j)
            sino_shifted = shift_sinogram(
                self.sino,
                self.total_shifts,
                self.background,
            )
            if j != 1:
                sino_rebin = blur_convolve(sino_shifted, j)
                sino_rebin = interpolate_ft(sino_rebin, j)
            else:
                sino_rebin = sino_shifted.copy()

            current_shifts = np.zeros(self.nangles)
            self.update_geometries(sino_rebin.shape[1])

            mass = 0

            for i in tqdm.tqdm(range(self.iterations), disable=not (show_progressbar)):
                current_sino = shift_sinogram(
                    sino_rebin,
                    current_shifts,
                    self.background,
                )
                rec = self.reconstruct(current_sino)

                reproj = self.forward_project(rec)

                resid = reproj - current_sino
                resid = high_pass_filter(resid, 5)

                grad_y = sino_gradient(reproj)
                grad_y = high_pass_filter(grad_y, 5)
                yshifts = -np.sum(grad_y * resid, axis=1) / np.sum(grad_y**2, axis=1)
                yshifts = (
                    np.minimum(0.5, np.abs(yshifts)) * np.sign(yshifts) * self.relax
                )
                yshifts = yshifts - np.median(yshifts)
                max_step = np.minimum(np.quantile(np.abs(yshifts), 0.99), 0.5)
                yshifts = np.minimum(max_step, np.abs(yshifts)) * np.sign(yshifts)
                current_shifts += yshifts

                if i == 0:
                    mass = np.median(np.mean(np.abs(current_sino), axis=1))
                    self.sino_update[idx] = current_sino[np.newaxis, :, :]
                    self.rec_update[idx] = rec[np.newaxis, :, :]
                else:
                    self.sino_update[idx] = np.concatenate(
                        [
                            cast("NDArray", self.sino_update[idx]),
                            current_sino[np.newaxis, :, :],
                        ],
                        axis=0,
                    )
                    self.rec_update[idx] = np.concatenate(
                        [
                            cast("NDArray", self.rec_update[idx]),
                            rec[np.newaxis, :, :],
                        ],
                        axis=0,
                    )
                self.error[idx] = np.append(
                    self.error[idx],
                    np.linalg.norm(resid) / mass,
                )
                max_update = np.max(np.quantile(np.abs(yshifts), 0.995))
                if max_update * j < self.minstep:
                    logger.info("Converged after %i iterations", i)
                    break

            current_shifts = current_shifts - np.median(current_shifts)
            self.total_shifts = self.total_shifts + current_shifts * j
            if idx == 0:
                self.shift_update = self.total_shifts[np.newaxis, :]
            else:
                self.shift_update = np.concatenate(
                    [self.shift_update, self.total_shifts[np.newaxis, :]],
                    axis=0,
                )
        self.delete_astra_objects()

        for i in range(len(self.levels)):
            self.sino_update[i] = Signal2D(self.sino_update[i])
            self.rec_update[i] = Signal2D(self.rec_update[i])
            self.error[i] = Signal1D(self.error[i])
        self.shift_update = Signal1D(self.shift_update)

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
        """Update projection and reconstruction parameters for current downsampling.

        Parameters
        ----------
        current_ny
            Number of pixels in the Y-dimension in the current downsampled sinogram.

        """
        self.delete_astra_objects()
        proj_geom = astra.create_proj_geom("parallel", 1.0, current_ny, self.thetas)
        vol_geom = astra.create_vol_geom((current_ny, current_ny))
        self.rec_id = astra.data2d.create("-vol", vol_geom)
        self.sino_id = astra.data2d.create(
            "-sino",
            proj_geom,
            np.zeros([self.nangles, current_ny]),
        )
        self.proj_id = create_projector(proj_geom, vol_geom, cuda=self.cuda)
        self.recon_config["ReconstructionDataId"] = self.rec_id
        self.recon_config["ProjectorId"] = self.proj_id
        self.recon_config["ProjectionDataId"] = self.sino_id
        self.recon_config["ReconstructionDataId"] = self.rec_id
        self.algorithm = astra.algorithm.create(self.recon_config)

    def delete_astra_objects(self):
        """Delete the ASTRA objects created for the current downsampling level."""
        if self.algorithm is not None:
            astra.algorithm.delete(self.algorithm)
            self.algorithm = None
        if self.rec_id is not None:
            astra.data2d.delete(self.rec_id)
            self.rec_id = None
        if self.sino_id is not None:
            astra.data2d.delete(self.sino_id)
            self.sino_id = None
        if self.proj_id is not None:
            astra.projector.delete(self.proj_id)
            self.proj_id = None

    def forward_project(
        self,
        current_rec: np.ndarray,
    ) -> np.ndarray:
        """Forward project the current reconstruction."""
        sino_id, proj = astra.create_sino(current_rec, self.proj_id)
        astra.data2d.delete(sino_id)
        return proj

    def reconstruct(
        self,
        current_sinogram: np.ndarray,
    ) -> np.ndarray:
        """Run reconstruction algorithm.

        Parameters
        ----------
        current_sinogram
            Sinogram to reconstruct.

        """
        astra.data2d.store(self.sino_id, current_sinogram)
        astra.algorithm.run(self.algorithm, self.recon_iterations)
        rec = astra.data2d.get(self.rec_id)
        return rec

    def apply_shifts(
        self,
    ) -> "TomoStack":
        """
        Apply calculated shifts to input stack.

        Returns
        -------
        shifted : TomoStack
            Copy of input stack after shifts are applied
        """
        if len(self.total_shifts) != self.stack.data.shape[0]:
            msg = (
                f"Number of shifts ({len(self.total_shifts)}) is not consistent "
                f"with number of images in the stack ({self.nangles})"
            )
            raise ValueError(msg)
        shifts = np.stack([self.total_shifts, np.zeros(self.nangles)], axis=1)
        return _apply_shifts(self.stack, shifts, "fourier")


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


def blur_convolve(
    sino: np.ndarray,
    factor: int,
) -> np.ndarray:
    """Convolve a sinogram or image stack to blur by a given factor.

    Parameters
    ----------
    sino : np.ndarray
        Sinogram or stack to blur
    factor : int
        Factor by which to blur

    Returns
    -------
    np.ndarray
        Blurred version of input data

    """
    grid = np.arange(-np.ceil(2 * factor), np.ceil(2 * factor) + 1) / factor
    kernel = np.exp(-(grid**2))
    kernel /= np.sum(kernel)
    shape = kernel.shape[0]

    if len(sino.shape) == DIM_2D:
        _, ny = sino.shape
        sino = convolve(sino, kernel.reshape(1, shape), mode="same")

        corr = np.ones([1, ny])
        corr = convolve(corr, kernel.reshape(1, shape), mode="same")
        sino = sino / corr
    return sino


def blur_edges(
    sino: np.ndarray,
    window: int = 5,
    sigma: float = 5,
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

    Returns
    -------
    sino_filtered : np.ndarray
        Filtered version of input data

    """
    window = max(window, 3)
    filtered = gaussian_filter(sino, sigma, axes=(1,))
    filtered[:, window:-window] = sino[:, window:-window]
    return filtered


def high_pass_filter(
    sino: np.ndarray,
    sigma: float = 5,
) -> np.ndarray:
    """High pass filter a sinogram or sinogram-like array using Gaussian convolution.

    Parameters
    ----------
    sino : np.ndarray
        Sinogram or stack to filter
    sigma : int
        Standard deviation of the Gaussian filter which controls the degree of
        filtering. Default is 5.

    Returns
    -------
    sino_filtered : np.ndarray
        Filtered version of input data

    """
    filtered = sino - gaussian_filter(sino, sigma, axes=(1,))
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


def interpolate_ft(
    sino: np.ndarray,
    blur_factor: int,
) -> np.ndarray:
    """Interpolate the FT of sinogram after convolution to maintain cente of mass.

    Parameters
    ----------
    sino : np.ndarray
        Input sinogram
    blur_factor : int
        Factor used for blurring

    Returns
    -------
    sino_centered : np.ndarray
        Interpolated version of blurred sinogram

    """
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


def sino_gradient(
    sino: np.ndarray,
    blur_window: int = 5,
    blur_sigma: float = 5.0,
) -> np.ndarray:
    """Calculate the gradient of a sinogram along Y axis.

    Parameters
    ----------
    sino : np.ndarray
        Input sinogram
    smooth_window : int
        Window for blurring edges. Default is 5.
    blur_sigma : float
        Sigma for Gaussian blur. Default is 5.0.

    Returns
    -------
    sino_grad : np.ndarray
        Gradient of input sinogram along the Y-axis

    """
    _, ny = sino.shape
    sino = blur_edges(sino, blur_window, blur_sigma)
    x = 2j * np.pi * fftfreq(ny)

    sino_fft = cast("np.ndarray", fft(sino, axis=1))
    sino_fft = sino_fft * x[np.newaxis, :]

    sino_grad = ifft(sino_fft, axis=1)
    sino_grad = np.real(cast("np.ndarray", sino_grad))
    return sino_grad
