"""Projection matching alignment."""

import logging
from typing import TYPE_CHECKING, Dict, Literal, Optional

import numpy as np
from hyperspy.signals import Signal1D, Signal2D
from scipy.fft import fft, fftfreq, fftshift, ifft
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve

if TYPE_CHECKING:
    from etspy.base import TomoStack

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DIM_2D = 2


# TODO: Drop internal shift functions and use the built-in ETSpy options
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
        cuda: bool = False,
        params: Optional[Dict] = None,
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
            "minstep"

        """
        self.stack_orig = stack.deepcopy()
        self.sino = stack.data.squeeze()
        self.tilts = stack.tilts.data.squeeze()
        if len(self.sino.shape) == DIM_2D:
            self.nangles, self.ny = self.sino.shape
            self.nx = None
            self.total_shifts = np.zeros(self.nangles)
        else:
            self.nangles, self.ny, self.nx = self.sino.shape
            self.total_shifts = np.zeros([self.nangles, 2])
        self.cuda = cuda
        if params is None:
            params = {}
        self.levels = params.get("levels", [8, 4, 2, 1])
        self.iterations = params.get("iterations", 50)
        self.recon_algorithm = params.get("recon_algorithm", "FBP")
        if self.recon_algorithm.lower() == "fbp":
            self.recon_iters = None
        else:
            self.recon_iters = params.get("recon_iterations", 20)
        self.relax = params.get("relax", 0.1)
        self.minstep = params.get("minstep", 0.01)

    def calculate_shifts(
        self,
        y_only: bool = False,
        show_progressbar=False,
    ):
        """Calculate shifts.

        Parameters
        ----------
        y_only : bool, optional
            If True, calculate alignments only in the Y direction (i.e. perpendicular
            to the tilt axis), by default False
        downsample_method : str, optional
            Method to use for downsampling.  If "interp", downsampling is performed
            using Hyperspy which employs an interpolation method.  The resulting data
            will be smaller than the input by a factor of the current level.  If "blur",
            the data is blurred by convolution and the result will have the same size
            as the input, by default "interp"

        """
        sino_shifted = self.sino.copy()
        return sino_shifted


def blur_convolve(
    sino: np.ndarray,
    factor: int,
) -> np.ndarray:
    """Convolve a sinogram or image stack to blur by a given factor.

    Parameters
    ----------
    sino : np.ndarray
        Sinogram or stack to downsample
    factor : int
        Factor by which to downsample

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
    else:
        _, ny, nx = sino.shape
        sino = convolve(sino, kernel.reshape(1, shape, 1), mode="same")
        sino = convolve(sino, kernel.reshape(1, 1, shape), mode="same")

        corr = np.ones([1, ny, nx])
        corr = convolve(corr, kernel.reshape(1, shape, 1), mode="same")
        corr = convolve(corr, kernel.reshape(1, 1, shape), mode="same")
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
        sino_fft = fft(sino)

    freq = fftfreq(ny)
    sigma = 256 / (ny) * sigma

    if sigma == 0:
        high_pass_filter = 2j * np.pi * freq
    else:
        high_pass_filter = 1 - np.exp(-0.5 * (freq / sigma) ** 2)

    sino_filtered = sino_fft * high_pass_filter

    if apply_fft:
        sino_filtered = np.fft.ifft(sino_filtered)
    if is_real:
        sino_filtered = np.real(sino_filtered)
    return sino_filtered
