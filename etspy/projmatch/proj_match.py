"""Projection matching alignment."""

import logging
from typing import Dict, Literal, Optional

import numpy as np
from hyperspy.signals import Signal1D, Signal2D
from scipy.signal import convolve

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
    downsample_method: Literal[
        "interp",
        "blur",
    ] = "interp",
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


def downsample(
    data: np.ndarray,
    factor: int,
) -> np.ndarray:
    """Downsample sinogram or image stack using Hyperspy.

    Parameters
    ----------
    data : np.ndarray
        Sinogram or stack to downsample
    factor : int
        Factor by which to downsample

    Returns
    -------
    downsampled : np.ndarray
        Downsampled version of input data

    """
    if len(data.shape) == DIM_2D:
        downsampled = Signal1D(data).rebin(scale=[1, factor]).data
    else:
        downsampled = Signal2D(data).rebin(scale=[1, factor, factor]).data
    return downsampled


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
