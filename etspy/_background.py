"""Estimation of the background (vacuum) level of projection images."""

import numpy as np
from scipy import ndimage

# images larger than this are subsampled when estimating the background
MAX_BACKGROUND_SAMPLE_SIZE = 1_000_000


def background_mode(image: np.ndarray, nbins: int = 256) -> float:
    """
    Estimate the background (vacuum) level of an image from its intensity histogram.

    The background is taken as the most common intensity: the peak of a lightly
    smoothed histogram (ignoring the extreme 0.5% of values at each end), refined
    as the median of the pixels close to that peak. Large images are subsampled
    on a regular grid to keep the estimate fast.

    Parameters
    ----------
    image
        2D image
    nbins
        Number of histogram bins

    Returns
    -------
    background : float
        Estimated background level
    """
    step = max(1, int(np.ceil(np.sqrt(image.size / MAX_BACKGROUND_SAMPLE_SIZE))))
    sample = image[::step, ::step]
    low, high = np.percentile(sample, [0.5, 99.5])
    if high <= low:
        return float(np.median(sample))
    counts, edges = np.histogram(sample, bins=nbins, range=(low, high))
    peak = int(np.argmax(ndimage.gaussian_filter1d(counts.astype(float), 2)))
    center = 0.5 * (edges[peak] + edges[peak + 1])
    bin_width = edges[1] - edges[0]
    return float(np.median(sample[np.abs(sample - center) <= 2 * bin_width]))


def projected_mass(data: np.ndarray) -> np.ndarray:
    """
    Convert projection images to projected mass for center of mass calculations.

    The background (vacuum) level of each projection, estimated with
    :py:func:`background_mode`, is subtracted, and negative values (noise in the
    vacuum, which cannot represent mass) are set to zero. This makes center of
    mass calculations independent of any offset or background in the data.

    Parameters
    ----------
    data
        Array of projections with shape ``(..., ny, nx)``

    Returns
    -------
    mass : :py:class:`~numpy.ndarray`
        Non-negative array of the same shape as ``data`` (as floating point)
    """
    images = data.reshape(-1, *data.shape[-2:])
    float_dtype = np.result_type(data.dtype, np.float32)
    background = np.array([background_mode(im) for im in images], dtype=float_dtype)
    mass = images.astype(float_dtype) - background[:, np.newaxis, np.newaxis]
    np.clip(mass, 0, None, out=mass)
    return mass.reshape(data.shape)
