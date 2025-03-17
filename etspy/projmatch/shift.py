"""Shift module for projection alignment module."""

import numpy as np
from scipy.fft import fft, fft2, fftfreq, ifft, ifft2

try:
    import cupy as cp
except ImportError:
    gpu_enabled = False
else:
    gpu_enabled = True

DIM_2D = 2
ERROR_SHAPE_MISMATCH = "Shapes of image and shifts is inconsistent"
ERROR_GPU_NOT_ENABLED = "cupy must be installed for GPU acceleration"


def imshift(
    image: np.ndarray,
    shifts: np.ndarray,
    pad: bool = True,
    cuda: bool = False,
) -> np.ndarray:
    """Shift image or stack of images using Fourier shift method.

    Parameters
    ----------
    image : np.ndarray
        Image or stack of images to shift
    shifts : np.ndarray
        Shifts to apply to stack
    pad : bool, optional
        If True, pad the input array to the next highest power of 2 to prevent
        wraparound of pixels, by default True
    cuda : bool, optional
        If True use cupy for CUDA acceleration, by default False

    Returns
    -------
    shifted: np.ndarray
        Shifted version of input images

    Raises
    ------
    ValueError
        ERROR_GPU_NOT_ENABLED

    """
    if np.all(shifts == 0):
        return image
    if cuda:
        if not gpu_enabled:
            raise ValueError(ERROR_GPU_NOT_ENABLED)
        shifted = imshift_gpu(image, shifts, pad)
    else:
        shifted = imshift_cpu(image, shifts, pad)
    return shifted


def imshift_cpu(
    image: np.ndarray,
    shifts: np.ndarray,
    pad: bool = True,
) -> np.ndarray:
    """CPU-based version of Fourier image shift.

    Parameters
    ----------
    image : np.ndarray
        Image or stack of images to shift
    shifts : np.ndarray
        Shifts to apply
    pad : bool, optional
        If True, pad image array to prevent wrap around of pixels

    Returns
    -------
    shifted : np.ndarray
        Shifted version of input array

    Raises
    ------
    ValueError
        _ERROR_SHAPE_MISMATCH

    """
    is_real = np.isrealobj(image)

    if len(image.shape) == DIM_2D:
        if len(shifts.shape) != 1:
            raise ValueError(ERROR_SHAPE_MISMATCH)

        ntilts, ny = image.shape
        if pad:
            y_pad_min = np.abs(shifts).max() + ny
            ny_pad = int(2 ** np.ceil(np.log2(y_pad_min)))
            pad_width = [(ny_pad - ny) // 2, (ny_pad - ny + 1) // 2]
            image_padded = np.pad(image, ((0, 0), pad_width), mode="constant")

            image = image_padded
            _, ny = image.shape

        image_fft = fft(image, axis=1)
        k = fftfreq(ny) * 2j * np.pi
        shift_matrix = shifts[:, np.newaxis] * k[np.newaxis, :]
        shift_matrix = np.exp(-shift_matrix)
        image_fft = image_fft * shift_matrix
        shifted = ifft(image_fft, axis=1)
        shifted = np.real(shifted)

        if pad:
            slices = [
                slice(0, None),
            ]
            pad_width[1] = None if pad_width[1] == 0 else -pad_width[1]
            slices.append(slice(pad_width[0], pad_width[1]))
            shifted = shifted[tuple(slices)]

    else:
        if len(shifts.shape) != DIM_2D:
            raise ValueError(ERROR_SHAPE_MISMATCH)

        _, ny, nx = image.shape
        if pad:
            y_pad_min = np.abs(shifts[:, 0]).max() + ny
            ny_pad = int(2 ** np.ceil(np.log2(y_pad_min)))
            y_pad_width = [(ny_pad - ny) // 2, (ny_pad - ny + 1) // 2]

            x_pad_min = np.abs(shifts[:, 1]).max() + nx
            nx_pad = int(2 ** np.ceil(np.log2(x_pad_min)))
            x_pad_width = [(nx_pad - nx) // 2, (nx_pad - nx + 1) // 2]

            image_padded = np.pad(
                image,
                ((0, 0), y_pad_width, x_pad_width),
                mode="constant",
            )

            image = image_padded
            _, ny, nx = image.shape

        image_fft = fft2(image, axes=(1, 2)) if is_real else image

        y = fftfreq(ny) * 2j * np.pi
        y_shift_matrix = np.exp(-shifts[:, 0][:, np.newaxis] * y)
        x = fftfreq(nx) * 2j * np.pi
        x_shift_matrix = np.exp(-shifts[:, 1][:, np.newaxis] * x)
        shifted = image_fft * y_shift_matrix[:, :, np.newaxis]
        shifted = shifted * x_shift_matrix[:, np.newaxis, :]

        shifted = ifft2(shifted, axes=(1, 2))

        if is_real:
            shifted = np.real(shifted)

        if pad:
            slices = [
                slice(0, None),
            ]
            for i in [y_pad_width, x_pad_width]:
                i[1] = None if i[1] == 0 else -i[1]
                slices.append(slice(i[0], i[1]))
            shifted = shifted[tuple(slices)]
    return shifted


def imshift_gpu(image: np.ndarray, shifts: np.ndarray, pad: bool = True):
    """GPU-based version of Fourier image shift.

    Parameters
    ----------
    image : np.ndarray
        Image or stack of images to shift
    shifts : np.ndarray
        Shifts to apply
    pad : bool, optional
        If True, pad image array to prevent wrap around of pixels

    Returns
    -------
    shifted : np.ndarray
        Shifted version of input array

    Raises
    ------
    ValueError
        ERROR_GPU_NOT_ENABLED
    ValueError
        ERROR_SHAPE_MISMATCH

    """
    if not gpu_enabled:
        raise ValueError(ERROR_GPU_NOT_ENABLED)
    image = cp.array(image)
    shifts = cp.array(shifts)
    is_real = cp.isrealobj(image)

    if len(image.shape) == DIM_2D:
        if len(shifts.shape) != 1:
            raise ValueError(ERROR_SHAPE_MISMATCH)

        ntilts, ny = image.shape
        if pad:
            y_pad_min = cp.abs(shifts).max() + ny
            ny_pad = int(2 ** cp.ceil(cp.log2(y_pad_min)))
            pad_width = [(ny_pad - ny) // 2, (ny_pad - ny + 1) // 2]
            image_padded = cp.pad(image, ((0, 0), pad_width), mode="constant")

            image = image_padded
            _, ny = image.shape

        image_fft = cp.fft.fft(image, axis=1)
        k = cp.fft.fftfreq(ny) * 2j * cp.pi
        shift_matrix = shifts[:, cp.newaxis] * k[cp.newaxis, :]
        shift_matrix = cp.exp(-shift_matrix)
        image_fft = image_fft * shift_matrix
        shifted = cp.fft.ifft(image_fft, axis=1)
        shifted = cp.real(shifted)

        if pad:
            slices = [
                slice(0, None),
            ]
            pad_width[1] = None if pad_width[1] == 0 else -pad_width[1]
            slices.append(slice(pad_width[0], pad_width[1]))
            shifted = shifted[tuple(slices)]

    else:
        if len(shifts.shape) != DIM_2D:
            raise ValueError(ERROR_SHAPE_MISMATCH)

        _, ny, nx = image.shape
        if pad:
            y_pad_min = cp.abs(shifts[:, 0]).max() + ny
            ny_pad = int(2 ** cp.ceil(cp.log2(y_pad_min)))
            y_pad_width = [(ny_pad - ny) // 2, (ny_pad - ny + 1) // 2]

            x_pad_min = np.abs(shifts[:, 1]).max() + nx
            nx_pad = int(2 ** cp.ceil(cp.log2(x_pad_min)))
            x_pad_width = [(nx_pad - nx) // 2, (nx_pad - nx + 1) // 2]

            image_padded = cp.pad(
                image,
                ((0, 0), y_pad_width, x_pad_width),
                mode="constant",
            )

            image = image_padded
            _, ny, nx = image.shape

        image_fft = cp.fft.fft2(image, axes=(1, 2)) if is_real else image

        y = cp.fft.fftfreq(ny) * 2j * cp.pi
        y_shift_matrix = cp.exp(-shifts[:, 0][:, cp.newaxis] * y)
        x = cp.fft.fftfreq(nx) * 2j * cp.pi
        x_shift_matrix = cp.exp(-shifts[:, 1][:, cp.newaxis] * x)
        shifted = image_fft * y_shift_matrix[:, :, cp.newaxis]
        shifted = shifted * x_shift_matrix[:, cp.newaxis, :]

        shifted = cp.fft.ifft2(shifted, axes=(1, 2))

        if is_real:
            shifted = cp.real(shifted)

        if pad:
            slices = [
                slice(0, None),
            ]
            for i in [y_pad_width, x_pad_width]:
                i[1] = None if i[1] == 0 else -i[1]
                slices.append(slice(i[0], i[1]))
            shifted = shifted[tuple(slices)]
    return cp.asnumpy(shifted)
