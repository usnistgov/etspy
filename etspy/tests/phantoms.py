"""Synthetic phantoms for physics tests."""

import numpy as np
from scipy import ndimage

from etspy import simulation
from etspy.base import TomoStack

TILTS = np.arange(-70, 71, 2.0)


def make_phantom(
    nx: int = 48,
    nz: int = 64,
    ny: int = 96,
    nblobs: int = 12,
    seed: int = 0,
) -> np.ndarray:
    """Random-blob volume."""
    rng = np.random.default_rng(seed)
    x, z, y = np.ogrid[:nx, :nz, :ny]
    vol = np.zeros((nx, nz, ny), np.float32)
    for _ in range(nblobs):
        cx = rng.uniform(8, nx - 8)
        cz = rng.uniform(16, nz - 16)
        cy = rng.uniform(25, ny - 25)
        r = rng.uniform(3, 6)
        vol[(x - cx) ** 2 + (z - cz) ** 2 + (y - cy) ** 2 < r**2] += 1
    vol[:, 20:24, 40:44] += 2
    return vol


def project(vol: np.ndarray, tilts: np.ndarray = TILTS) -> TomoStack:
    """Create tilt series of phantom."""
    stack = simulation.create_model_tilt_series(vol, tilts, cuda=False)
    stack.data = stack.data.astype(np.float32)
    return stack


def shift_images(stack: TomoStack, shifts: np.ndarray) -> TomoStack:
    """Shifted stack in x and y."""
    out = stack.deepcopy()
    for i, shift in enumerate(shifts):
        out.data[i] = ndimage.shift(stack.data[i], shift, order=3)
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cross-correlation of two arrays."""
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
