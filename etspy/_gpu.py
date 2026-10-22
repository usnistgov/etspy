"""CuPy-based GPU support for ETSpy."""

try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import affine_transform as affine_transform_gpu
    from cupyx.scipy.ndimage import shift as shift_gpu

    has_cupy = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    affine_transform_gpu = None
    shift_gpu = None
    has_cupy = False
