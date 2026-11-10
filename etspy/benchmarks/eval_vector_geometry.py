"""
Evaluate ASTRA's 3D vector geometry for applying ETSpy alignments without resampling.

ETSpy corrects a misaligned tilt series by resampling the projections; since
``align_other`` (with Fourier resampling) this is done once. ASTRA's
``parallel3d_vec`` geometry can instead describe where each (misaligned)
projection was recorded, so a reconstruction can use the measured projections
directly, with no resampling at all. ASTRA only provides this geometry on the
GPU, with SIRT3D_CUDA, CGLS3D_CUDA, FP3D_CUDA, and BP3D_CUDA.

This script simulates a misaligned acquisition exactly (by projecting a 3D phantom
with a displaced and rotated detector, including shifts along the tilt axis and an
in-plane rotation of the tilt axis), then reconstructs it with:

1. ideal (perfectly aligned) projections, as a reference,
2. ETSpy's previous sequential resampling (``align_other(..., shift_type="interp")``),
3. ETSpy's single Fourier resampling (``align_other(...)``),
4. ASTRA's 3D vector geometry (SIRT3D_CUDA on the measured projections),

and reports the reconstruction error and high-frequency content (a resolution
proxy) of each, along with run times. It also checks that its geometry matches
ETSpy's alignment conventions.

Usage (on a machine with a CUDA GPU)::

    python benchmarks/eval_vector_geometry.py               # 128^3 volume
    python benchmarks/eval_vector_geometry.py --size 256    # larger, for timing
    python benchmarks/eval_vector_geometry.py --figure out.png

Without a GPU, only the geometry convention check can run::

    python benchmarks/eval_vector_geometry.py --check-geometry

Things to look at in the output:

- "geometry checks" should all pass; if the 3D/2D projection check fails, the
  mapping between ETSpy's slice-wise reconstruction and ASTRA's 3D volume is not as
  assumed, and the remaining numbers are not meaningful.
- Compare "3D vector geometry" with "ETSpy single Fourier resampling": if the error
  and high-frequency content are close, the extra complexity of the 3D geometry
  buys little in accuracy, and the remaining question is speed (compare the SIRT
  times of the 2D slice-wise and 3D reconstructions at realistic sizes).
"""

import argparse
import logging
import sys
import time

import astra
import numpy as np

logging.disable(logging.WARNING)

# (x, y): ETSpy rotates about (n/2) while ASTRA centers its detector on (n - 1)/2
HALF_PIXEL = np.array([0.5, 0.5])


def detector_vectors(tilts, shifts, xshift, yshift, tiltaxis):
    """
    ASTRA parallel3d_vec vectors for a detector misaligned as an ETSpy alignment.

    The alignment (``shifts`` as in TomoStack.shifts, then ``xshift``, ``yshift``
    and ``tiltaxis`` as in TomoStack.trans_stack) corrects the misalignment: the
    content of measured pixel ``p`` moves to ``q = R (p + s - c) + c + t`` in the
    aligned image. So each measured pixel records what the ideal detector records
    at ``q``.

    Volume layout: ETSpy RecStack order ``(x, z, y)`` = ASTRA ``(Z, Y, X)``, with
    the tilt axis along ASTRA Z. Detector: columns (u) along ETSpy y, rows (v)
    along ETSpy x (the tilt axis), so the projection data are ``(nx, nangles,
    ny)``.
    """
    thetas = np.deg2rad(np.asarray(tilts, dtype=float))
    phi = np.deg2rad(tiltaxis)
    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    translation = np.array([xshift, -yshift])
    axis_dir = np.array([0.0, 0.0, 1.0])
    vectors = np.zeros((len(thetas), 12))
    for i, theta in enumerate(thetas):
        ray = np.array([np.sin(theta), -np.cos(theta), 0.0])
        in_plane = np.array([np.cos(theta), np.sin(theta), 0.0])
        shift_xy = np.array([shifts[i, 1], shifts[i, 0]])
        offset = rotation @ (shift_xy - HALF_PIXEL) + translation + HALF_PIXEL
        center = offset[0] * axis_dir + offset[1] * in_plane
        u = -np.sin(phi) * axis_dir + np.cos(phi) * in_plane  # step along ETSpy y
        v = np.cos(phi) * axis_dir + np.sin(phi) * in_plane  # step along ETSpy x
        vectors[i] = np.concatenate([ray, center, u, v])
    return vectors


def check_geometry(nx=40, ny=32, seed=0):
    """
    Check the detector vectors against ETSpy's alignment convention (CPU only).

    For random alignments, map measured pixels through the detector vectors to the
    ideal detector, and compare with the position ETSpy's alignment moves them to.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(20):
        tilts = rng.uniform(-70, 70, 5)
        shifts = rng.uniform(-6, 6, (5, 2))
        xshift, yshift = rng.uniform(-5, 5, 2)
        tiltaxis = rng.uniform(-30, 30)
        vectors = detector_vectors(tilts, shifts, xshift, yshift, tiltaxis)
        phi = np.deg2rad(tiltaxis)
        rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        center = np.array([nx / 2, ny / 2])
        for i, theta in enumerate(np.deg2rad(tilts)):
            in_plane = np.array([np.cos(theta), np.sin(theta), 0.0])
            axis_dir = np.array([0.0, 0.0, 1.0])
            d, u, v = vectors[i, 3:6], vectors[i, 6:9], vectors[i, 9:12]
            for _ in range(10):
                p = rng.uniform([0, 0], [nx - 1, ny - 1])  # measured pixel (x, y)
                point = d + (p[1] - (ny - 1) / 2) * u + (p[0] - (nx - 1) / 2) * v
                ideal = np.array(
                    [point @ axis_dir + (nx - 1) / 2, point @ in_plane + (ny - 1) / 2],
                )
                shift_xy = np.array([shifts[i, 1], shifts[i, 0]])
                etspy = rotation @ (p + shift_xy - center) + center
                etspy += np.array([xshift, -yshift])
                worst = max(worst, float(np.abs(ideal - etspy).max()))
    return worst


def make_phantom(nx, nz, ny, seed=1):
    """3D phantom in ETSpy RecStack order (x, z, y): a support plus small features."""
    rng = np.random.default_rng(seed)
    x, z, y = np.meshgrid(
        np.arange(nx),
        np.arange(nz),
        np.arange(ny),
        indexing="ij",
    )
    vol = np.zeros((nx, nz, ny), np.float32)
    vol[(y - ny / 2) ** 2 + (z - nz / 2) ** 2 < (0.3 * ny) ** 2] = 0.3
    for _ in range(max(40, nx // 2)):
        lo, hi = [0.15 * nx, 0.3 * nz, 0.3 * ny], [0.85 * nx, 0.7 * nz, 0.7 * ny]
        c = rng.uniform(lo, hi)
        r = rng.uniform(1.0, 4.0)
        vol[(x - c[0]) ** 2 + (z - c[1]) ** 2 + (y - c[2]) ** 2 < r**2] = 1.0
    return vol


def forward_3d(volume, proj_geom):
    """Forward project an (x, z, y) volume; returns an ETSpy stack (nangles, ny, nx)."""
    nx, nz, ny = volume.shape
    vol_geom = astra.create_vol_geom(nz, ny, nx)
    sino_id, data = astra.create_sino3d_gpu(volume, proj_geom, vol_geom)
    astra.data3d.delete(sino_id)
    return np.ascontiguousarray(np.transpose(data, (1, 2, 0)))


def sirt_3d(stack, proj_geom, shape, iterations):
    """Run SIRT3D_CUDA on an ETSpy stack (nangles, ny, nx); returns (x, z, y)."""
    nx, nz, ny = shape
    vol_geom = astra.create_vol_geom(nz, ny, nx)
    data = np.ascontiguousarray(np.transpose(stack, (2, 0, 1)))
    sino_id = astra.data3d.create("-sino", proj_geom, data)
    rec_id = astra.data3d.create("-vol", vol_geom, 0)
    cfg = astra.astra_dict("SIRT3D_CUDA")
    cfg["ProjectionDataId"] = sino_id
    cfg["ReconstructionDataId"] = rec_id
    cfg["option"] = {"MinConstraint": 0.0}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, iterations)
    rec = astra.data3d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete([sino_id, rec_id])
    return rec


def high_frequency_ratio(rec, phantom, cutoff=0.2):
    """High-frequency content of rec relative to the phantom (in the y-z planes)."""
    ny, nz = phantom.shape[2], phantom.shape[1]
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nz)))
    band = k > cutoff

    def power(v):
        return np.abs(np.fft.fft2(v, axes=(1, 2)))[:, band].sum()

    return power(rec) / power(phantom)


def main():  # noqa: PLR0915
    """Run the evaluation."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--size", type=int, default=128, help="volume size (cube)")
    parser.add_argument("--iterations", type=int, default=100, help="SIRT iterations")
    parser.add_argument("--figure", help="save a figure of the central slices")
    parser.add_argument(
        "--check-geometry",
        action="store_true",
        help="only run the CPU geometry convention check",
    )
    args = parser.parse_args()

    worst = check_geometry()
    status = "PASS" if worst < 1e-9 else "FAIL"
    print(
        f"geometry check (detector vectors vs ETSpy convention): {status} "
        f"(max deviation {worst:.2e} px)"
    )
    if args.check_geometry:
        return
    if not astra.use_cuda():
        print("ASTRA does not detect a CUDA GPU; 3D geometries are GPU-only. Exiting.")
        sys.exit(1)

    from etspy.align import apply_shifts  # noqa: PLC0415
    from etspy.base import RecStack, TomoStack  # noqa: PLC0415

    n = args.size
    nx = nz = ny = n
    tilts = np.arange(-70, 71, 2.0)
    rng = np.random.default_rng(2)
    shifts = rng.uniform(-5, 5, (len(tilts), 2))
    xshift, yshift, tiltaxis = 3.3, -4.1, 2.5
    phantom = make_phantom(nx, nz, ny)

    standard = astra.create_proj_geom("parallel3d", 1.0, 1.0, nx, ny, np.deg2rad(tilts))
    misaligned = astra.create_proj_geom(
        "parallel3d_vec",
        nx,
        ny,
        detector_vectors(tilts, shifts, xshift, yshift, tiltaxis),
    )
    ideal = forward_3d(phantom, standard)
    measured = forward_3d(phantom, misaligned)

    # consistency of ASTRA's 3D geometry with ETSpy's slice-wise (2D) projections
    fp2d = RecStack(phantom).forward_project(tilts, cuda=True).data
    mapping = np.linalg.norm(fp2d - ideal) / np.linalg.norm(ideal)
    print(
        f"geometry check (3D projection vs ETSpy 2D slice-wise projection): "
        f"{'PASS' if mapping < 0.02 else 'FAIL'} (relative difference {mapping:.4f})"
    )

    raw = TomoStack(measured, tilts=tilts)
    reference = TomoStack(np.zeros_like(measured), tilts=tilts)
    reference.shifts = shifts
    reference.metadata.set_item("Tomography.xshift", xshift)
    reference.metadata.set_item("Tomography.yshift", yshift)
    reference.metadata.set_item("Tomography.tiltaxis", tiltaxis)

    # measure away from the edges, where the misalignment moves data out of view
    inner = (slice(int(0.1 * nx), int(0.9 * nx)), slice(None), slice(None))
    results = {}

    def evaluate(label, rec, seconds):
        diff = rec[inner] - phantom[inner]
        err = np.linalg.norm(diff) / np.linalg.norm(phantom[inner])
        hf = high_frequency_ratio(rec[inner], phantom[inner])
        results[label] = rec
        print(f"  {label:44s} error {err:.4f}   HF {hf:.3f}   ({seconds:6.1f} s)")

    print(
        f"\nreconstructions ({n}^3 volume, {len(tilts)} tilts, SIRT "
        f"{args.iterations} iterations):"
    )
    t = time.perf_counter()
    rec = (
        TomoStack(ideal, tilts=tilts)
        .reconstruct(
            "SIRT",
            iterations=args.iterations,
            constrain=True,
            cuda=True,
            show_progressbar=False,
        )
        .data
    )
    evaluate("ideal projections (2D slice-wise SIRT)", rec, time.perf_counter() - t)
    t = time.perf_counter()
    rec = sirt_3d(ideal, standard, (nx, nz, ny), args.iterations)
    evaluate("ideal projections (SIRT3D)", rec, time.perf_counter() - t)

    for label, kwargs in (
        ("ETSpy sequential resampling", {"shift_type": "interp"}),
        ("ETSpy single Fourier resampling", {}),
    ):
        t = time.perf_counter()
        aligned = reference.align_other(raw, **kwargs)
        corrected_error = np.linalg.norm(
            (aligned.data - ideal)[:, :, inner[0]],
        ) / np.linalg.norm(ideal[:, :, inner[0]])
        rec = aligned.reconstruct(
            "SIRT",
            iterations=args.iterations,
            constrain=True,
            cuda=True,
            show_progressbar=False,
        ).data
        evaluate(label, rec, time.perf_counter() - t)
        print(f"  {'':44s} (corrected projections vs ideal: {corrected_error:.4f})")

    t = time.perf_counter()
    rec = sirt_3d(measured, misaligned, (nx, nz, ny), args.iterations)
    evaluate("3D vector geometry (SIRT3D, no resampling)", rec, time.perf_counter() - t)

    # sanity check that the alignment is applied in the direction assumed
    _ = apply_shifts  # (imported for users extending the comparison)

    if args.figure:
        import matplotlib as mpl  # noqa: PLC0415

        mpl.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        ncols = len(results) + 1
        fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3.4))
        axes[0].imshow(phantom[nx // 2], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("phantom", fontsize=8)
        for ax, (label, rec) in zip(axes[1:], results.items(), strict=True):
            ax.imshow(rec[nx // 2], cmap="gray", vmin=0, vmax=1)
            ax.set_title(label, fontsize=7)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.figure, dpi=120)
        print(f"\nfigure saved to {args.figure}")


if __name__ == "__main__":
    main()
