"""
Benchmark projection matching alignment on analytic test data.

Tilt series of a phantom with features that are sharp along both x (the tilt axis)
and y are simulated with a known misalignment (see
:py:func:`etspy.simulation.create_alignment_test_series`), and aligned with
:py:class:`etspy.projmatch.ProjMatch` in several configurations:

- a single slice (Y-shifts only, as ETSpy did before)
- all slices, Y-shifts only
- all slices, Y- and X-shifts, with FBP, SIRT (the default) and SART reconstructions

For each configuration and dose, the script reports the RMS error of the Y- and
X-shifts (without the components that cannot be determined: a + b cos + c sin for
Y, a constant for X), the error of a SIRT reconstruction of the aligned series
(relative to the reconstruction of the ideal series), and the run time.

It also measures the accuracy of the tilt axis rotation found by
:py:meth:`etspy.base.TomoStack.tilt_align` (CoM method), the method to use before
projection matching, after the true shifts have been corrected.

Run with, for example::

    python benchmarks/eval_projmatch.py --doses none 200 50
"""

import argparse
import logging
import time

import numpy as np

from etspy import projmatch
from etspy import simulation as sim
from etspy.base import TomoStack

CONFIGS = {
    "single slice (FBP)": {"nslice": "middle", "params": {"recon_algorithm": "FBP"}},
    "all slices, Y only (FBP)": {
        "params": {"align_x": False, "recon_algorithm": "FBP"},
    },
    "all slices, Y+X (FBP)": {"params": {"recon_algorithm": "FBP"}},
    "all slices, Y+X (SIRT, default)": {"params": {}},
    "all slices, Y+X (SART)": {
        "params": {"recon_algorithm": "SART", "recon_iterations": 5},
    },
}


def undeterminable_part(shifts, truth, tilts):
    """
    Return the part of ``shifts - truth`` that cannot be determined.

    Y-shifts a + b cos(theta) + c sin(theta) only translate the reconstruction (or
    move the tilt axis), and a constant X-shift only translates it along the tilt
    axis.
    """
    thetas = np.deg2rad(tilts)
    basis = np.array([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)]).T
    part = np.zeros_like(shifts)
    error_y = shifts[:, 0] - truth[:, 0]
    part[:, 0] = basis @ np.linalg.lstsq(basis, error_y, rcond=None)[0]
    part[:, 1] = np.median(shifts[:, 1] - truth[:, 1])
    return part


def shift_errors(shifts, truth, tilts):
    """RMS errors of (Y, X) shifts, without the undeterminable components."""
    error = shifts - truth - undeterminable_part(shifts, truth, tilts)
    return np.sqrt(np.mean(error[:, 0] ** 2)), np.sqrt(np.mean(error[:, 1] ** 2))


def reconstruction_error(series, shifts, ideal_rec, background):
    """
    Relative RMS error of the SIRT reconstruction of a series aligned by ``shifts``.

    The undeterminable part of the shifts, which only translates the
    reconstruction, is removed first, so that the reconstruction can be compared
    with that of the ideal series.
    """
    tilts = series.misaligned.tilts.data.squeeze()
    shifts = shifts - undeterminable_part(shifts, series.shifts, tilts)
    pm = projmatch.ProjMatch(series.misaligned, nslice=0)
    pm.total_shifts = shifts
    stack = pm.apply_shifts()
    stack.data = stack.data - background
    rec = stack.reconstruct("SIRT", iterations=30, cuda=False, show_progressbar=False)
    diff = rec.data - ideal_rec
    return np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(ideal_rec**2))


def run_projmatch(series, config):
    """Align a series with one configuration; return the ProjMatch and time."""
    kwargs = dict(config)
    params = dict(kwargs.pop("params"))
    if kwargs.get("nslice") == "middle":
        kwargs["nslice"] = series.misaligned.data.shape[2] // 2
    start = time.perf_counter()
    pm = projmatch.ProjMatch(series.misaligned, params=params, **kwargs)
    pm.calculate_shifts()
    return pm, time.perf_counter() - start


def benchmark_projmatch(args):
    """Compare the projection matching configurations at each dose."""
    print(
        f"\nProjection matching, {args.size} x {args.size} pixels, tilts -70 to 70 "
        "degrees in 2 degree steps, shifts up to about +/-5 px",
    )
    print(
        f"{'dose':>6}  {'configuration':28s} {'Y (px)':>7} {'X (px)':>7} "
        f"{'rec err':>8} {'time (s)':>9}",
    )
    for dose in args.doses:
        series = sim.create_alignment_test_series(
            shape=(args.size, args.size),
            tilt_axis=0.0,
            blur=0.5,
            dose=dose,
            seed=args.seed,
        )
        tilts = series.misaligned.tilts.data.squeeze()
        ideal = series.aligned.deepcopy()
        ideal.data = ideal.data - args.background
        ideal_rec = ideal.reconstruct(
            "SIRT",
            iterations=30,
            cuda=False,
            show_progressbar=False,
        ).data
        label = "none" if dose is None else str(dose)
        error_y, error_x = shift_errors(
            np.zeros_like(series.shifts),
            series.shifts,
            tilts,
        )
        rec_error = reconstruction_error(
            series,
            np.zeros_like(series.shifts),
            ideal_rec,
            args.background,
        )
        print(
            f"{label:>6}  {'(no alignment)':28s} {error_y:7.3f} {error_x:7.3f} "
            f"{rec_error:8.3f}",
        )
        for name, config in CONFIGS.items():
            pm, elapsed = run_projmatch(series, config)
            error_y, error_x = shift_errors(pm.total_shifts, series.shifts, tilts)
            rec_error = reconstruction_error(
                series,
                pm.total_shifts,
                ideal_rec,
                args.background,
            )
            print(
                f"{label:>6}  {name:28s} {error_y:7.3f} {error_x:7.3f} "
                f"{rec_error:8.3f} {elapsed:9.1f}",
                flush=True,
            )


def benchmark_tilt_axis(args):
    """Measure the accuracy of TiltCOM after correcting the true shifts."""
    print("\nTilt axis (CoM method) after correcting the true shifts:")
    print(f"{'dose':>6}  {'true (deg)':>10} {'found (deg)':>11} {'error':>7}")
    for dose in args.doses:
        for angle in (0.5, 1.0, 2.0):
            series = sim.create_alignment_test_series(
                shape=(args.size, args.size),
                tilt_axis=angle,
                blur=0.5,
                dose=dose,
                seed=args.seed,
            )
            pm = projmatch.ProjMatch(series.misaligned, nslice=0)
            pm.total_shifts = series.shifts
            shifted = pm.apply_shifts()
            stack = TomoStack(shifted.data, series.misaligned.tilts.data.squeeze())
            found = float(stack.tilt_align("CoM").metadata.Tomography.tiltaxis)
            label = "none" if dose is None else str(dose)
            print(
                f"{label:>6}  {angle:10.2f} {found:11.3f} {found - angle:+7.3f}",
                flush=True,
            )


def main():
    """Run the benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--size", type=int, default=128, help="image size (pixels)")
    parser.add_argument(
        "--doses",
        nargs="+",
        default=["none", "200", "50"],
        help='counts per pixel at an intensity of 1, or "none" for no noise',
    )
    parser.add_argument("--seed", type=int, default=1, help="phantom seed")
    parser.add_argument(
        "--skip-tilt-axis",
        action="store_true",
        help="skip the tilt axis benchmark",
    )
    args = parser.parse_args()
    args.doses = [None if d.lower() == "none" else float(d) for d in args.doses]
    args.background = 0.2  # background level of the simulated series
    logging.disable(logging.WARNING)
    benchmark_projmatch(args)
    if not args.skip_tilt_axis:
        benchmark_tilt_axis(args)


if __name__ == "__main__":
    main()
