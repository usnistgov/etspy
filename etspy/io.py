"""Data input/output module for ETSpy package."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from hyperspy._signals.signal2d import (
    Signal2D,  # import from _signals for type-checking
)
from hyperspy.io import (
    load as hs_load,  # import load function directly for better type-checking
)
from hyperspy.misc.utils import DictionaryTreeBrowser as Dtb
from hyperspy.misc.utils import (
    stack as hs_stack,  # import stack function directly for better type-checking
)

from etspy.base import TomoStack

if TYPE_CHECKING:
    from hyperspy.misc.utils import DictionaryTreeBrowser as Dtb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
PathLike = str | Path

# declare known file types for later use
hspy_file_types = [".hdf5", ".h5", ".hspy"]
mrc_file_types = [".mrc", ".ali", ".rec"]
dm_file_types = [".dm3", ".dm4"]
known_file_types = hspy_file_types + mrc_file_types + dm_file_types


def get_mrc_tilts(
    stack: Signal2D | TomoStack,
    filename: PathLike,
) -> np.ndarray | None:
    """Extract tilts from an MRC file.

    Parameters
    ----------
    stack
        A HyperSpy or TomoStack signal
    filename
        Name of MRC file from which to extract tilts

    Returns
    -------
    tilts: :py:class:`~numpy.ndarray` or None
        Tilt angles extracted from MRC file (or ``None`` if not present)

    Group
    -----
    io
    """
    if isinstance(filename, str):
        filename = Path(filename)
    tiltfile = filename.with_suffix(".rawtlt")
    tilts = None
    if stack.original_metadata.has_item("fei_header"):
        fei_header = cast("Dtb", stack.original_metadata.fei_header)
        if fei_header.has_item("a_tilt"):
            tilts = fei_header["a_tilt"][0 : stack.data.shape[0]]
            logger.info("Tilts found in MRC file header")
    elif stack.original_metadata.has_item("std_header"):
        header = parse_mrc_header(filename)
        if header["ext_type"] == "SERI" and header["nint"] and header["nreal"] & 1:
            logger.info("SerialEM generated MRC file detected")
            stride = int(header["nint"]) // 2
            tilts = header["ext_header"][::stride][0 : stack.data.shape[0]] / 100
    if tilts is None and tiltfile.is_file():
        tilts = np.loadtxt(tiltfile)
        logger.info(".rawtlt file detected.")
        if len(tilts) == stack.data.shape[0]:
            logger.info("Tilts loaded from .rawtlt file")
        else:
            msg = "Number of tilts in .rawtlt file inconsistent with data shape"
            raise ValueError(msg)
    return tilts


def get_dm_tilts(s: Signal2D | TomoStack) -> np.ndarray:
    """Extract tilts from DM tags.

    Parameters
    ----------
    s
        A HyperSpy or ETSpy signal containing DigitalMigrograph
        metadata tags

    Returns
    -------
    tilts: :py:class:`~numpy.ndarray`
        Tilt angles extracted from the DM tags

    Group
    -----
    io
    """
    maxtilt = s.original_metadata["ImageList"]["TagGroup0"]["ImageTags"]["Tomography"][
        "Tomography_setup"
    ]["Tilt_angles"]["Maximum_tilt_angle_deg"]

    mintilt = s.original_metadata["ImageList"]["TagGroup0"]["ImageTags"]["Tomography"][
        "Tomography_setup"
    ]["Tilt_angles"]["Minimum_tilt_angle_deg"]

    tiltstep = s.original_metadata["ImageList"]["TagGroup0"]["ImageTags"]["Tomography"][
        "Tomography_setup"
    ]["Tilt_angles"]["Tilt_angle_step_deg"]

    tilts = mintilt + tiltstep * np.arange(s.data.shape[0])
    if abs(tilts[-1] - maxtilt) > abs(tiltstep) / 2:
        logger.warning(
            "Tilts from DM tags end at %.2f, not the recorded maximum of %.2f",
            tilts[-1],
            maxtilt,
        )
    return tilts


def parse_mdoc(
    mdoc_file: PathLike,
    series: bool = False,
) -> tuple[dict, np.ndarray | float]:
    """Parse experimental parameters from a SerialEM MDOC file.

    Parameters
    ----------
    mdoc_file
        Name of a SerialEM MDOC file
    series
        If ``True``, the MDOC files originated from a multiscan SerialEM acquisition.
        If ``False``, the files originated from a single scan SerialEM acquisition.

    Returns
    -------
    metadata : dict
        A dictionary containing the metadata read from the MDOC file
    tilt : :py:class:`~numpy.ndarray` or float
        If ``series`` is true, tilt will be a single float value, otherwise
        it will be an ndarray containing multiple tilt values.

    Group
    -----
    io
    """
    keys = [
        "PixelSpacing",
        "Voltage",
        "ImageFile",
        "Image Size",
        "DataMode",
        "Magnification",
        "ExposureTime",
        "SpotSize",
        "Defocus",
    ]
    metadata = {}
    tilts = []
    if isinstance(mdoc_file, str):
        mdoc_file = Path(mdoc_file)
    sections = 0
    for line in mdoc_file.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("[ZValue"):
            sections += 1
            continue
        key, sep, value = stripped.partition("=")
        key, value = key.strip(), value.strip()
        if not sep:
            continue
        if key == "TiltAngle":
            tilts.append(float(value))
        # acquisition parameters come from the header and the first image section
        elif key in keys and key not in metadata and sections <= 1:
            metadata[key] = value if key == "ImageFile" else _parse_mdoc_value(value)
    if series:
        return metadata, tilts[0] if tilts else np.array([])
    return metadata, np.array(tilts)


def _parse_mdoc_value(value: str) -> float | tuple[float, ...] | str:
    """Convert an MDOC value to a float, a tuple of floats, or leave it a string."""
    try:
        numbers = tuple(float(v) for v in value.split())
    except ValueError:
        return value
    return numbers[0] if len(numbers) == 1 else numbers


def load_serialem(mrcfile: PathLike, mdocfile: PathLike) -> TomoStack:
    """
    Load a multi-frame series collected by SerialEM.

    Parameters
    ----------
    mrcfile
        Path to MRC file containing tilt series data.

    mdocfile
        Path to SerialEM metadata file for tilt series data.

    Returns
    -------
    stack : TomoStack
        Tilt series

    Group
    -----
    io
    """
    mrc_logger = logging.getLogger("hyperspy.io_plugins.mrc")
    log_level = mrc_logger.getEffectiveLevel()
    mrc_logger.setLevel(logging.ERROR)

    meta, _ = parse_mdoc(mdocfile)
    stack = hs_load(mrcfile)
    stack = cast("Signal2D", stack)

    stack.axes_manager[1].scale = stack.axes_manager[1].scale / 10
    stack.axes_manager[2].scale = stack.axes_manager[2].scale / 10
    stack.axes_manager[1].units = "nm"
    stack.axes_manager[2].units = "nm"

    _set_serialem_metadata(stack, meta)

    logger.info("SerialEM stack successfully loaded. ")
    mrc_logger.setLevel(log_level)
    return TomoStack(stack)


def load_serialem_series(
    mrcfiles: list[str] | list[Path],
    mdocfiles: list[str] | list[Path],
) -> tuple[Signal2D, np.ndarray]:
    """
    Load a multi-frame series collected by SerialEM.

    A multi-frame series is a tilt series with multiple frames at each tilt.
    The resulting signal will have two navigation axes, ``"Frames"`` and
    ``"Projections"``.

    Parameters
    ----------
    mrcfiles
        List of MRC file paths containing multi-frame tilt series data.

    mdocfiles
        List of SerialEM metadata file paths for multi-frame tilt series data.

    Returns
    -------
    stack : :py:class:`~hyperspy.api.signals.Signal2D`
        Tilt series with multiple frames at each tilt. Will have navigation shape
        (`nframes`, `ntilts`) and signal shape (`x`, `y`). The underlying data
        array will be of shape (`ntilts`, `nframes`, `y`, `x`).
    tilts : :py:class:`~numpy.ndarray`
        The tilt values for each image in the stack. Will be of size
        (`ntilts`, `nframes`).

    Group
    -----
    io
    """
    mrc_logger = logging.getLogger("hyperspy.io_plugins.mrc")
    log_level = mrc_logger.getEffectiveLevel()
    mrc_logger.setLevel(logging.ERROR)

    stack = []
    meta = []
    tilts = np.zeros(len(mdocfiles))
    for i in range(len(mdocfiles)):
        mdoc_output = parse_mdoc(mdocfiles[i], series=True)
        meta.append(mdoc_output[0])
        tilts[i] = mdoc_output[1]

    tilts_sort = np.argsort(tilts)
    tilts.sort()

    for i in range(len(mrcfiles)):
        mdoc_filename = mdocfiles[tilts_sort[i]]
        if isinstance(mdoc_filename, str):
            mdoc_filename = Path(mdoc_filename)

        # sometimes files are named "filename.mrc.mdoc", other times
        # "filename.mrc" and "filename.mdoc", so need to handle both
        if mdoc_filename.stem.lower().endswith(".mrc"):
            # "filename.mrc.mdoc" case
            mrc_filename = mdoc_filename.parent / mdoc_filename.stem
        else:
            mrc_filename = mdoc_filename.with_suffix(".mrc")
        frames = hs_load(mrc_filename)
        frames = cast("Signal2D", frames)
        nav = frames.axes_manager.navigation_axes[0]
        del nav.scale
        nav.name = "Frames"
        nav.units = "images"

        stack.append(frames)

    # build hyperspy signal from stack along a new Tilt axis
    # shape will be (ntilts, nframes, x, y)
    stack = hs_stack(stack, show_progressbar=False, new_axis_name="Projections")
    stack.axes_manager["Projections"].units = "degrees"
    stack.axes_manager["Projections"].offset = tilts[0]
    stack.axes_manager["Projections"].scale = np.diff(tilts).mean()

    # tile tilts array to match frames
    # shape will be (ntilts, nframes)
    tilts = np.tile(tilts, (stack.axes_manager["Frames"].size, 1)).T
    _set_serialem_metadata(stack, meta[0], stack.axes_manager["Frames"].size)

    logger.info(
        "SerialEM Multiframe stack successfully loaded. "
        "Use etspy.utils.register_serialem_stack to align frames.",
    )
    mrc_logger.setLevel(log_level)
    return stack, tilts


def _set_serialem_metadata(stack: Signal2D, meta: dict, images_per_tilt: int = 1):
    """Copy SerialEM MDOC parameters present in ``meta`` into the stack metadata."""
    if not stack.metadata.has_item("Acquisition_instrument.TEM"):
        stack.metadata.add_node("Acquisition_instrument.TEM")
    tem = cast("Dtb", stack.metadata.Acquisition_instrument.TEM)
    for attribute, key in (
        ("magnification", "Magnification"),
        ("beam_energy", "Voltage"),
        ("spot_size", "SpotSize"),
        ("defocus", "Defocus"),
    ):
        if key in meta:
            tem.set_item(attribute, meta[key])
    if "ExposureTime" in meta:
        npixels = stack.data.shape[-2] * stack.data.shape[-1]
        tem.set_item("dwell_time", meta["ExposureTime"] * images_per_tilt / npixels)
    if "ImageFile" in meta:
        stack.metadata.set_item("General.original_filename", meta["ImageFile"])


def parse_mrc_header(filename: PathLike) -> dict[str, Any]:
    """
    Read the mrc header and return as dictionary.

    Parameters
    ----------
    filename
        Name of the MRC file to parse

    Returns
    -------
    header : dict
        Dictionary with header values from an MRC file

    Group
    -----
    io
    """
    header = {}
    if isinstance(filename, str):
        filename = Path(filename)
    with filename.open("rb") as h:
        header["nx"], header["ny"], header["nz"] = np.fromfile(h, np.uint32, 3)
        header["mode"] = np.fromfile(h, np.uint32, 1)[0]
        header["nxstart"], header["nystart"], header["nzstart"] = np.fromfile(
            h,
            np.uint32,
            3,
        )
        header["mx"], header["my"], header["mz"] = np.fromfile(h, np.uint32, 3)
        header["xlen"], header["ylen"], header["zlen"] = np.fromfile(h, np.float32, 3)
        _ = np.fromfile(h, np.uint32, 6)
        header["amin"], header["amax"], header["amean"] = np.fromfile(h, np.float32, 3)
        _ = np.fromfile(h, np.uint32, 1)
        header["nextra"] = np.fromfile(h, np.uint32, 1)[0]
        _ = np.fromfile(h, np.uint16, 1)[0]
        _ = np.fromfile(h, np.uint8, 6)
        strbits = np.fromfile(h, np.int8, 4)
        header["ext_type"] = "".join([chr(item) for item in strbits])
        header["nversion"] = np.fromfile(h, np.uint32, 1)[0]
        _ = np.fromfile(h, np.uint8, 16)
        header["nint"] = np.fromfile(h, np.uint16, 1)[0]
        header["nreal"] = np.fromfile(h, np.uint16, 1)[0]
        _ = np.fromfile(h, np.int8, 20)
        header["imodStamp"] = np.fromfile(h, np.uint32, 1)[0]
        header["imodFlags"] = np.fromfile(h, np.uint32, 1)[0]
        header["idtype"] = np.fromfile(h, np.uint16, 1)[0]
        header["lens"] = np.fromfile(h, np.uint16, 1)[0]
        header["nd1"], header["nd2"], header["vd1"], header["vd2"] = np.fromfile(
            h,
            np.uint16,
            4,
        )
        _ = np.fromfile(h, np.float32, 6)
        header["xorg"], header["yorg"], header["zorg"] = np.fromfile(h, np.float32, 3)
        strbits = np.fromfile(h, np.int8, 4)
        header["cmap"] = "".join([chr(item) for item in strbits])
        header["stamp"] = np.fromfile(h, np.int8, 4)
        header["rms"] = np.fromfile(h, np.float32, 1)[0]
        header["nlabl"] = np.fromfile(h, np.uint32, 1)[0]
        strbits = np.fromfile(h, np.int8, 800)
        header["text"] = "".join([chr(item) for item in strbits])
        header["ext_header"] = np.fromfile(h, np.int16, int(header["nextra"] / 2))
    return header


def _load_single_file(filename: Path) -> TomoStack:
    """Load a HyperSpy signal and any tilts from a single file."""
    ext = filename.suffix
    tilts, shifts = None, None
    if ext.lower() in hspy_file_types:
        stack = hs_load(filename, file_format="HSPY")
        stack = cast("Signal2D", stack)
        if stack.metadata.has_item("Tomography.tilts"):
            tilts = stack.metadata.Tomography.tilts
            del stack.metadata.Tomography.tilts
        if stack.metadata.has_item("Tomography.shifts"):
            shifts = stack.metadata.Tomography.shifts
            del stack.metadata.Tomography.shifts
    elif ext.lower() in dm_file_types:
        stack = hs_load(filename)
        stack = cast("Signal2D", stack)
        stack.axes_manager.navigation_axes[0].name = "Projections"
        stack.axes_manager.navigation_axes[0].units = "degrees"
        stack.change_dtype(np.float32)
        tilts = get_dm_tilts(stack)
    elif ext.lower() in mrc_file_types:
        try:
            stack = hs_load(filename, file_format="mrc")
            stack = cast("Signal2D", stack)
            # TODO(jat): does a single mrc file ever have multiple tilts,
            # or is it just multiframe?
            tilts = get_mrc_tilts(stack, filename)
        except TypeError as exc:
            msg = "Unable to read MRC with Hyperspy"
            raise RuntimeError(msg) from exc
    else:
        msg = f'Unknown file type "{ext}". Must be one of {known_file_types}'
        raise TypeError(msg)

    tomo_stack = TomoStack(stack, tilts=tilts, shifts=shifts)
    return tomo_stack


def load(
    filename: PathLike | list[str] | list[Path],
    tilts: list[float] | np.ndarray | None = None,
    mdocs: list[str] | list[Path] | None = None,
) -> TomoStack:
    """
    Create a TomoStack object using data from a file.

    Parameters
    ----------
    filename
        Name of file that contains data to be read.
        Accepted formats (.MRC, .DM3, .DM4)

    tilts
        List of floats indicating the specimen tilt at each projection (optional).
        If provided, these replace any tilts read from the file(s).

    mdocs
        List of mdoc files for SerialEM data (optional)

    Returns
    -------
    stack : TomoStack
        The resulting TomoStack object

    Group
    -----
    io

    Order
    -----
    1
    """
    if isinstance(filename, (str, Path)):
        if isinstance(filename, str):
            # coerce filename to Path
            filename = Path(filename)
        stack = _load_single_file(filename)
        if tilts is not None:
            stack.tilts = np.asarray(tilts)

    elif isinstance(filename, list):
        first_filename = filename[0]
        if isinstance(first_filename, str):
            first_filename = Path(first_filename)
        ext = first_filename.suffix
        if ext.lower() in dm_file_types:
            signals = cast("list[Signal2D]", hs_load(filename))
            file_tilts = [
                i.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha for i in signals
            ]
            sorted_order = np.argsort(file_tilts)
            file_tilts = np.sort(file_tilts)
            stack = hs_stack(
                [signals[i] for i in sorted_order],
                new_axis_name="Projections",
                show_progressbar=False,
            )
            stack.metadata.General.title = Path(filename[sorted_order[0]]).parent.stem
            stack.axes_manager.navigation_axes[0].units = "degrees"
            stack.axes_manager.navigation_axes[0].scale = np.diff(file_tilts).mean()
            stack.axes_manager.navigation_axes[0].offset = file_tilts[0]
        elif ext.lower() == ".mrc":
            logger.info("Data appears to be a SerialEM multiframe series.")
            if mdocs is not None:
                mdoc_files = mdocs
            else:
                # generate mdoc filenames from mrc filenames
                mrc_files = [Path(p) for p in filename]  # coerce to Path objects
                mdoc_files = [i.with_suffix(".mdoc") for i in mrc_files]
            stack, file_tilts = load_serialem_series(filename, mdoc_files)
        else:
            msg = f'Unknown file type "{ext}". Must be one of {known_file_types}'
            raise TypeError(msg)
        stack = TomoStack(stack, file_tilts if tilts is None else np.asarray(tilts))
    else:
        msg = (
            f"Unknown filename type {type(filename)}. "
            "Must be either a string, Path, or list of either."
        )
        raise TypeError(msg)

    stack.change_data_type("float32")
    if stack.data.min() < 0:
        stack.data -= stack.data.min()
    return stack
