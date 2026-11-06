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
PathLike = str | Path

# declare known file types for later use
hspy_file_types = [".hdf5", ".h5", ".hspy"]
mrc_file_types = [".mrc", ".ali", ".rec"]
dm_file_types = [".dm3", ".dm4"]
known_file_types = hspy_file_types + mrc_file_types + dm_file_types

# int16 data with more than this fraction of values below this threshold are
# likely unsigned values stored with an offset of -32768
INT16_OFFSET_THRESHOLD = -16384
INT16_OFFSET_FRACTION = 0.5

# maximum difference (in counts) between the offsets implied by the FEI header's
# mean intensity of each image for them to be considered a constant offset
FEI_MEAN_TOLERANCE = 0.1


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
    fei_header = cast("Dtb | None", stack.original_metadata.get_item("fei_header"))
    if fei_header is not None and fei_header.has_item("a_tilt"):
        tilts = fei_header["a_tilt"][0 : stack.data.shape[0]]
        logger.info("Tilts found in MRC file header")
    else:
        header = parse_mrc_header(filename)
        if header["ext_type"] == "SERI":
            logger.info("SerialEM generated MRC file detected")
            tilts = _get_serialem_tilts(header, stack.data.shape[0])

    # fall back to a .rawtlt file if the header did not contain tilts
    if tilts is None and tiltfile.is_file():
        tilts = np.loadtxt(tiltfile)
        logger.info(".rawtlt file detected.")
        if len(tilts) == stack.data.shape[0]:
            logger.info("Tilts loaded from .rawtlt file")
        else:
            msg = "Number of tilts in .rawtlt file inconsistent with data shape"
            raise ValueError(msg)
    return tilts


def _get_serialem_tilts(header: dict[str, Any], nimages: int) -> np.ndarray | None:
    """
    Extract tilts from a SerialEM ('SERI') MRC extended header.

    Each section's entry in the extended header is ``nint`` bytes long, and
    ``nreal`` holds bit flags for which values are present. When bit 1 is set,
    the first value of each entry is the tilt angle in degrees * 100, stored as
    a signed 16-bit integer. See https://bio3d.colorado.edu/imod/doc/mrc_format.txt

    Returns
    -------
    tilts : :py:class:`~numpy.ndarray` or None
        Tilt angles for (at most) the first ``nimages`` sections, or ``None`` if
        the extended header does not contain tilt angles
    """
    nint, nreal = int(header["nint"]), int(header["nreal"])
    min_entry_bytes = 2
    if not nreal & 1 or nint < min_entry_bytes:
        logger.info("SerialEM extended header does not contain tilt angles")
        return None
    ext_header = header["ext_header"]  # int16 values
    stride = nint // 2
    nsections = min(nimages, ext_header.shape[0] // stride)
    return ext_header[0 : nsections * stride : stride] / 100


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

    # integer-indexed so that floating point error cannot add or drop a tilt
    # (as np.arange with a float step can)
    ntilts = round((maxtilt - mintilt) / tiltstep) + 1
    return mintilt + tiltstep * np.arange(ntilts)


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
        "ImageSize",
        "DataMode",
        "Magnification",
        "ExposureTime",
        "SpotSize",
        "Defocus",
    ]
    if isinstance(mdoc_file, str):
        mdoc_file = Path(mdoc_file)

    # Values are stored as "key = value" lines, first in a global header and then
    # in one "[ZValue = n]" section per image. Other bracketed lines are titles.
    global_values: dict[str, str] = {}
    sections: list[dict[str, str]] = []
    with mdoc_file.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("[ZValue"):
                sections.append({})
            elif line and not line.startswith("[") and "=" in line:
                key, value = (part.strip() for part in line.split("=", 1))
                (sections[-1] if sections else global_values)[key] = value

    metadata = {}
    first_section = sections[0] if sections else {}
    for k in keys:
        value = global_values.get(k, first_section.get(k))
        if value is None:
            continue
        if k == "ImageFile":
            metadata[k] = value
        elif k == "ImageSize":
            metadata[k] = [int(v) for v in value.split()]
        else:
            metadata[k] = float(value)

    tilts = [float(sec["TiltAngle"]) for sec in sections if "TiltAngle" in sec]
    # a single-tilt (series) mdoc returns its one tilt value as a float
    tilt = tilts[0] if (series and tilts) else np.array(tilts)
    return metadata, tilt


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
    meta, _ = parse_mdoc(mdocfile)
    stack = hs_load(mrcfile)
    stack = cast("Signal2D", stack)

    stack.axes_manager[1].scale = stack.axes_manager[1].scale / 10
    stack.axes_manager[2].scale = stack.axes_manager[2].scale / 10
    stack.axes_manager[1].units = "nm"
    stack.axes_manager[2].units = "nm"

    if not stack.metadata.has_item("Acquisition_instrument.TEM"):
        stack.metadata.add_node("Acquisition_instrument.TEM")
    stack.metadata.Acquisition_instrument.TEM.magnification = meta["Magnification"]
    stack.metadata.Acquisition_instrument.TEM.beam_energy = meta["Voltage"]
    stack.metadata.Acquisition_instrument.TEM.dwell_time = meta["ExposureTime"] / (
        stack.data.shape[1] * stack.data.shape[2]
    )
    stack.metadata.Acquisition_instrument.TEM.spot_size = meta["SpotSize"]
    stack.metadata.Acquisition_instrument.TEM.defocus = meta["Defocus"]
    stack.metadata.General.original_filename = meta["ImageFile"]
    logger.info("SerialEM stack successfully loaded. ")
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

    images_per_tilt = stack.axes_manager["Frames"].size
    if not stack.metadata.has_item("Acquisition_instrument.TEM"):
        stack.metadata.add_node("Acquisition_instrument.TEM")
    stack.metadata.Acquisition_instrument.TEM.magnification = meta[0]["Magnification"]
    stack.metadata.Acquisition_instrument.TEM.beam_energy = meta[0]["Voltage"]
    stack.metadata.Acquisition_instrument.TEM.dwell_time = (
        meta[0]["ExposureTime"]
        * images_per_tilt
        / (stack.data.shape[2] * stack.data.shape[3])
    )
    stack.metadata.Acquisition_instrument.TEM.spot_size = meta[0]["SpotSize"]
    stack.metadata.Acquisition_instrument.TEM.defocus = meta[0]["Defocus"]
    stack.metadata.General.original_filename = meta[0]["ImageFile"]
    logger.info(
        "SerialEM Multiframe stack successfully loaded. "
        "Use etspy.utils.register_serialem_stack to align frames.",
    )
    return stack, tilts


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
        header["nx"], header["ny"], header["nz"] = np.fromfile(h, np.int32, 3)
        header["mode"] = np.fromfile(h, np.int32, 1)[0]
        header["nxstart"], header["nystart"], header["nzstart"] = np.fromfile(
            h,
            np.int32,
            3,
        )
        header["mx"], header["my"], header["mz"] = np.fromfile(h, np.int32, 3)
        header["xlen"], header["ylen"], header["zlen"] = np.fromfile(h, np.float32, 3)
        _ = np.fromfile(h, np.uint32, 6)
        header["amin"], header["amax"], header["amean"] = np.fromfile(h, np.float32, 3)
        _ = np.fromfile(h, np.uint32, 1)
        header["nextra"] = np.fromfile(h, np.int32, 1)[0]
        _ = np.fromfile(h, np.uint16, 1)[0]
        _ = np.fromfile(h, np.uint8, 6)
        strbits = np.fromfile(h, np.uint8, 4)
        header["ext_type"] = (
            bytes(strbits).decode("ascii", errors="ignore").strip("\x00")
        )
        header["nversion"] = np.fromfile(h, np.int32, 1)[0]
        _ = np.fromfile(h, np.uint8, 16)
        header["nint"] = np.fromfile(h, np.int16, 1)[0]
        header["nreal"] = np.fromfile(h, np.int16, 1)[0]
        _ = np.fromfile(h, np.int8, 20)
        header["imodStamp"] = np.fromfile(h, np.int32, 1)[0]
        header["imodFlags"] = np.fromfile(h, np.int32, 1)[0]
        header["idtype"] = np.fromfile(h, np.int16, 1)[0]
        header["lens"] = np.fromfile(h, np.int16, 1)[0]
        header["nd1"], header["nd2"], header["vd1"], header["vd2"] = np.fromfile(
            h,
            np.int16,
            4,
        )
        _ = np.fromfile(h, np.float32, 6)
        header["xorg"], header["yorg"], header["zorg"] = np.fromfile(h, np.float32, 3)
        strbits = np.fromfile(h, np.uint8, 4)
        header["cmap"] = bytes(strbits).decode("ascii", errors="ignore").strip("\x00")
        header["stamp"] = np.fromfile(h, np.int8, 4)
        header["rms"] = np.fromfile(h, np.float32, 1)[0]
        header["nlabl"] = np.fromfile(h, np.int32, 1)[0]
        strbits = np.fromfile(h, np.uint8, 800)
        header["text"] = bytes(strbits).decode("ascii", errors="ignore").strip("\x00")
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


def _apply_user_tilts(stack: TomoStack, tilts: list[float] | np.ndarray):
    """Set user-provided tilts on a loaded stack, overriding any from the file."""
    tilts = np.asarray(tilts, dtype=float)
    if tilts.shape[0] != stack.data.shape[0]:
        msg = (
            f"Number of tilts provided ({tilts.shape[0]}) does not match the "
            f"number of projections in the data ({stack.data.shape[0]})"
        )
        raise ValueError(msg)
    if tilts.ndim == 1 and stack.axes_manager.navigation_dimension == 2:  # noqa: PLR2004
        # multiframe stack: repeat each tilt for every frame at that tilt
        nframes = stack.data.shape[1]
        tilts = np.tile(tilts[:, np.newaxis], (1, nframes))

    # Warn if the file contained tilts that differ from those provided (tilts
    # that were not found in the file are stored as all zeros)
    file_tilts = stack.tilts.data
    if (
        np.any(file_tilts != 0)
        and file_tilts.size == tilts.size
        and not np.allclose(file_tilts.reshape(tilts.shape), tilts)
    ):
        logger.warning(
            "Overriding tilts found in file (%.1f to %.1f degrees, %d values) with "
            "user-provided tilts (%.1f to %.1f degrees); maximum difference is "
            "%.2f degrees",
            file_tilts.min(),
            file_tilts.max(),
            file_tilts.shape[0],
            tilts.min(),
            tilts.max(),
            np.abs(file_tilts.reshape(tilts.shape) - tilts).max(),
        )
    else:
        logger.info("Using user-provided tilts")
    stack.tilts = tilts


def load(
    filename: PathLike | list[str] | list[Path],
    tilts: list[float] | np.ndarray | None = None,
    mdocs: list[str] | list[Path] | None = None,
    fei_offset_correction: bool = True,
) -> TomoStack:
    """
    Create a TomoStack object using data from a file.

    Parameters
    ----------
    filename
        Name of file that contains data to be read.
        Accepted formats (.MRC, .DM3, .DM4)

    tilts
        Tilt angle (in degrees) of each projection (optional). If provided, these
        values override any tilts read from the file(s). Must contain one value per
        projection.

    mdocs
        List of mdoc files for SerialEM data (optional)

    fei_offset_correction
        Some FEI software stores unsigned 16-bit detector counts as signed
        integers by subtracting 32768. If True (default), and the FEI extended
        header of an MRC file shows that the stored values are offset from the
        acquired values (the mean intensity it records for every image differs
        from the stored mean by the same constant), that constant is added back.
        The correction is logged, and recorded in
        ``metadata.Tomography.intensity_offset``.

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

    elif isinstance(filename, list):
        first_filename = filename[0]
        if isinstance(first_filename, str):
            first_filename = Path(first_filename)
        ext = first_filename.suffix
        if ext.lower() in dm_file_types:
            s = hs_load(filename)
            s = cast("Signal2D", s)
            file_tilts = [
                i.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha for i in s
            ]
            sorted_order = np.argsort(file_tilts)
            file_tilts = np.sort(file_tilts)
            files_sorted = list(np.array(filename)[sorted_order])
            del s
            stack = hs_load(files_sorted, stack=True, new_axis_name="Projections")
            stack = cast("Signal2D", stack)
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
        stack = TomoStack(stack, file_tilts)
    else:
        msg = (
            f"Unknown filename type {type(filename)}. "
            "Must be either a string, Path, or list of either."
        )
        raise TypeError(msg)

    _finalize_loaded_stack(stack, tilts, fei_offset_correction)
    return stack


def _finalize_loaded_stack(
    stack: TomoStack,
    tilts: list[float] | np.ndarray | None,
    fei_offset_correction: bool,
):
    """Apply tilts and intensity corrections, and check a newly loaded stack."""
    if tilts is not None:
        _apply_user_tilts(stack, tilts)
    elif np.all(stack.tilts.data == 0):
        logger.warning(
            "No tilt angles were found for the loaded data (all tilts are zero). "
            "Set them with stack.set_tilts(start, increment) or by passing tilts "
            "to load() before alignment or reconstruction.",
        )

    if fei_offset_correction:
        _correct_fei_int16_offset(stack)
    _warn_if_offset_int16(stack)
    stack.change_data_type("float32")


def _correct_fei_int16_offset(stack: TomoStack):
    """
    Restore intensities that FEI software stored as signed integers with an offset.

    The FEI extended header records the mean intensity of each image as acquired
    (``mean_int``). If, for every image, this differs from the mean of the stored
    values by the same nonzero constant (e.g. 32768, when unsigned 16-bit counts
    were stored as signed integers), the constant is added to the data.
    """
    if stack.data.dtype != np.int16:
        return
    fei_header = stack.original_metadata.get_item("fei_header")
    if fei_header is None or not fei_header.has_item("mean_int"):
        return
    images = stack.data.reshape(-1, *stack.data.shape[-2:])
    header_means = np.asarray(fei_header["mean_int"], dtype=float).ravel()
    if header_means.shape[0] < images.shape[0]:
        return
    stored_means = images.reshape(images.shape[0], -1).mean(axis=1)
    differences = header_means[: images.shape[0]] - stored_means
    offset = round(float(np.median(differences)))
    if offset == 0 or np.abs(differences - offset).max() > FEI_MEAN_TOLERANCE:
        return
    stack.data = stack.data.astype(np.float32) + np.float32(offset)
    stack.metadata.set_item("Tomography.intensity_offset", offset)
    logger.warning(
        "The FEI extended header shows that the stored intensities are offset by "
        "%d from the acquired values (as when unsigned 16-bit counts are stored "
        "as signed integers), so %d was added to restore them. Use "
        "load(..., fei_offset_correction=False) to keep the stored values.",
        -offset,
        offset,
    )


def _warn_if_offset_int16(stack: TomoStack):
    """
    Warn if signed 16-bit data appear to be unsigned values stored with an offset.

    Some software stores unsigned 16-bit detector counts as signed integers by
    subtracting 32768, so that most of the stored values (including the
    background) are large and negative. The values are not changed, since the
    offset is removed along with the rest of the background by
    :py:meth:`~etspy.base.TomoStack.remove_background`.
    """
    if stack.data.dtype != np.int16:
        return
    fraction_low = float(np.mean(stack.data < INT16_OFFSET_THRESHOLD))
    if fraction_low > INT16_OFFSET_FRACTION:
        logger.warning(
            "The data are stored as signed 16-bit integers, but %.0f%% of the values "
            "are below -16384, so they may be unsigned values stored with an "
            "offset of -32768. The values have not been changed. For dark-field "
            "data, use stack.remove_background() to remove the offset along with "
            "the background; otherwise add 32768 to recover the original values.",
            100 * fraction_low,
        )
