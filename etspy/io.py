"""Data input/output module for ETSpy package."""

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
from hyperspy._signals.signal2d import (
    Signal2D,  # import from _signals for type-checking
)
from hyperspy.axes import UniformDataAxis as Uda
from hyperspy.io import (
    load as hs_load,  # import load function directly for better type-checking
)
from hyperspy.misc.utils import DictionaryTreeBrowser as Dtb
from hyperspy.misc.utils import (
    stack as hs_stack,  # import stack function directly for better type-checking
)

from etspy.base import TomoStack

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
PathLike = Union[str, Path]

# declare known file types for later use
hspy_file_types = [".hdf5", ".h5", ".hspy"]
mrc_file_types = [".mrc", ".ali", ".rec"]
dm_file_types = [".dm3", ".dm4"]
known_file_types = hspy_file_types + mrc_file_types + dm_file_types


class MismatchedTiltError(ValueError):
    """
    Error for when number of tilts in signal does not match tilt dimension.

    Group
    -----
    io
    """

    def __init__(self, num_tilts, tilt_dimension):
        """Create a MismatchedTiltError."""
        super().__init__(
            f"Number of tilts ({num_tilts}) does not match "
            f"the tilt dimension of the data array ({tilt_dimension})",
        )


def create_stack(
    stack: Union[Signal2D, np.ndarray],
    tilts: Optional[np.ndarray] = None,
) -> TomoStack:
    """
    Create a TomoStack from existing in-memory tilt series data.

    Parameters
    ----------
    stack
        Tilt series data (ntilts, ny, nx)
    tilts
        An (optional) array defining the tilt angles

    Returns
    -------
    stack: TomoStack
        A TomoStack instance containing the provided data

    Group
    -----
    io

    Order
    -----
    2
    """
    if isinstance(stack, Signal2D):
        ntilts = stack.data.shape[0]
        if tilts is None:
            tilts = np.zeros(ntilts)
        if ntilts != tilts.shape[0]:
            raise MismatchedTiltError(tilts.shape[0], ntilts)

        if stack.metadata.has_item("Tomography"):
            pass
        else:
            ntilts = stack.data.shape[0]
            if tilts is None:
                tilts = np.zeros(ntilts)
            tomo_metadata = {
                "cropped": False,
                "shifts": np.zeros([ntilts, 2]),
                "tiltaxis": 0,
                "tilts": tilts,
                "xshift": 0,
                "yshift": 0,
            }
            stack.metadata.add_node("Tomography")
            # cast for type-checking:
            tomo_meta_node = cast(Dtb, stack.metadata.Tomography)
            tomo_meta_node.add_dictionary(tomo_metadata)
        axes_list = [x for _, x in sorted(stack.axes_manager.as_dictionary().items())]
        metadata_dict = stack.metadata.as_dictionary()
        original_metadata_dict = stack.original_metadata.as_dictionary()
        stack = TomoStack(
            stack,
            axes=axes_list,
            metadata=metadata_dict,
            original_metadata=original_metadata_dict,
        )
    elif isinstance(stack, np.ndarray):
        ntilts = stack.shape[0]
        if tilts is None:
            tilts = np.zeros(ntilts)
        if ntilts != tilts.shape[0]:
            raise MismatchedTiltError(tilts.shape[0], ntilts)

        tomo_metadata = {
            "cropped": False,
            "shifts": np.zeros([ntilts, 2]),
            "tiltaxis": 0,
            "tilts": tilts,
            "xshift": 0,
            "yshift": 0,
        }
        stack = TomoStack(stack)
        stack.metadata.add_node("Tomography")
        cast(Dtb, stack.metadata.Tomography).add_dictionary(tomo_metadata)
    stack = cast(TomoStack, stack)  # type-checking cast
    cast(Uda, stack.axes_manager[0]).name = "Tilt"
    cast(Uda, stack.axes_manager[0]).units = "degrees"
    cast(Uda, stack.axes_manager[1]).name = "x"
    cast(Uda, stack.axes_manager[2]).name = "y"
    if tilts is None:
        logger.info("Unable to find tilt angles. Calibrate axis 0.")
    return stack


def get_mrc_tilts(
    stack: Union[Signal2D, TomoStack],
    filename: PathLike,
) -> Optional[np.ndarray]:
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
        fei_header = cast(Dtb, stack.original_metadata.fei_header)
        if fei_header.has_item("a_tilt"):
            tilts = fei_header["a_tilt"][0 : stack.data.shape[0]]
            logger.info("Tilts found in MRC file header")
    elif stack.original_metadata.has_item("std_header"):
        logger.info("SerialEM generated MRC file detected")
        ext_header = parse_mrc_header(filename)["ext_header"]
        tilts = ext_header[np.arange(0, int(ext_header.shape[0]), 7)][
            0 : stack.data.shape[0]
        ]
        tilts = tilts / 100
    elif tiltfile.is_file():
        tilts = np.loadtxt(tiltfile)
        logger.info(".rawtlt file detected.")
        if len(tilts) == stack.data.shape[0]:
            logger.info("Tilts loaded from .rawtlt file")
        else:
            msg = "Number of tilts in .rawtlt file inconsistent with data shape"
            raise ValueError(msg)
    return tilts


def get_dm_tilts(s: Union[Signal2D, TomoStack]) -> np.ndarray:
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

    return np.arange(mintilt, maxtilt + tiltstep, tiltstep)


def parse_mdoc(
    mdoc_file: PathLike,
    series: bool = False,
) -> Tuple[dict, Union[np.ndarray, float]]:
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
    tilt = np.array([])
    if isinstance(mdoc_file, str):
        mdoc_file = Path(mdoc_file)
    with mdoc_file.open("r") as f:
        lines = f.readlines()
        for i in range(35):
            for k in keys:
                if k in lines[i]:
                    if k == "ImageFile":
                        metadata[k] = lines[i].split("=")[1].strip()
                    else:
                        metadata[k] = float(lines[i].split("=")[1].strip())
    if series:
        for i in range(35):
            if "TiltAngle" in lines[i]:
                tilt = float(lines[i].split("=")[1].strip())
    else:
        tilt = []
        for i in lines:
            if "TiltAngle" in i:
                tilt.append(float(i.split("=")[1].strip()))
        tilt = np.array(tilt)
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
    mrc_logger = logging.getLogger("hyperspy.io_plugins.mrc")
    log_level = mrc_logger.getEffectiveLevel()
    mrc_logger.setLevel(logging.ERROR)

    meta, _ = parse_mdoc(mdocfile)
    stack = hs_load(mrcfile)

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
    mrc_logger.setLevel(log_level)
    return stack


def load_serialem_series(
    mrcfiles: Union[List[str], List[Path]],
    mdocfiles: Union[List[str], List[Path]],
) -> Tuple[TomoStack, np.ndarray]:
    """
    Load a multi-frame series collected by SerialEM.

    Parameters
    ----------
    mrcfiles
        List of MRC file paths containing multi-frame tilt series data.

    mdocfiles
        List of SerialEM metadata file paths for multi-frame tilt series data.

    Returns
    -------
    stack : :py:class:`~etspy.base.TomoStack`
        Tilt series resulting by averaging frames at each tilt
    tilts : :py:class:`~numpy.ndarray`
        The tilt values for each image in the stack

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
        stack.append(hs_load(mrc_filename))

    images_per_tilt = stack[0].data.shape[0]
    stack = hs_stack(stack, show_progressbar=False)

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
    mrc_logger.setLevel(log_level)
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
    with filename.open("r") as h:
        header["nx"], header["ny"], header["nz"] = np.fromfile(h, np.uint32, 3)
        header["mode"] = np.fromfile(h, np.uint32, 1)[0]
        header["nxstart"], header["nystart"], header["nzstart"] = np.fromfile(
            h,
            np.uint32,
            3,
        )
        header["mx"], header["my"], header["mz"] = np.fromfile(h, np.uint32, 3)
        header["xlen"], header["ylen"], header["zlen"] = np.fromfile(h, np.uint32, 3)
        _ = np.fromfile(h, np.uint32, 6)
        header["amin"], header["amax"], header["amean"] = np.fromfile(h, np.uint32, 3)
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


def _load_single_file(filename: Path) -> Tuple[Signal2D, Optional[np.ndarray]]:
    """Load a HyperSpy signal and any tilts from a single file."""
    ext = filename.suffix
    tilts = None
    if ext.lower() in hspy_file_types:
        stack = hs_load(filename, reader="HSPY")
        if stack.metadata.has_item("Tomography"):
            tilts = stack.metadata.Tomography.tilts
    elif ext.lower() in dm_file_types:
        stack = hs_load(filename)
        stack.change_dtype(np.float32)
        tilts = get_dm_tilts(stack)
    elif ext.lower() in mrc_file_types:
        try:
            stack = hs_load(filename, reader="mrc")
            tilts = get_mrc_tilts(stack, filename)
        except TypeError as exc:
            msg = "Unable to read MRC with Hyperspy"
            raise RuntimeError(msg) from exc
    else:
        msg = f'Unknown file type "{ext}". Must be one of {known_file_types}'
        raise TypeError(msg)

    return stack, tilts


def load(
    filename: Union[PathLike, List[str], List[Path]],
    tilts: Optional[Union[List[float], np.ndarray]] = None,
    mdocs: Optional[Union[List[str], List[Path]]] = None,
) -> TomoStack:
    """
    Create a TomoStack object using data from a file.

    Parameters
    ----------
    filename
        Name of file that contains data to be read.
        Accepted formats (.MRC, .DM3, .DM4)

    tilts
        List of floats indicating the specimen tilt at each projection (optional)

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
        stack, tilts = _load_single_file(filename)

    elif isinstance(filename, list):
        first_filename = filename[0]
        if isinstance(first_filename, str):
            first_filename = Path(first_filename)
        ext = first_filename.suffix
        if ext.lower() in dm_file_types:
            s = hs_load(filename)
            tilts = [i.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha for i in s]
            sorted_order = np.argsort(tilts)
            tilts = np.sort(tilts)
            files_sorted = list(np.array(filename)[sorted_order])
            del s
            stack = hs_load(files_sorted, stack=True)
        elif ext.lower() == ".mrc":
            logger.info("Data appears to be a SerialEM multiframe series.")
            if mdocs is not None:
                mdoc_files = mdocs
            else:
                # generate mdoc filenames from mrc filenames
                mrc_files = [Path(p) for p in filename]  # coerce to Path objects
                mdoc_files = [i.with_suffix(".mdoc") for i in mrc_files]
            stack, tilts = load_serialem_series(filename, mdoc_files)
        else:
            msg = f'Unknown file type "{ext}". Must be one of {known_file_types}'
            raise TypeError(msg)
    else:
        msg = (
            f"Unknown filename type {type(filename)}. "
            "Must be either a string, Path, or list of either."
        )
        raise TypeError(msg)

    stack = create_stack(stack, tilts)
    stack.change_data_type("float32")
    if stack.data.min() < 0:
        stack.data -= stack.data.min()
    return stack
