# -*- coding: utf-8 -*-
#
# This file is part of TomoTools

"""
Data input/output module for TomoTools package.

@author: Andrew Herzing
"""

import numpy as np
import os
import hyperspy.api as hspy
from tomotools.base import TomoStack
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_to_tomo_stack(data, tilts=None, manual_tilts=False):
    """
    Create a TomoStack object from other data structures.

    Parameters
    ----------
    data : Numpy array, Hyperspy Signal2D
        Tilt series data to be converted.  If NumPy array, the first
        dimension must represent the tilt dimension. If Hyperspy Signal2D, the
        the tilt dimension must be the navigation axis.

    tilts : None, Numpy array, or Hyperpsy Signal1D
        Array containing the tilt values in degrees for each projection.
        If provided, these values will be stored in
        stack.metadata.Tomography.tilts

    manual_tilts : bool
        If True, prompt for input of maximum positive tilt, maximum negative
        tilt, and tilt increment in order to population stack.metadata.Tomography.tilts


    Returns
    -------
    tomo_stack_signal : TomoStack object


    Examples
    --------
    >>> import numpy as np
    >>> from tomotools.io import convert_to_tomo_stack
    >>> s = np.random.random((50, 500,500))
    >>> s_new = convert_to_tomo_stack(s)
    >>> s_new
    <TomoStack, title: , dimensions: (50|500, 500)>

    >>> import numpy as np
    >>> from tomotools.io import convert_to_tomo_stack
    >>> import hyperspy.api as hs
    >>> s = hs.signals.Signal2D(np.random.random((50, 500,500)))
    >>> s_new = convert_to_tomo_stack(s)
    >>> s_new
    <TomoStack, title: , dimensions: (50|500, 500)>

    """
    def _set_axes(s):
        s.axes_manager[0].name = 'Tilt'
        s.axes_manager[0].units = 'degrees'
        s.axes_manager[1].name = 'x'
        s.axes_manager[2].name = 'y'
        return s

    def _set_tomo_metadata(s):
        tomo_metadata = {"cropped": False,
                         "shifts": np.zeros([s.data.shape[0], 2]),
                         "tiltaxis": 0,
                         "tilts": np.zeros(s.data.shape[0]),
                         "xshift": 0,
                         "yshift": 0}
        s.metadata.add_node("Tomography")
        s.metadata.Tomography.add_dictionary(tomo_metadata)
        return s

    def _get_manual_tilts():
        negtilt = eval(input('Enter maximum negative tilt: '))
        postilt = eval(input('Enter maximum positive tilt: '))
        tiltstep = eval(input('Enter tilt step: '))
        tilts = np.arange(negtilt, postilt + tiltstep, tiltstep)
        return tilts

    if type(data) is np.ndarray:
        stack = hspy.signals.Signal2D(data)
    elif type(data) is hspy.signals.Signal2D:
        stack = data.deepcopy()
    else:
        raise TypeError("Unsupported data type. Must be either"
              "NumPy Array or Hyperspy Signal2D")

    stack = _set_tomo_metadata(stack)

    if type(tilts) is np.ndarray:
        pass
    elif type(tilts) is hspy.signals.Signal1D:
        tilts = tilts.data
    elif manual_tilts:
        tilts = _get_manual_tilts()
    else:
        tilts = np.zeros(stack.data.shape[0])
        logger.info('Tilts are not defined. Please add tilts to Tomography metadata.')

    if tilts.shape[0] != stack.data.shape[0]:
        raise ValueError("Number of tilts is not consistent with data shape."
                         "%s does not equal %s" % (tilts.shape[0], stack.data.shape[0]))

    stack = _set_axes(stack)
    stack.metadata.Tomography.tilts = tilts
    stack.axes_manager[0].offset = tilts[0]
    stack.axes_manager[0].scale = tilts[1] - tilts[0]

    axes_list = [x for _,
                 x in sorted(stack.axes_manager.as_dictionary().items())]
    metadata_dict = stack.metadata.as_dictionary()
    original_metadata_dict = stack.original_metadata.as_dictionary()
    stack = TomoStack(stack, axes=axes_list, metadata=metadata_dict,
                      original_metadata=original_metadata_dict)
    return stack


def load_hspy(filename, tilts=None, reader=None):
    """
    Read an MRC file to a TomoStack object using the Hyperspy reader.

    Parameters
    ----------
    filename : string
        Name of file that contains data to be read.

    tilts : list or NumPy array
        List of floats indicating the specimen tilt at each projection

    reader : str
        Hyperspy file reader to use

    Returns
    ----------
    stack : TomoStack object

    """
    def _set_axes(s):
        s.axes_manager[0].name = 'Tilt'
        s.axes_manager[0].units = 'degrees'
        s.axes_manager[1].name = 'x'
        s.axes_manager[2].name = 'y'
        return s

    def _set_tomo_metadata(s):
        tomo_metadata = {"cropped": False,
                         "shifts": np.zeros([s.data.shape[0], 2]),
                         "tiltaxis": 0,
                         "tilts": np.zeros(s.data.shape[0]),
                         "xshift": 0,
                         "yshift": 0}
        s.metadata.add_node("Tomography")
        s.metadata.Tomography.add_dictionary(tomo_metadata)
        return s

    # serialem_format = False
    stack = hspy.load(filename, reader=reader)
    stack = _set_axes(stack)
    if not stack.metadata.has_item("Tomography"):
        stack = _set_tomo_metadata(stack)
    ext = os.path.splitext(filename)[1]
    if ext.lower() in ['.mrc', '.ali', '.rec']:
        tiltfile = os.path.splitext(filename)[0] + '.rawtlt'
        if stack.original_metadata.has_item('fei_header'):
            if stack.original_metadata.fei_header.has_item('a_tilt'):
                tilts = stack.original_metadata.fei_header['a_tilt'][0:stack.data.shape[0]]
                logger.info('Tilts found in MRC file header')
            elif os.path.isfile(tiltfile):
                tilts = np.loadtxt(tiltfile)
                logger.info('.rawtlt file detected.')
                if len(tilts) == stack.data.shape[0]:
                    logger.info('Tilts loaded from .rawtlt file')
                else:
                    logger.info('Number of tilts in .rawtlt file inconsistent'
                                ' with data shape')
        elif stack.original_metadata.has_item('std_header'):
            # serialem_format = True
            logger.info('SerialEM generated MRC file detected')
        else:
            tilts = np.zeros(stack.data.shape[0])
            logger.info('Unable to find tilt angles. Calibrate axis 0.')
    elif ext.lower() in ['.hdf5', '.hd5', '.hspy']:
        tilts = stack.metadata.Tomography.tilts
    else:
        raise ValueError('Cannot read file type: %s' % ext)

    stack.axes_manager[0].scale = tilts[1] - tilts[0]
    stack.axes_manager[0].offset = tilts[0]
    stack.metadata.Tomography.tilts = tilts

    if stack.data.min() < 0:
        stack.data = np.float32(stack.data)
        stack.data += np.abs(stack.data.min())

    axes_list = [x for _,
                 x in sorted(stack.axes_manager.as_dictionary().items())]
    metadata_dict = stack.metadata.as_dictionary()
    original_metadata_dict = stack.original_metadata.as_dictionary()
    stack = TomoStack(stack, axes=axes_list, metadata=metadata_dict,
                      original_metadata=original_metadata_dict)
    return stack


def load_dm(filename):
    """
    Read DM image series.

    Read series of images in a single DM3 file to a TomoStack object
    using the Hyperspy reader.

    Parameters
    ----------
    filename : string
        Name of DM3 file that contains data to be read.

    Returns
    ----------
    stack : TomoStack object

    """
    s = hspy.load(filename)
    s.change_dtype(np.float32)
    maxtilt = (s.original_metadata['ImageList']
               ['TagGroup0']
               ['ImageTags']
               ['Tomography']
               ['Tomography_setup']
               ['Tilt_angles']
               ['Maximum_tilt_angle_deg'])

    mintilt = (s.original_metadata['ImageList']
               ['TagGroup0']
               ['ImageTags']
               ['Tomography']
               ['Tomography_setup']
               ['Tilt_angles']
               ['Minimum_tilt_angle_deg'])

    tiltstep = (s.original_metadata['ImageList']
                ['TagGroup0']
                ['ImageTags']
                ['Tomography']
                ['Tomography_setup']
                ['Tilt_angles']
                ['Tilt_angle_step_deg'])

    tilts = np.arange(mintilt, maxtilt + tiltstep, tiltstep)

    s = convert_to_tomo_stack(s, tilts)
    return s


def load_dm_series(files):
    """
    Load a series of individual DM3/DM4 files as a TomoStack object.

    Parameters
    ----------
    input_data : string or list of files
        Path to image series data or a list of files.

    Returns
    ----------
    stack : TomoStack object

    """
    s = hspy.load(files)
    tilts = [i.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha for i in s]
    sorted_order = np.argsort(tilts)
    tilts_sorted = np.sort(tilts)
    files_sorted = list(np.array(files)[sorted_order])
    del s
    stack = hspy.load(files_sorted, stack=True)
    stack = convert_to_tomo_stack(stack, tilts_sorted)
    return stack


def load_serialem_series(mrcfiles, mdocfiles):
    """
    Load a multi-frame series collected by SerialEM.

    Parameters
    ----------
    mrc_files : list of files
        MRC files containing multi-frame tilt series data.

    mdoc_files : list of files
        SerialEM metadata files for multi-frame tilt series data.

    Returns
    ----------
    stack : TomoStack object
        Tilt series resulting by averaging frames at each tilt

    """

    def _parse_mdoc(mdoc_file):
        """
        Parse a SerialEM mdoc file.

        Args
        ----------
        mdoc_file : str
            Name of the mdoc file to parse

        Returns
        ----------
        metadata : dict
            Dictionary with values parsed from mdoc file

        """
        keys = ['PixelSpacing', 'Voltage', 'ImageFile', 'Image Size', 'DataMode',
                'TiltAngle', 'Magnification', 'ExposureTime', 'SpotSize', 'Defocus']
        metadata = {}
        with open(mdoc_file, 'r') as f:
            for i in range(0, 35):
                line = f.readline()
                for k in keys:
                    if k in line:
                        if k == 'ImageFile':
                            metadata[k] = line.split('=')[1].strip()
                        else:
                            metadata[k] = float(line.split('=')[1].strip())
        return metadata

    def _set_axes_serialem(s, tilts, meta):
        s.axes_manager[0].scale = 1
        s.axes_manager[0].offset = 0
        s.axes_manager[0].name = 'multiframe'
        s.axes_manager[0].units = 'scan'
        s.axes_manager[1].scale = tilts[1] - tilts[0]
        s.axes_manager[1].offset = tilts[0]
        s.axes_manager[1].units = 'degrees'
        s.axes_manager[1].name = 'Tilt'
        s.axes_manager[2].scale = meta[0]['PixelSpacing'] / 10
        s.axes_manager[3].scale = meta[0]['PixelSpacing'] / 10
        s.axes_manager[2].units = 'nm'
        s.axes_manager[3].units = 'nm'
        s.axes_manager[2].name = 'y'
        s.axes_manager[3].name = 'x'
        return s

    def _set_tomo_metadata_serialem(s):
        tomo_metadata = {"cropped": False,
                         "shifts": np.zeros([s.data.shape[0], 2]),
                         "tiltaxis": 0,
                         "tilts": np.zeros(s.data.shape[0]),
                         "xshift": 0,
                         "yshift": 0}
        s.metadata.add_node("Tomography")
        s.metadata.Tomography.add_dictionary(tomo_metadata)
        return s

    mrc_logger = logging.getLogger("hyperspy.io_plugins.mrc")
    log_level = mrc_logger.getEffectiveLevel()
    mrc_logger.setLevel(logging.ERROR)

    stack = [None] * len(mrcfiles)
    meta = [None] * len(mdocfiles)
    for i in range(0, len(mdocfiles)):
        meta[i] = _parse_mdoc(mdocfiles[i])

    tilts = np.array([meta[i]['TiltAngle'] for i in range(0, len(meta))])
    tilts_sort = np.argsort(tilts)
    tilts.sort()

    for i in range(0, len(mrcfiles)):
        fn = mdocfiles[tilts_sort[i]][:-5]
        if fn[-3:].lower() != 'mrc':
            fn = fn + 'mrc'
        stack[i] = hspy.load(fn)

    images_per_tilt = stack[0].data.shape[0]
    stack = hspy.stack(stack)
    stack = _set_axes_serialem(stack, tilts, meta)

    if not stack.metadata.has_item('Acquisition_instrument.TEM'):
        stack.metadata.add_node('Acquisition_instrument.TEM')
    stack.metadata.Acquisition_instrument.TEM.magnification = meta[0]['Magnification']
    stack.metadata.Acquisition_instrument.TEM.beam_energy = meta[0]['Voltage']
    stack.metadata.Acquisition_instrument.TEM.dwell_time = meta[
        0]['ExposureTime'] * images_per_tilt / (stack.data.shape[2] * stack.data.shape[3])
    stack.metadata.Acquisition_instrument.TEM.spot_size = meta[0]['SpotSize']
    stack.metadata.Acquisition_instrument.TEM.defocus = meta[0]['Defocus']
    stack.metadata.General.original_filename = meta[0]['ImageFile']
    stack = _set_tomo_metadata_serialem(stack)
    stack.metadata.Tomography.tilts = tilts
    logger.info('SerialEM Multiframe stack successfully loaded. '
                'Use tomotools.utils.register_serialem_stack to align frames.')
    mrc_logger.setLevel(log_level)
    return stack


def parse_mrc_header(filename):
    """
    Read the mrc header and return as dictionary.

    Args
    ----------
    filename : str
        Name of the MRC file to parse

    Returns
    ----------
    headar : dict
        Dictionary with header values

    """
    header = {}
    with open(filename, 'r') as h:
        header['nx'], header['ny'], header['nz'] = np.fromfile(h, np.uint32, 3)
        header['mode'] = np.fromfile(h, np.uint32, 1)[0]
        header['nxstart'], header['nystart'], header['nzstart'] = np.fromfile(
            h, np.uint32, 3)
        header['mx'], header['my'], header['mz'] = np.fromfile(h, np.uint32, 3)
        header['xlen'], header['ylen'], header['zlen'] = np.fromfile(
            h, np.uint32, 3)
        _ = np.fromfile(h, np.uint32, 6)
        header['amin'], header['amax'], header['amean'] = np.fromfile(
            h, np.uint32, 3)
        _ = np.fromfile(h, np.uint32, 1)
        header['nextra'] = np.fromfile(h, np.uint32, 1)[0]
        _ = np.fromfile(h, np.uint16, 1)[0]
        _ = np.fromfile(h, np.uint8, 6)
        strbits = np.fromfile(h, np.int8, 4)
        header['ext_type'] = ''.join([chr(item) for item in strbits])
        header['nversion'] = np.fromfile(h, np.uint32, 1)[0]
        _ = np.fromfile(h, np.uint8, 16)
        header['nint'] = np.fromfile(h, np.uint16, 1)[0]
        header['nreal'] = np.fromfile(h, np.uint16, 1)[0]
        _ = np.fromfile(h, np.int8, 20)
        header['imodStamp'] = np.fromfile(h, np.uint32, 1)[0]
        header['imodFlags'] = np.fromfile(h, np.uint32, 1)[0]
        header['idtype'] = np.fromfile(h, np.uint16, 1)[0]
        header['lens'] = np.fromfile(h, np.uint16, 1)[0]
        header['nd1'], header['nd2'], header['vd1'], header['vd2'] = np.fromfile(
            h, np.uint16, 4)
        _ = np.fromfile(h, np.float32, 6)
        header['xorg'], header['yorg'], header['zorg'] = np.fromfile(
            h, np.float32, 3)
        strbits = np.fromfile(h, np.int8, 4)
        header['cmap'] = ''.join([chr(item) for item in strbits])
        header['stamp'] = np.fromfile(h, np.int8, 4)
        header['rms'] = np.fromfile(h, np.float32, 1)[0]
        header['nlabl'] = np.fromfile(h, np.uint32, 1)[0]
        strbits = np.fromfile(h, np.int8, 800)
        header['text'] = ''.join([chr(item) for item in strbits])
        header['ext_header'] = np.fromfile(
            h, np.int16, int(header['nextra'] / 2))
    return header


def load(filename, tilts=None):
    """
    Create a TomoStack object using data from a file.

    Parameters
    ----------
    filename : string
        Name of file that contains data to be read.  Accepted formats (.MRC,
        .RAW/.RPL pair, .DM3, .DM4)

    tilts : list or NumPy array
        List of floats indicating the specimen tilt at each projection

    Returns
    ----------
    stack : TomoStack object

    """
    known_file_types = ['.hdf5', '.mrc', '.ali', '.rec', '.hspy', '.dm3', '.dm4']
    hspy_file_types = ['.hdf5', '.h5', '.mrc', '.ali', '.rec', '.hspy']
    dm_file_types = ['.dm3', '.dm4']

    if type(filename) is str:
        ext = os.path.splitext(filename)[1]
        if ext.lower() in hspy_file_types:
            if ext.lower() in ['.ali', '.rec']:
                stack = load_hspy(filename, tilts, reader='mrc')
            else:
                stack = load_hspy(filename, tilts)
        elif ext.lower() in dm_file_types:
            stack = load_dm(filename)
        else:
            raise TypeError("Unknown file type %s. Must be %s one of " % (ext, [i for i in known_file_types]))

    elif type(filename) is list:
        ext = os.path.splitext(filename[0])[1]
        if ext.lower() in dm_file_types:
            stack = load_dm_series(filename)
        elif ext.lower() == '.mrc':
            logger.info('Data appears to be a SerialEM multiframe series.')
            mdocfiles = [i[:-3] + "mdoc" for i in filename]
            stack = load_serialem_series(filename, mdocfiles)
        else:
            raise TypeError("Unknown file type %s. Must be one of %s " % (ext, [i for i in known_file_types]))
    else:
        raise TypeError("Unknown filename type %s.  Must be either a string or list of strings." % type(filename))
    return stack
