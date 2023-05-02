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
import glob

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def numpy_to_tomo_stack(data, tilts=None, manual_tilts=False):
    """
    Create a TomoStack object from a NumPy array.

    This will retain both the axes information and the metadata.

    Parameters
    ----------
    data : Numpy array
        Array containing tilt series data.  First dimension must represent
        the tilt axis. The second and third dimensions are the X and Y
        image dimentsions, respectively

    tilts : Numpy array
        Array containing the tilt values in degrees for each projection.
        If provided, these values will be stored in
        stack.metadata.Tomography.tilts

    manual_tilts : bool
        If True, prompt for input of maximum positive tilt, maximum negative
        tilt, and tilt increment


    Returns
    -------
    tomo_stack_signal : TomoStack object


    Examples
    --------
    >>> import numpy as np
    >>> s = np.random.random((50, 500,500))
    >>> from tomotools.io import numpy_to_tomo_stack
    >>> s_new = numpy_to_tomo_stack(s)
    >>> s_new
    <TomoStack, title: , dimensions: (50|500, 500)>

    """
    s = signal_to_tomo_stack(hspy.signals.Signal2D(data), tilts)

    s.axes_manager[0].name = 'Tilt'
    s.axes_manager[0].units = 'unknown'
    s.axes_manager[1].name = 'x'
    s.axes_manager[1].units = 'unknown'
    s.axes_manager[2].name = 'y'
    s.axes_manager[2].units = 'unknown'

    if type(tilts) is np.ndarray:
        s.metadata.Tomography.tilts = tilts
        s.axes_manager[0].units = 'degrees'
        s.axes_manager[0].offset = tilts[0]
        s.axes_manager[0].scale = tilts[1] - tilts[0]
    elif manual_tilts:
        negtilt = eval(input('Enter maximum negative tilt: '))
        postilt = eval(input('Enter maximum positive tilt: '))
        tiltstep = eval(input('Enter tilt step: '))
        tilts = np.arange(negtilt, postilt + tiltstep, tiltstep)
        s.metadata.Tomography.tilts = tilts
        s.axes_manager[0].scale = tilts[1] - tilts[0]
        s.axes_manager[0].offset = tilts[0]
        s.axes_manager[0].units = 'degrees'

    axes_list = [x for _,
                 x in sorted(s.axes_manager.as_dictionary().items())]
    metadata_dict = s.metadata.as_dictionary()
    original_metadata_dict = s.original_metadata.as_dictionary()
    s = TomoStack(s, axes=axes_list, metadata=metadata_dict,
                  original_metadata=original_metadata_dict)
    return s


def signal_to_tomo_stack(s, tilt_signal=None, manual_tilts=False):
    """
    Create a TomoStack object from a HyperSpy signal.

    This will retain both the axes information and the metadata.

    Parameters
    ----------
    s : HyperSpy Signal2D or BaseSignal
        HyperSpy signal to be converted to TomoStack object

    tilt_signal : HyperSpy Signal1D or BaseSignal or NumPy array
        Signal or array that defines the tilt axis for the signal in degrees.

    manual_tilts : bool
        If True, prompt for input of maximum positive tilt, maximum negative
        tilt, and tilt increment


    Returns
    -------
    tomo_stack_signal : TomoStack object


    Examples
    --------
    >>> import numpy as np
    >>> import hyperspy.api as hs
    >>> s = hs.signals.Signal2D(np.random.random((50, 500,500)))
    >>> s.metadata.General.title = "test dataset"
    >>> s
    <Signal2D, title: test dataset, dimensions: (50|500, 500)>
    >>> from tomotools.io import signal_to_tomo_stack
    >>> s_new = signal_to_tomo_stack(s)
    >>> s_new
    <TomoStack, title: test dataset, dimensions: (50|500, 500)>
    """
    if isinstance(type(s), hspy.signals.BaseSignal):
        s = s.as_signal2D((0, 1))

    s_new = s.deepcopy()

    if tilt_signal is not None:
        if type(tilt_signal) in [np.ndarray, list]:
            if not s_new.metadata.has_item("Tomography"):
                s_new.metadata.add_node("Tomography")
            s_new.metadata.Tomography.tilts = tilt_signal
            s_new.axes_manager[0].name = 'Tilt'
            s_new.axes_manager[0].units = 'degrees'
            s_new.axes_manager[0].offset = tilt_signal[0]
            s_new.axes_manager[0].scale = tilt_signal[1] - tilt_signal[0]
        else:
            s_new.axes_manager[0].name = tilt_signal.axes_manager[0].name
            s_new.axes_manager[0].units = tilt_signal.axes_manager[0].units
            s_new.axes_manager[0].scale = tilt_signal.axes_manager[0].scale
            s_new.axes_manager[0].offset = tilt_signal.axes_manager[0].offset

    elif manual_tilts:
        negtilt = eval(input('Enter maximum negative tilt: '))
        postilt = eval(input('Enter maximum positive tilt: '))
        tiltstep = eval(input('Enter tilt step: '))
        tilts = np.arange(negtilt, postilt + tiltstep, tiltstep)
        logger.info('User provided tilts stored')
        s_new.axes_manager[0].name = 'Tilt'
        s_new.axes_manager[0].units = 'degrees'
        s_new.axes_manager[0].scale = tilts[1] - tilts[0]
        s_new.axes_manager[0].offset = tilts[0]

    elif s.metadata.has_item("Tomography"):
        if s.metadata.Tomography.tilts:
            tilts = s_new.metadata.Tomography.tilts
            logger.info("Tilts found in TomoStack metadata")

    elif s.metadata.has_item('Acquisition_instrument.TEM.Stage.tilt_alpha'):
        tilt_alpha = s.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha
        if type(tilt_alpha) is np.ndarray:
            n = s.data.shape[0]
            tilts = s.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha[0:n]
            logger.info('Tilts found in metadata')
            s_new.axes_manager[0].name = 'Tilt'
            s_new.axes_manager[0].units = 'degrees'
            s_new.axes_manager[0].scale = tilts[1] - tilts[0]
            s_new.axes_manager[0].offset = tilts[0]

    elif s.metadata.General.has_item('original_filename'):
        tiltfile = ('%s.rawtlt' % (os.path.split(os.path.splitext(
            s.metadata.General.original_filename)[0])[1]))
        if os.path.isfile(tiltfile):
            tilts = np.loadtxt(tiltfile)
            logger.info('.rawtlt file detected.')
            s_new.axes_manager[0].name = 'Tilt'
            s_new.axes_manager[0].units = 'degrees'
            s_new.axes_manager[0].scale = tilts[1] - tilts[0]
            s_new.axes_manager[0].offset = tilts[0]
            if len(tilts) == s.data.shape[0]:
                logger.info('Tilts loaded from .rawtlt file')
            else:
                logger.info('Number of tilts in .rawtlt file inconsistent'
                            ' with data shape')

    elif s_new.axes_manager[0].name in ['Tilt', 'Tilts', 'Angle', 'Angles',
                                        'Theta', 'tilt', 'tilts', 'angle',
                                        'angles', 'theta']:
        logger.info("Tilts found in HyperSpy signal axis 0")

    else:
        s_new.axes_manager[0].name = 'Tilt'
        s_new.axes_manager[0].units = 'unknown'
        if s_new.axes_manager[1].name not in ['x', 'X']:
            s_new.axes_manager[1].name = 'x'
            s_new.axes_manager[1].units = 'unknown'
        if s_new.axes_manager[2].name not in ['y', 'Y']:
            s_new.axes_manager[2].name = 'y'
            s_new.axes_manager[2].units = 'unknown'
        logger.info('Tilts not found.  Calibrate axis 0')
        tilts = None

    axes_list =\
        [x for _, x in sorted(s_new.axes_manager.as_dictionary().items())]

    metadata = s.metadata.as_dictionary()
    original_metadata = s.original_metadata.as_dictionary()

    s_new = TomoStack(s.data, axes=axes_list, metadata=metadata,
                      original_metadata=original_metadata)
    return s_new


def loadhspy(filename, tilts=None):
    """
    Read an MRC file to a TomoStack object using the Hyperspy reader.

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
    stack = hspy.load(filename)
    if not stack.metadata.has_item("Tomography"):
        stack.metadata.add_node("Tomography")
    ext = os.path.splitext(filename)[1]
    if ext.lower() in ['.mrc', '.ali', '.rec']:
        tiltfile = os.path.splitext(filename)[0] + '.rawtlt'
        txtfile = os.path.splitext(filename)[0] + '.txt'
        if stack.original_metadata.has_item('fei_header'):
            if stack.original_metadata.fei_header.has_item('a_tilt'):
                tilts = stack.original_metadata.fei_header['a_tilt'][0:stack.data.shape[0]]
                stack.axes_manager[0].name = 'Tilt'
                stack.axes_manager[0].units = 'degrees'
                stack.axes_manager[0].scale = tilts[1] - tilts[0]
                stack.axes_manager[0].offset = tilts[0]
                stack.metadata.Tomography.tilts = tilts
                logger.info('Tilts found in MRC file header')
            elif os.path.isfile(tiltfile):
                tilts = np.loadtxt(tiltfile)
                logger.info('.rawtlt file detected.')
                stack.axes_manager[0].name = 'Tilt'
                stack.axes_manager[0].units = 'degrees'
                stack.axes_manager[0].scale = tilts[1] - tilts[0]
                stack.axes_manager[0].offset = tilts[0]
                stack.metadata.Tomography.tilts = tilts
                if len(tilts) == stack.data.shape[0]:
                    logger.info('Tilts loaded from .rawtlt file')
                else:
                    logger.info('Number of tilts in .rawtlt file inconsistent'
                                ' with data shape')
            if stack.original_metadata.fei_header.has_item('pixel_size'):
                pixel_size = stack.original_metadata.fei_header.pixel_size[0]
                logger.info('Pixel size found in MRC file header')
            elif os.path.isfile(txtfile):
                pixel_line = None
                with open(txtfile, 'r') as h:
                    text = h.readlines()
                for i in text:
                    if 'Image pixel size' in i:
                        pixel_line = i
                if pixel_line:
                    pixel_size = np.float32(pixel_line.split()[-1:])[0]
                    pixel_units = pixel_line.split()[-2:-1][0][1:-2]
                    stack.axes_manager[1].name = 'x'
                    stack.axes_manager[1].units = pixel_units
                    stack.axes_manager[1].scale = pixel_size
                    stack.axes_manager[1].offset = 0

                    stack.axes_manager[2].name = 'y'
                    stack.axes_manager[2].units = pixel_units
                    stack.axes_manager[2].scale = pixel_size
                    stack.axes_manager[2].offset = 0
                    logger.info('Pixel size loaded from text file')
                else:
                    logger.info('Unable to find pixel size in text file')
            else:
                logger.info('Unable to find pixel size')
                stack.axes_manager[1].name = 'x'
                stack.axes_manager[1].units = 'unknown'

                stack.axes_manager[2].name = 'y'
                stack.axes_manager[2].units = 'unknown'
        elif stack.original_metadata.has_item('std_header'):
            logger.info('SerialEM generated MRC file detected')
        else:
            logger.info('Unable to find tilt angles. Calibrate axis 0.')
            stack.axes_manager[0].name = 'Tilt'
            stack.axes_manager[0].units = 'degrees'

    elif ext.lower() in ['.hdf5', '.hd5', '.hspy']:
        pass
    else:
        raise ValueError('Cannot read file type: %s' % ext)
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


def loaddm(filename):
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

    s_new = TomoStack(s)
    s_new.axes_manager[0].name = 'Tilt'
    s_new.axes_manager[0].units = 'degrees'
    s_new.axes_manager[0].offset = tilts[0]
    s_new.axes_manager[0].scale = tilts[1] - tilts[0]
    logger.info('Tilts found in metadata')

    s_new.metadata.Tomography.tilts = tilts
    s_new.metadata.Tomography.shifts = np.zeros([s_new.data.shape[0], 2])
    s_new.metadata.Tomography.tiltaxis = 0.0
    s_new.metadata.Tomography.xshift = 0.0

    return s_new


def load_dm_series(input_data):
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
    if type(input_data) is str:
        dirname = input_data
        if dirname[-1] != "/":
            dirname = dirname + "/"
        dm3files, dm4files = [
            glob.glob(i) for i in [dirname + "*.dm3", dirname + "*.dm4"]]
        if len(dm3files) == 0 and len(dm4files) == 0:
            raise ValueError("No DM files found in path")
        elif len(dm3files) > 0 and len(dm4files) > 0:
            raise ValueError("Multipe DM formats found in path")
        elif len(dm3files) > 0:
            files = dm3files
        else:
            files = dm4files
    elif type(input_data) is list:
        files = input_data
    else:
        raise ValueError("Unknown input data type.")
    s = hspy.load(files)
    tilts = [i.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha for i in s]
    sorted_order = np.argsort(tilts)
    tilts_sorted = np.sort(tilts)
    files_sorted = list(np.array(files)[sorted_order])
    del s
    stack = hspy.load(files_sorted, stack=True, new_axis_name='tilt')
    stack.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha\
        = tilts_sorted
    stack.axes_manager[0].scale = tilts_sorted[1] - tilts_sorted[0]
    stack.axes_manager[0].units = 'degrees'
    stack.axes_manager[0].offset = tilts_sorted[0]

    stack = signal_to_tomo_stack(stack)
    stack.metadata.Tomography.tilts = tilts_sorted
    return stack


def parse_mdoc(mdoc_file):
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


# def load_serialem_series(mrcfiles, mdocfiles, align=True):
#     """
#     Load a multi-frame series collected by SerialEM.

#     Parameters
#     ----------
#     mrc_files : list of files
#         MRC files containing multi-frame tilt series data.

#     mdoc_files : list of files
#         SerialEM metadata files for multi-frame tilt series data.

#     align : bool
#         If True, align the frames using PyStackReg at each tilt prior to averaging.

#     Returns
#     ----------
#     stack : TomoStack object
#         Tilt series resulting by averaging frames at each tilt

#     """
#     stack = np.zeros([len(mrcfiles), 1024, 1024], np.float32)

#     sr = StackReg(StackReg.TRANSLATION)

#     meta = [None] * len(mdocfiles)
#     for i in range(0, len(mdocfiles)):
#         meta[i] = parse_mdoc(mdocfiles[i])

#     tilts = np.array([meta[i]['TiltAngle'] for i in range(0, len(meta))])
#     tilts_sort = np.argsort(tilts)

#     for i in range(0, len(mrcfiles)):
#         fn = mdocfiles[tilts_sort[i]][:-5]
#         s = hspy.load(fn)
#         if align:
#             ali = sr.register_transform_stack(s.data, reference='previous')
#             stack[i] = ali.mean(0)
#         else:
#             stack[i] = s.data.mean(0)
#     stack = numpy_to_tomo_stack(stack, tilts[tilts_sort])
#     stack.axes_manager[1].scale = meta[0]['PixelSpacing'] / 10
#     stack.axes_manager[2].scale = meta[0]['PixelSpacing'] / 10
#     stack.axes_manager[1].name = 'nm'
#     stack.axes_manager[2].name = 'nm'

#     if not stack.metadata.has_item('Acquisition_instrument.TEM'):
#         stack.metadata.add_node('Acquisition_instrument.TEM')
#     stack.metadata.Acquisition_instrument.TEM.magnification = meta[0]['Magnification']
#     stack.metadata.Acquisition_instrument.TEM.beam_energy = meta[0]['Voltage']
#     stack.metadata.Acquisition_instrument.TEM.dwell_time = meta[
#         0]['ExposureTime'] * s.data.shape[0] * 1e-6
#     stack.metadata.Acquisition_instrument.TEM.spot_size = meta[0]['SpotSize']
#     stack.metadata.Acquisition_instrument.TEM.defocus = meta[0]['Defocus']
#     stack.metadata.General.original_filename = meta[0]['ImageFile']
#     return stack


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
    ext = os.path.splitext(filename)[1]
    if ext in ['.HDF5', '.hdf5', '.hd5', '.HD5', '.MRC', '.mrc', '.ALI',
               '.ali', '.REC', '.rec', '.hspy', '.HSPY']:
        stack = loadhspy(filename, tilts)
    elif ext in ['.dm3', '.DM3', '.dm4', '.DM4']:
        stack = loaddm(filename)
    else:
        raise ValueError("Unknown file type")
    return stack


def read_serialem_series(mrcfiles, mdocfiles):
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

    stack = [None] * len(mrcfiles)
    meta = [None] * len(mdocfiles)
    for i in range(0, len(mdocfiles)):
        meta[i] = parse_mdoc(mdocfiles[i])

    tilts = np.array([meta[i]['TiltAngle'] for i in range(0, len(meta))])
    tilts_sort = np.argsort(tilts)
    tilts.sort()

    for i in range(0, len(mrcfiles)):
        fn = mdocfiles[tilts_sort[i]][:-5]
        stack[i] = hspy.load(fn)

    images_per_tilt = stack[0].data.shape[0]
    stack = hspy.stack(stack)
    stack.axes_manager[1].scale = tilts[1] - tilts[0]
    stack.axes_manager[1].offset = tilts[0]
    stack.axes_manager[1].units = 'degrees'
    stack.axes_manager[1].units = 'Tilt'
    stack.axes_manager[2].scale = meta[0]['PixelSpacing'] / 10
    stack.axes_manager[3].scale = meta[0]['PixelSpacing'] / 10
    stack.axes_manager[2].units = 'nm'
    stack.axes_manager[3].units = 'nm'
    stack.axes_manager[2].name = 'y'
    stack.axes_manager[3].name = 'x'

    if not stack.metadata.has_item('Acquisition_instrument.TEM'):
        stack.metadata.add_node('Acquisition_instrument.TEM')
    stack.metadata.Acquisition_instrument.TEM.magnification = meta[0]['Magnification']
    stack.metadata.Acquisition_instrument.TEM.beam_energy = meta[0]['Voltage']
    stack.metadata.Acquisition_instrument.TEM.dwell_time = meta[
        0]['ExposureTime'] * images_per_tilt * 1e-6
    stack.metadata.Acquisition_instrument.TEM.spot_size = meta[0]['SpotSize']
    stack.metadata.Acquisition_instrument.TEM.defocus = meta[0]['Defocus']
    stack.metadata.General.original_filename = meta[0]['ImageFile']
    return stack, tilts


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
