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
    s = signal_to_tomo_stack(hspy.signals.Signal2D(data))

    s.axes_manager[0].name = 'Tilt'
    s.axes_manager[0].units = 'unknown'
    s.axes_manager[1].name = 'x'
    s.axes_manager[1].units = 'unknown'
    s.axes_manager[2].name = 'y'
    s.axes_manager[2].units = 'unknown'

    if tilts:
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
        if stack.original_metadata.fei_header.has_item('a_tilt'):
            tilts = stack.original_metadata.\
                      fei_header['a_tilt'][0:stack.data.shape[0]]
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
        else:
            logger.info('Unable to find tilt angles. Calibrate axis 0.')
            stack.axes_manager[0].name = 'Tilt'
            stack.axes_manager[0].units = 'degrees'

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
    s_new.metadata.Tomography.shifts = None
    s_new.metadata.Tomography.tiltaxis = 0.0
    s_new.metadata.Tomography.xshift = 0.0

    return s_new


def load_dm_series(dirname):
    """
    Load a series of individual DM3/DM4 files as a TomoStack object.

    Parameters
    ----------
    dirname : string
        Path to image series data.

    Returns
    ----------
    stack : TomoStack object

    """
    if dirname[-1] != "/":
        dirname = dirname + "/"
    files = glob.glob(dirname + "*.dm3")
    s = hspy.load(files)
    tilts = [i.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha for i in s]
    sorted_order = np.argsort(tilts)
    tilts_sorted = np.sort(tilts)
    files_sorted = list(np.array(files)[sorted_order])
    del s
    stack = hspy.load(files_sorted, stack=True, new_axis_name='tilt')
    stack.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha\
        = tilts_sorted
    stack.axes_manager[0].scale = tilts_sorted[1]-tilts_sorted[0]
    stack.axes_manager[0].units = 'degrees'
    stack.axes_manager[0].offset = tilts_sorted[0]

    stack = signal_to_tomo_stack(stack)
    return stack, tilts_sorted


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
