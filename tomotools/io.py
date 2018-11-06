# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:39:29 2015

@author: aherzing
"""
import numpy as np
import os

from PyQt5 import QtWidgets, QtGui
import hyperspy.api as hspy
import sys
from tomotools.base import TomoStack


def numpy_to_tomo_stack(data, manual_tilts=False):
    """Make a TomoStack object from a NumPy array.

    This will retain both the axes information and the metadata.
    If the signal is lazy, the function will return LazyPixelatedSTEM.

    Parameters
    ----------
    data : Numpy array
        Array containing tilt series data.  First dimension must represent
        the tilt axis. The second and third dimensions are the X and Y
        image dimentsions, respectively

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
    Tilts not found.  Calibrate axis 0
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

    if manual_tilts:
        negtilt = eval(input('Enter maximum negative tilt: '))
        postilt = eval(input('Enter maximum positive tilt: '))
        tiltstep = eval(input('Enter tilt step: '))
        tilts = np.arange(negtilt, postilt + tiltstep, tiltstep)
        print('User provided tilts stored')
        s.axes_manager[0].scale = tilts[1] - tilts[0]
        s.axes_manager[0].offset = tilts[0]
        s.axes_manager[0].units = 'degrees'

    return s


def signal_to_tomo_stack(s, manual_tilts=None):
    """Make a TomoStack object from a HyperSpy signal.

    This will retain both the axes information and the metadata.
    If the signal is lazy, the function will return LazyPixelatedSTEM.

    Parameters
    ----------
    s : HyperSpy Signal2D

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
    Tilts not found.  Calibrate axis 0
    >>> s_new
    <TomoStack, title: test dataset, dimensions: (50|500, 500)>

    """

    axes_list = [x for _, x in sorted(s.axes_manager.as_dictionary().items())]

    metadata = s.metadata.as_dictionary()
    original_metadata = s.original_metadata.as_dictionary()

    s_new = TomoStack(s.data, axes=axes_list, metadata=metadata, original_metadata=original_metadata)

    if s.axes_manager[0].name in ['Tilt', 'Tilts', 'Angle', 'Angles', 'Theta', 'tilt', 'tilts', 'angle', 'angles',
                                  'theta']:
        print('Tilts found in metadata')
        return s_new

    elif s.metadata.has_item('Acquisition_instrument.TEM.Stage.tilt_alpha'):
        tilts = s.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha[0:s.data.shape[0]]
        print('Tilts found in metadata')
        s_new.axes_manager[0].name = 'Tilt'
        s_new.axes_manager[0].units = 'degrees'
        s_new.axes_manager[0].scale = tilts[1] - tilts[0]
        s_new.axes_manager[0].offset = tilts[0]

    elif manual_tilts:
        negtilt = eval(input('Enter maximum negative tilt: '))
        postilt = eval(input('Enter maximum positive tilt: '))
        tiltstep = eval(input('Enter tilt step: '))
        tilts = np.arange(negtilt, postilt + tiltstep, tiltstep)
        print('User provided tilts stored')
        s_new.axes_manager[0].name = 'Tilt'
        s_new.axes_manager[0].units = 'degrees'
        s_new.axes_manager[0].scale = tilts[1] - tilts[0]
        s_new.axes_manager[0].offset = tilts[0]

    elif s.metadata.General.has_item('original_filename'):
        tiltfile = ('%s.rawtlt' % (os.path.split(os.path.splitext(s.metadata.General.original_filename)[0])[1]))
        if os.path.isfile(tiltfile):
            tilts = np.loadtxt(tiltfile)
            print('Tilts loaded from .RAWTLT File')
            s_new.axes_manager[0].name = 'Tilt'
            s_new.axes_manager[0].units = 'degrees'
            s_new.axes_manager[0].scale = tilts[1] - tilts[0]
            s_new.axes_manager[0].offset = tilts[0]

    else:
        s_new.axes_manager[0].name = 'Tilt'
        s_new.axes_manager[0].units = 'unknown'
        if s_new.axes_manager[1].name != 'x':
            s_new.axes_manager[1].name = 'x'
            s_new.axes_manager[1].units = 'unknown'
        if s_new.axes_manager[2].name != 'y':
            s_new.axes_manager[2].name = 'y'
            s_new.axes_manager[2].units = 'unknown'
        print('Tilts not found.  Calibrate axis 0')

    s_new.original_metadata.shifts = None
    s_new.original_metadata.tiltaxis = 0.0
    s_new.original_metadata.xshift = 0.0

    return s_new


def getfile(message='Choose files', filetypes='Tilt Series Type (*.mrc *.ali *.rec *.dm3 *.dm4)'):
    if 'PyQt5.QtWidgets' in sys.modules:
        app = QtWidgets.QApplication([])
        filename = QtWidgets.QFileDialog.getOpenFileName(None, message, os.getcwd(), filetypes)[0]
    elif 'PyQt4.QtGui' in sys.modules:
        app = QtGui.QApplication([])
        filename = QtGui.QFileDialog.getOpenFileName(None, message, os.getcwd(), filetypes)
    else:
        raise NameError('GUI applications require either PyQt4 or PyQt5')
    return filename


def loadhspy(filename, tilts=None):
    """
    Function to read an MRC file to a TomoStack object using the Hyperspy reader

    Parameters
    ----------
    filename : string
        Name of file that contains data to be read.  Accepted formats (.MRC, .RAW/.RPL pair, .DM3, .DM4)

    tilts : list or NumPy array
        List of floats indicating the specimen tilt at each projection

    Returns
    ----------
    stack : TomoStack object
    """

    if filename:
        file = filename
    else:
        file = getfile()

    stack = hspy.load(file)
    if stack.data.min() < 0:
        stack.data = np.float32(stack.data)
        stack.data += np.abs(stack.data.min())
    return signal_to_tomo_stack(stack, tilts)


def loaddm(filename):
    if filename:
        file = filename
    else:
        file = getfile()

    s = hspy.load(file)
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

    if s.data.min() < 0:
        s.data = np.float32(s.data)
        s.data += np.abs(s.data.min())

    axes_list = [x for _, x in sorted(s.axes_manager.as_dictionary().items())]

    metadata = s.metadata.as_dictionary()
    original_metadata = s.original_metadata.as_dictionary()

    s_new = TomoStack(s.data, axes=axes_list, metadata=metadata, original_metadata=original_metadata)
    s_new.axes_manager[0].axis = tilts
    print('Tilts found in metadata')

    s_new.axes_manager[0].units = 'degrees'
    s_new.original_metadata.shifts = None
    s_new.original_metadata.tiltaxis = 0.0
    s_new.original_metadata.xshift = 0.0

    return s_new


def load(filename=None, tilts=None):
    """
    Function to create a TomoStack object using data from a file

    Parameters
    ----------
    filename : string
        Name of file that contains data to be read.  Accepted formats (.MRC, .RAW/.RPL pair, .DM3, .DM4)

    tilts : list or NumPy array
        List of floats indicating the specimen tilt at each projection

    Returns
    ----------
    stack : TomoStack object

    """
    if filename is None:
        filename = getfile()

    ext = os.path.splitext(filename)[1]
    if ext in ['.HDF5', '.hdf5', '.hd5', '.HD5', '.MRC', '.mrc', '.ALI', '.ali', '.REC', '.rec']:
        stack = loadhspy(filename, tilts)
    elif ext in ['.dm3', '.DM3', '.dm4', '.DM4']:
        stack = loaddm(filename)
    else:
        raise ValueError("Unknown file type")
    return stack
