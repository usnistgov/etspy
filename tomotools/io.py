# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:39:29 2015

@author: aherzing
"""
import tomotools
import numpy as np
import os
try:
    from PyQt5 import QtWidgets
except:
    from PyQt4 import QtGui
from collections import OrderedDict
import hyperspy.api as hspy
import sys
from tomotools.base import TomoStack

def signal_to_tomo_stack(s,tilts,manual_tilts=None):
    """Make a TomoStack object from a HyperSpy signal.

    This will retain both the axes information and the metadata.
    If the signal is lazy, the function will return LazyPixelatedSTEM.

    Parameters
    ----------
    s : HyperSpy signal
        Should work for any HyperSpy signal.

    Returns
    -------
    tomo_stack_signal : TomoStack object

    """
    tiltfile = ('%s.rawtlt' % (os.path.split(os.path.splitext(s.metadata.General.original_filename)[0])[1]))    

    axes_list = [x for _, x in sorted(s.axes_manager.as_dictionary().items())]

    metadata = s.metadata.as_dictionary()
    original_metadata = s.original_metadata.as_dictionary()

    s_new = TomoStack(s.data,axes=axes_list,metadata=metadata,original_metadata=original_metadata)
    tiltfile = ('%s.rawtlt' % (os.path.split(os.path.splitext(s.metadata.General.original_filename)[0])[1]))
    
    
    if s.metadata.has_item('Acquisition_instrument.TEM.Stage.tilt_alpha'):
        tilts = s.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha[0:s.data.shape[0]]
        s_new.axes_manager[0].axis = tilts
        print('Tilts found in metadata') 
    
    elif os.path.isfile(tiltfile)==True:
        tilts = np.loadtxt(tiltfile)
        s_new.axes_manager[0].axis = tilts
        print('Tilts loaded from .RAWTLT File')
    
    elif manual_tilts:
        negtilt = eval(input('Enter maximum negative tilt: '))
        postilt = eval(input('Enter maximum positive tilt: '))    
        tiltstep = eval(input('Enter tilt step: '))
        tilts = np.arange(negtilt,postilt+tiltstep,tiltstep)
        s_new.axes_manager[0].axis = tilts
        print('User provided tilts stored')
    
    else:
        print('Tilts not found.  Calibrate axis 0')
    
    s_new.axes_manager[0].units = 'degrees'

    return s_new

def getFile(message='Choose files',filetypes='Tilt Series Type (*.mrc *.ali *.rec *.dm3 *.dm4)'):
    if 'PyQt5.QtWidgets' in sys.modules:
        app = QtWidgets.QApplication([])
        filename = QtWidgets.QFileDialog.getOpenFileName(None, message,os.getcwd(),filetypes)[0]
    elif 'PyQt4.QtGui' in sys.modules:
        app = QtGui.QApplication([])
        filename = QtGui.QFileDialog.getOpenFileName(None, message,os.getcwd(),filetypes)
    else:
        raise NameError('GUI applications require either PyQt4 or PyQt5')
    return(filename)

def LoadHspy(filename,tilts=None):
    """
    Function to read an MRC file to a TomoStack object using the Hyperspy reader

    Args
    ----------
    filename : string
        Name of file that contains data to be read.  Accepted formats (.MRC, .RAW/.RPL pair, .DM3, .DM4)
        
    Returns
    ----------
    stack : TomoStack object
    """  
    if filename:
        file = filename
    else:
        file = getFile()
        
    stack = hspy.load(file)
    if stack.data.min() < 0:
        stack.data = np.float32(stack.data)
        stack.data += np.abs(stack.data.min())
    return(signal_to_tomo_stack(stack,tilts))

def load(filename=None,reader=None,tilts=None):
    """
    Function to create a TomoStack object using data from a file

    Args
    ----------
    filename : string
        Name of file that contains data to be read.  Accepted formats (.MRC, .RAW/.RPL pair, .DM3, .DM4)
    reader : string
        File reader to employ for loading the data. If None, Hyperspy's load function is used.

    Returns
    ----------
    stack : TomoStack object
    """
    if filename is None:
        filename = getFile()
        
    #if reader is None:
    ext = os.path.splitext(filename)[1]
    if ext in ['.HDF5','.hdf5','.hd5','.HD5','.MRC','.mrc','.ALI','.ali','.REC','.rec']:
        stack = LoadHspy(filename,tilts)
#    if (ext in ['.HDF5','.hdf5','.hd5','.HD5']) or (reader in ['HSPY','hspy']):
#        stack = LoadHspy(filename)
#    elif (ext in ['.MRC','.mrc','.ALI','.ali','.REC','.rec']) and (reader in ['IMOD','imod']):
#        stack = LoadIMOD(filename)
#    elif (ext in ['.MRC','.mrc','.ALI','.ali','.REC','.rec']) and (reader in ['FEI','fei']):
#        stack = LoadFEI(filename)
    else:
        raise ValueError("Unknown file type")
    return(stack)
    
