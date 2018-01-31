# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:39:29 2015

@author: aherzing
"""
import tomotools
import numpy as np
import os
try:
    from PyQt5 import QtWidgets as QtGui
except:
    from PyQt4 import QtGui
from collections import OrderedDict
import hyperspy.api as hspy
    
def LoadHspy(filename):
    """
    Function to read an MRC file to a Stack object using the Hyperspy reader

    Args
    ----------
    filename : string
        Name of file that contains data to be read.  Accepted formats (.MRC, .RAW/.RPL pair, .DM3, .DM4)
        
    Returns
    ----------
    stack : Stack object
    """  
    if filename:
        file = filename
    else:
        app = QtGui.QApplication([])
        file = QtGui.QFileDialog.getOpenFileName(None, 'Choose files',os.getcwd(),'Tilt Series Type (*.mrc *.ali *.rec *.dm3 *.dm4)')[0]
        
    stack = tomotools.base.Stack()
    temp = hspy.load(file)
    stack.data = np.float32(temp)
    if stack.data.min() < 0:
        stack.data += np.abs(stack.data.min())
    if isinstance(temp.axes_manager[0].units,str):
        stack.tilts = temp.axes_manager[0].axis
        stack.tilts = np.around(stack.tilts,decimals=1)
        print('Tilts found in metadata!')
    elif temp.metadata.has_item('Acquisition_instrument.TEM.Stage.tilt_alpha'):
        if np.size(temp.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha) >= temp.data.shape[0]:
            stack.tilts = temp.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha[0:temp.data.shape[0]]
            stack.tilts = np.around(stack.tilts,decimals=1)
            print('Tilts found in metadata!')
    else:
        print('Tilts not found in metadata. Define Axis Zero')
    stack.pixelsize = temp.axes_manager[1].scale
    stack.pixelunits = temp.axes_manager[1].units
    stack.header,stack.extheader = MakeHeader(stack.data)
    return(stack)

def LoadIMOD(filename):
    """
    Function to read an MRC file to a Stack object using the Hyperspy reader

    Args
    ----------
    filename : string
        Name of file that contains data to be read.  Accepted formats (.MRC, .RAW/.RPL pair, .DM3, .DM4)
        
    Returns
    ----------
    stack : Stack object
    """
    
    if filename:
        file = filename
    else:
        app = QtGui.QApplication([])
        file = QtGui.QFileDialog.getOpenFileName(None, 'Choose files',os.getcwd(),'Tilt Series Type (*.mrc *.ali *.rec *.dm3 *.dm4)')[0]

    with open(file,'rb') as h:

        # Read header from file
        header = OrderedDict()
        header['nx'] = np.fromfile(h,'int32',1)
        header['ny'] = np.fromfile(h,'int32',1)
        header['nz'] = np.fromfile(h,'int32',1)
        header['mode']= np.fromfile(h,'int32',1)
        header['nxstart']= np.fromfile(h,'int32',1)
        header['nystart']= np.fromfile(h,'int32',1)
        header['nzstart']= np.fromfile(h,'int32',1)
        header['mx']= np.fromfile(h,'int32',1)
        header['my']= np.fromfile(h,'int32',1)
        header['mz']= np.fromfile(h,'int32',1)
        header['xlen']= np.fromfile(h,'float32',1)
        header['ylen']= np.fromfile(h,'float32',1)
        header['zlen']= np.fromfile(h,'float32',1)
        header['alpha']= np.fromfile(h,'float32',1)
        header['beta']= np.fromfile(h,'float32',1)
        header['gamma']= np.fromfile(h,'float32',1)
        header['mapc']= np.fromfile(h,'int32',1)
        header['mapr']= np.fromfile(h,'int32',1)
        header['maps']= np.fromfile(h,'int32',1)
        header['amin']= np.fromfile(h,'float32',1)
        header['amax']= np.fromfile(h,'float32',1)
        header['amean']= np.fromfile(h,'float32',1)
        header['ispg']= np.fromfile(h,'int32',1)
        header['next']= np.fromfile(h,'int32',1)
        header['createid']= np.fromfile(h,'int16',1)
        header['extra data']= np.fromfile(h,'int8',6)
        strbits = np.fromfile(h,'int8',4)
        header['extType']= ''.join([chr(item) for item in strbits])
        header['nversion']= np.fromfile(h,'int32',1)
        header['extra data']= np.fromfile(h,'int8',16)
        header['nint']= np.fromfile(h,'int16',1)
        header['nreal']= np.fromfile(h,'int16',1)
        header['extra data']= np.fromfile(h,'int8',20)
        header['imodStamp']= np.fromfile(h,'int32',1)
        header['imodFlags']= np.fromfile(h,'int32',1)
        header['idtype']= np.fromfile(h,'int16',1)
        header['lens']= np.fromfile(h,'int16',1)
        header['nd1']= np.fromfile(h,'int16',1)
        header['nd2']= np.fromfile(h,'int16',1)
        header['vd1']= np.fromfile(h,'int16',1)
        header['vd2']= np.fromfile(h,'int16',1)
        header['tiltangles']= np.fromfile(h,'float32',6)
        header['xorg']= np.fromfile(h,'float32',1)
        header['yorg']= np.fromfile(h,'float32',1)
        header['zorg']= np.fromfile(h,'float32',1)
        strbits = np.fromfile(h,'int8',4)
        header['cmap']= ''.join([chr(item) for item in strbits])
        header['stamp'] = np.fromfile(h,'int8',4)
        header['rms']= np.fromfile(h,'float32',1)
        header['nlabl']= np.fromfile(h,'int32',1)
        header['labels'] = np.fromfile(h,'int8',800)
        labels = ''.join([chr(item) for item in header['labels'][0:header['nlabl'][0]*80]])
        header['labels'] = labels
    
        extralength = (header['nz'][0]*(header['nint'][0]*2 + header['nreal'][0]*4))
        extra = np.fromfile(h,'int8',extralength)
    
        #Determine byte length of input data from Mode key in header
        if header['mode'] == 0:
            fmt = 'int8'
        elif header['mode'] ==1:
            fmt = 'int16'
        elif header['mode'] == 2:
            fmt = 'float32'
        elif header['mode'] == 6:
            fmt = 'uint16'
        else:
            fmt = 'uint16'
         
        datasize = header['nx'][0]*header['ny'][0]*header['nz'][0]
        stack = TomoTools.base.Stack()
        stack.data = np.fromfile(h,fmt,datasize)
        
    stack.data = np.reshape(stack.data,(header['nz'][0],header['ny'][0],header['nx'][0]))
    stack.data = np.float32(stack.data)
    stack.data += np.abs(np.min(stack.data))
    stack.header = header
    stack.pixelsize = stack.header['xlen'][0]/stack.header['nx'][0]/10
    stack.pixelunits = 'nm'

    tiltfile = ('%s.rawtlt' % (os.path.split(os.path.splitext(file)[0])[1]))
    if os.path.isfile(tiltfile)==True:
        print('Tilts loaded from .RAWTLT File')
        tilts = np.loadtxt(tiltfile)
    else:
        negtilt = eval(input('Enter maximum negative tilt: '))
        postilt = eval(input('Enter maximum positive tilt: '))    
        tiltstep = eval(input('Enter tilt step: '))
        tilts = np.arange(negtilt,postilt+tiltstep,tiltstep)
        k = open(tiltfile,'w')              
        tilts.tofile(k,'\n')
        k.close()
    stack.tilts = tilts
    return(stack)

    
def WriteMRC(out,outfile=None):
    """
    Function to write data to FEI style .MRC file.  If the Stack has no header, one will
    be created using the MakeHeader() function

    Args
    ----------
    out : Stack object
        Stack object containing tilt series or reconstructed volume to be saved
    outfile : string (optional)
        Filename of the output file. If None, user will be prompted to provide a filename.
    """
    if outfile == None:
        app = QtGui.QApplication([])
        #file = QtGui.QFileDialog.getOpenFileName(None, 'Choose files',os.getcwd(),'Tilt Series Type (*.mrc *.ali *.rec *.dm3 *.dm4)')
        outfile = QtGui.QFileDialog.getSaveFileName(None,'Save tomography data','')   
    #if (out.header == None) and (out.extheader == None) or newheader:
    out.header,out.extheader = MakeHeader(out.data)     
    if out.data.dtype == 'uint16':
        out.data = np.int16(np.single(out.data) + 32768) 
    with open(outfile,'wb') as h:
        for key in out.header:
            out.header[key].tofile(h)       
        for key in out.extheader:
                out.extheader[key].tofile(h)
        out.data.tofile(h)
    print(('\nData written to file: %s' % outfile))
    return

def WriteHDF5(data,outfile=None,compression=None):
    """
    Function to write data to a Hyperspy HDF5 file

    Args
    ----------
    out : Stack object
        Stack object containing tilt series or reconstructed volume to be saved
    outfile : string (optional)
        Filename of the output file. If None, user will be prompted to provide a filename.
    """
    if outfile == None:
        outfile = QtGui.QFileDialog.getSaveFileName(None,'Save tomography data','')
    out = hspy.signals.Signal2D(np.zeros([data.data.shape[0],data.data.shape[1],data.data.shape[2]],data.data.dtype))
    out.data = data.data

    out.axes_manager[0].name = 'Y'
    out.axes_manager[1].name = 'X'
    out.axes_manager[2].name = 'Z'

    out.axes_manager[1].scale = data.pixelsize
    out.axes_manager[2].scale = data.pixelsize
    out.axes_manager[0].scale = data.pixelsize

    out.axes_manager[0].units = data.pixelunits
    out.axes_manager[1].units = data.pixelunits
    out.axes_manager[2].units = data.pixelunits

    out.axes_manager[0].axis = np.arange(0,data.data.shape[0]*data.pixelsize,data.pixelsize)
    out.axes_manager[1].axis = np.arange(0,data.data.shape[1]*data.pixelsize,data.pixelsize)
    out.axes_manager[2].axis = np.arange(0,data.data.shape[2]*data.pixelsize,data.pixelsize)

    out.save(outfile,overwrite=True,compression=compression)
        
def MakeHeader(data):
    """
    Function to create a header for a stack that does not have one.  Relevant parameters
    will be determined from the data itself

    Args
    ----------
    data : numpy array
        Array containing tilt series or reconstructed volume to be saved

    Returns
    ----------
    header : dictionary
        Dictionary containing the standard header info
    extheader : dictionary
        Dictionary with all values set to zero to serve as placeholder in MRC output file
    """
    header = OrderedDict()
    header['nx'] = np.int32(np.size(data,2))
    header['ny'] = np.int32(np.size(data,1))
    header['nz'] = np.int32(np.size(data,0))
    if data.dtype == 'int8':
        header['mode'] = np.int32(0)
    elif data.dtype == 'int16':
        header['mode'] = np.int32(1)
    elif data.dtype == 'float32':
        header['mode'] = np.int32(2)
    elif data.dtype == 'uint16':
        header['mode'] = np.int32(6)
    else:
        raise AssertionError('Unrecognized data type %s' % data.dtype)   
    header['nxstart']= np.int32(0)
    header['nystart']= np.int32(0)
    header['nzstart']= np.int32(0)
    header['mx']= np.int32(np.size(data,1))
    header['my']= np.int32(np.size(data,2))
    header['mz']= np.int32(np.size(data,0))
    header['xlen']= np.float32(np.size(data,1))
    header['ylen']= np.float32(np.size(data,2))
    header['zlen']= np.float32(np.size(data,0))
    header['alpha']= np.float32(90)
    header['beta']= np.float32(90)
    header['gamma']= np.float32(90)
    header['mapc']= np.int32(1)
    header['mapr']= np.int32(1)
    header['maps']= np.int32(1)
    header['amin']= np.float32(np.min(np.min(np.min(data))))
    header['amax']= np.float32(np.max(np.max(np.max(data))))
    header['amean']= np.float32(np.mean(np.mean(np.mean(data))))
    header['ispg']= np.int16(0)
    header['nsymbt']= np.int16(0)
    header['next']= np.int32(131072)
    header['dvid']= np.int16(0)
    header['extra']= np.zeros(30,'int8')
    header['numintegers']=  np.int16(0)
    header['numfloats']= np.int16(32)
    header['sub']= np.int16(0)
    header['zfac']= np.int16(0)
    header['min2']= np.float32(0)
    header['max2']= np.float32(0)
    header['min3']= np.float32(0)
    header['max3']= np.float32(0)
    header['min4']= np.float32(0)
    header['max4']= np.float32(0)
    header['idtype']= np.int16(0)
    header['lens']=np.int16(0)
    header['nd1']=np.int16(0)
    header['nd2']=np.int16(0)
    header['vd1']=np.int16(0)
    header['vd2']=np.int16(0)
    header['tiltangles']=np.zeros(9,'float32')
    header['zorg']=np.float32(0)
    header['xorg']=np.float32(0)
    header['yorg']=np.float32(0)
    header['nlabl']=np.int32(1)
    header['labl']=np.zeros(34,'int8')
    header['lblextra']=np.zeros(766,'int8')
    
    extheader = OrderedDict()
    for i in range(0,1024):    
        extheader['a_tilt',i]=np.float32(0)
        extheader['b_tilt',i]=np.float32(0)
        extheader['x_stage',i]=np.float32(0)
        extheader['y_stage',i]=np.float32(0)
        extheader['z_stage',i]=np.float32(0)
        extheader['x_shift',i]=np.float32(0)
        extheader['y_shift',i]=np.float32(0)
        extheader['defocus',i]=np.float32(0)
        extheader['exp_time',i]=np.float32(0)
        extheader['mean_int',i]=np.float32(0)
        extheader['tilt_axis',i]=np.float32(0)
        extheader['pixel_size',i]=np.float32(0)
        extheader['magnification',i]=np.float32(0)
        extheader['ht',i]=np.float32(0)
        extheader['binning',i]=np.float32(0)
        extheader['appliedDefocus',i]=np.float32(0)
        extheader['remainder',i]= np.zeros(16,'float32')
    return[header,extheader]
