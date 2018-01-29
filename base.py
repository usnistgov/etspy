# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:38:16 2016

@author: aherzing
"""

import numpy as np
from TomoTools import io , recon , align
import copy
import os
import cv2
import tqdm
import pylab as plt
import matplotlib.animation as animation
try:
    from PyQt5 import QtGui
except:
    from PyQt4 import QtGui

class Stack:
    """
    Create a Stack object for tomography data.

    Note: All attributes are initialized with values of None or 0.0 in __init__.

    Attributes
    ----------
    data : numpy array
        The tilt series or reconstruction data array. It can be either 2 or 3 dimensional.
        For a tilt series, the first dimension must be the tilt increment axis (e.g. [theta,X] or [theta,X,Y]).
        Prior to reconstruction, the third dimension must describe the tilt axis orientation.
        For a reconstruction, the dimensions will be [Y,X,Z].
    datafile : string
        File from which data is read.
    datapath : string
        Location of the datafile to be read.
    extheader : dictionary (not currently used)
        A dictionary containing extended metadata read from an extended MRC format.
    header: dictionary (not currently used)
        A dictionary containing all relevant metadata, typically read from an MRC file.
    offset : integer
        The value by which 'int16' data is offset in order to make all values positive.
    pixelsize : float
        Spatial dimension of each image pixel.
    pixelunits : string
        Units for the pixel size, typically 'nm'
    shifts : numpy array
        X,Y shifts calculated for each image for stack registration
    tiltaxis : float
        Angular orientation (in degrees) by which data is rotated to orient the stack so that
        the tilt axis is vertical
    tiltfile : string
        ASCII file containing a list of sample orientations for each image in the stack.
        Some tomorgaphy software outputs this as a .RAWTLT file.
    xshift : float
        Lateral shift of the tilt axis from the center of the stack.
    """
        
    def __init__(self):
        self.data = None
        self.datafile = None
        self.datapath = None
        self.extheader = None
        self.header = None
        self.offset = 0.0
        self.pixelsize = 0.0
        self.pixelunits = 'nm'
        self.shifts = None
        self.tiltaxis = 0.0
        self.tiltfile = None
        self.tilts = None
        self.xshift = 0.0

    def __repr__(self):
        string = '<TomoToolsStack'
        if self.data is None:
            string += ", Empty"
        elif len(self.data.shape)==2:
            string += ", Dimensions: %s x %s" % (str(self.data.shape[0]),
                                                  str(self.data.shape[1]))
        elif len(self.data.shape)==3:
            string += ", Dimensions: %s x %s x %s" % (str(self.data.shape[0]),
                                                  str(self.data.shape[1]),
                                                  str(self.data.shape[2]))
        string += '>'
        return string
    
    def alignOther(self,other):
        """
        Method to apply the alignment calculated for one dataset to another. This will include the spatial registration,
        tilt axis, and tilt axis shift if they have been previously calulated.

        Args
        ----------
        other : Stack object
            The tilt series which is to be aligned using the previously calculated parameters.
            The data array in the Stack must be of the same size as that in self.data
        
        Returns
        ----------
        out : Stack object
            The result of applying the alignment to other    
        """
        if self.shifts is None:
            raise ValueError('Spatial registration has not been calculated for this stack')
            
        out = align.alignToOther(self,other)
        return(out)

    def checkErrorCPU(self,thickness,tilts=None,N=None,minError=2.0,maxiters=100):
        """
        Method to calculate the number of SIRT iterations required to produce a minimum change
        between successive iterations.  This method uses the CPU-based SIRT reconstruction
        algorithm of the Astra toolbox.

        Args
        ----------
        thickness : integer
            Size in pixels of the Z-dimension of the output reconstruction.
        tilts : list
            List of floats indicating the specimen tilt for each image in the stack
        N : integer
            Location of the slice to use for error calculation
        minError : float
            Percentage change between successive iterations that terminates the algorithm
        maxiters : integer
            Maximum number of iterations to perform if minError is not met
        """
        
        if thickness == None:
            thickness = eval(input('Enter the thickness for the reconstruction in pixels:'))
        if N is None:
            N = np.int32(self.data.shape[1]/2)
        data = self.data[:,N,:]
        
        if tilts is None:
            tilts = self.tilts
        recon.errorSIRTCPU(data,thickness=thickness,tilts=tilts,minError=minError,maxiters=maxiters)
        return
    
    def checkErrorGPU(self,thickness=None,nIters=200,N=None,output=False):
        """
        Method to calculate the number of SIRT iterations required to produce a minimum change
        between successive iterations.  This method uses the GPU-based SIRT reconstruction
        algorithm of the Astra toolbox.

        Args
        ----------
        thickness : integer
            Size in pixels of the Z-dimension of the output reconstruction.
        nIters : integer
            Number of iterations to perform
        N : integer
            Location of the slice to use for error calculation. If None, the middle slice is chosen.
        output : boolean
            If True, the results of the calculation are displayed
        """
        
        if thickness == None:
            thickness = eval(input('Enter the thickness for the reconstruction in pixels:'))
        #if step == None:
           # step = eval(input('Enter the iteration step:'))
        if N == None:
            N = np.int16(self.data.shape[1]/2)
        error,diff,rec = recon.errorSIRTGPU(self,thickness,N,nIters)
#        if output:
#            fig,ax = plt.subplots(1)
#            ax.plot(np.arange(start+step,start+len(error)*step,step),diff)
#            ax.set_xlabel('Iterations')
#            ax.set_ylabel('% Error')
#            fig2,ax2 = plt.subplots(1)
#            plt.imshow(rec[0,:,:],clim=[0.0,0.5*rec.max()],cmap='inferno')
        self.error = error
        self.diff = diff
        self.errorRec = rec
        return

    def deepcopy(self):
        """
        Method to create a copy, including metadata of the stack.

        Returns
        ----------
        A copy of the input stack
        
        """
        return(copy.deepcopy(self))    

    def stackRegister(self,method='ECC',start=None):
        """
        Method which calls a function in the align module to spatially register a stack using one of two 
        OpenCV based algorithms: Phase Correlation (PC) or Enhanced Correlation Coefficient (ECC) maximization.

        Args
        ----------
        method : string
            Algorithm to use for registration calculation. Must be either 'PC' or 'ECC'
        start : integer
            Position in tilt series to use as starting point for the alignment. If None, the central projection is used.

        Returns
        ----------
        out : Stack object
            Spatially registered copy of the input stack
        """
        
        if method == 'ECC':
            out = align.rigidECC(self,start)
        elif method == 'PC':
            out = align.rigidPC(self,start)
        else:
            print("Unknown registration method.  Must use 'ECC' or 'PC'")
            return()
        return(out)

    def tiltAlign(self,method,limit=10,delta=0.3,offset=0.0):
        """
        Method to call one of two tilt axis calculation functions in the align module ('CoM' and 'MaxImage')
        and apply the calculated rotation.

        Available options are 'CoM' and 'Error'

        CoM: track the center of mass (CoM) of the projections at three locations.  Fit the
        motion of the CoM as a function of tilt to that expected for an ideal cylinder to calculate
        an X-shift at each location.  Perform a  linear fit of the three X-shifts to calculate
        an ideal rotation.

        MaxImage: Perform automated determination of the tilt axis of a Stack by measuring
        the rotation of the projected maximum image.  Maximum image is rotated postively
        and negatively, filterd using a Hamming window, and the rotation angle is
        determined by iterative histogram analysis

        Args
        ----------
        method : string
            Algorithm to use for registration alignment. Must be either 'CoM' or 'MaxImage'
        limit : integer
            Position in tilt series to use as starting point for the alignment. If None, the central projection is used.
        delta : integer
            Position i
        offset : integer
            Not currently used
        limit : integer or float
            Maximum rotation angle to use for MaxImage calculation
        delta : float
            Angular increment for MaxImage calculation
            
        Returns
        ----------
        out : Stack object
            Copy of the input stack rotated by calculated angle
        """
        
        if method == 'CoM':
            out = align.tiltCorrect(self,offset)
        elif method == 'MaxImage':
            angle = align.tiltAnalyze(self,limit,delta)
            if angle > 0.1:
                out = self.rotate(angle,True,False)
            else:
                out = self.deepcopy()
            out.tiltaxis = angle
        else:
            print('Invalid alignment method: Enter either "CoM" or "Error"')
            return
        return(out)

    def rebin(self, factor):
        """
        Method to spatiall bin the Stack by an integer factor in the X-Y dimension.  The function calls the
        OpenCV 'resize' function.

        Args
        ----------
        factor : integer
            Factor by which to downsize the input data stack
            
        Returns
        ----------
        out : Stack object
            Copy of the input stack downsampled by factor
        """
        out = copy.deepcopy(self)
        out.data = np.zeros([np.shape(self.data)[0],np.int(np.shape(self.data)[1]/factor),np.int(np.shape(self.data)[2]/factor)],dtype=self.data.dtype)
        for i in range(0,np.size(self.data,0)):
            out.data[i,:,:] = cv2.resize(self.data[i,:,:], tuple(np.shape(out.data)[1:3]))
        out.pixelsize = out.pixelsize*factor
        return(out)

    def reconstruct(self,method='astraWBP',thickness=None,iterations=None,constrain=False,thresh=0,CUDA=None):
        """
        Function to reconstruct a stack using one of the available methods:
        astraWBP, astraSIRT, astraSIRT_GPU

        Args
        ----------
        method : string
            Reconstruction algorithm to use.  Must be either 'astraWBP' (default), 'astraSIRT', or 'astraSIRT_GPU'
        thickness : integer
            Number of pixels in the Z dimension of the reconstructed data
        iterations : integer
            Number of iterations for the SIRT reconstruction (for astraSIRT and astraSIRT_GPU ,methods only)
        chunksize : integer
            Number of chunks by which to divide the data for multiprocessing
        constrain : boolean
            If True, output reconstruction is constrained above value given by 'thresh'
        thresh : integer or float
            Value above which to constrain the reconstructed data

        Returns
        ----------
        out : Stack object
            Stack containing the reconstructed volume
        """
        if CUDA is None:    
            if 'CUDA_Path' in os.environ.keys():
                CUDA = True
            else:
                CUDA = False
        if thickness == None:
            thickness = eval(input('Enter the thickness for the reconstruction in pixels:'))
        out = copy.deepcopy(self)
        out.data = recon.run(self,method,thickness,iterations,constrain,thresh,CUDA)
        if len(np.shape(out.data))==3:
            out.header['nx'] = np.int32(np.shape(out.data)[2])
        elif len(np.shape(out.data))==2:
            out.header['nx'] = np.int32(1)
        out.header['ny'] = np.int32(np.shape(out.data)[1])
        out.header['nz'] = np.int32(np.shape(out.data)[0])
        out.header['mode'] = np.int32(2)
        return(out)
    
    def rotate(self,angle,progressbar=True,resize=False):
        """
        Method to rotate the stack by a given angle using the OpenCV warpAffine function

        Args
        ----------
        angle : float
            Angle by which to rotate the data in the Stack about the XY plane
        progressbar : boolean
            If True, use the tqdm module to output a progressbar during rotation.
        resize : boolean
            If True, output stack size is increased relative to input so that no pixels are lost.
            If False, output stack is the same size as the input.
            
        Returns
        ----------
        rot : Stack object
            Rotated copy of the input stack
        """
        if resize:
            (oldY,oldX) = self.data.shape[1:3]
            M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2),angle=-angle, scale=1.0)
            r = np.deg2rad(-angle)
            newX,newY = (abs(np.sin(r)*oldY) + abs(np.cos(r)*oldX),abs(np.sin(r)*oldX) + abs(np.cos(r)*oldY))
            newX,newY = self.data.shape[1:3]
            (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
            M[0,2] += tx
            M[1,2] += ty
        else:
            newX,newY = self.data.shape[1:3]
            M = cv2.getRotationMatrix2D(center=(newX/2,newY/2),angle=-angle, scale=1.0)
        rot = copy.deepcopy(self)
        rot.data = np.zeros([np.shape(self.data)[0],int(newX),int(newY)],dtype=self.data.dtype)
        if progressbar:
            print('Rotating image stack...')
            for i in tqdm.tqdm(range(0,rot.data.shape[0])):
                rot.data[i,:,:] = cv2.warpAffine(self.data[i,:,:],M,dsize=(int(newY),int(newX)),flags=cv2.INTER_LINEAR)
            print('Rotation complete')
        else:
            for i in range(0,rot.data.shape[0]):
                rot.data[i,:,:] = cv2.warpAffine(self.data[i,:,:],M,dsize=(int(newX),int(newY)),flags=cv2.INTER_LINEAR)
        #rot.header['nx'] = np.array(np.shape(rot.data)[2])
        #rot.header['ny'] = np.array(np.shape(rot.data)[1])
        return(rot)

    def testAlign(self,xshift=0.0,tilt=0.0,thickness=None,slices=None):
        """
        Method to produce quickly reconstruct three slices from the input data for inspection of the
        quality of the alignment.

        Args
        ----------
        xshift : float
            Number of pixels by which to shift the input data.
        tilt : float
            Angle by which to rotate the input data.
        thickness : integer
            Number of pixels to include in the Z-dimension of the output reconstructions.
        slices : list
            Position of slices to use for the reconstruction.  If None, positions at 1/4, 1/2, and 3/4 of the full
            size of the stack are chosen.
        """
        if slices is None:
            mid = np.int32(self.data.shape[1]/2)
            slices = np.int32([mid/2,mid,mid+mid/2])
        temp = self.deepcopy()
        temp.data = temp.data[:,slices,:]

        shifted = temp.transStack(xshift,0,0)
   
        rec = recon.run(shifted,method='astraWBP',thickness=thickness,CUDA=False)
        
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(8,10))
        ax1.imshow(rec[0,:,:],cmap='afmhot')
        ax1.set_title('Slice %s' % str(slices[0]))
        ax1.set_axis_off()

        ax2.imshow(rec[1,:,:],cmap='afmhot')
        ax2.set_title('Slice %s' % str(slices[1]))
        ax2.set_axis_off()

        ax3.imshow(rec[2,:,:],cmap='afmhot')
        ax3.set_title('Slice %s' % str(slices[2]))
        ax3.set_axis_off()
        
        return

    def transStack(self,xshift=0.0,yshift=0.0,angle=0.0):
        """
        Method to transform the stack using the OpenCV warpAffine function

        Args
        ----------
        xshift : float
            Number of pixels by which to shift in the X dimension
        yshift : float
            Number of pixels by which to shift the stack in the Y dimension
        angle : float
            Number of degrees by which to rotate the stack about the X-Y plane
            
        Returns
        ----------
        out : Stack object
            Transformed copy of the input stack
        """
        out = self.deepcopy()
        if angle:
            image_center = tuple(np.array(out.data[0,:,:].shape)/2)
            rot_mat = cv2.getRotationMatrix2D(image_center,angle,scale=1.0)
            for i in range(0,out.data.shape[0]):
                out.data[i,:,:] = cv2.warpAffine(out.data[i,:,:], rot_mat, out.data[i,:,:].T.shape,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0.0)
        if xshift != 0.0 or yshift != 0.0:
            trans_mat = np.array([[1.,0,xshift],[0,1.,yshift]])
            for i in range(0,out.data.shape[0]):
                out.data[i,:,:] = cv2.warpAffine(out.data[i,:,:], trans_mat, out.data[i,:,:].T.shape,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0.0)
        return(out)

    def save(self,outfile=None,compression=None):
        """
        Method to save the Stack as an HDF5 file.

        Args
        ----------
        outfile : string
            Filename for output. If None, a UI will prompt for a filename.
        """
        if outfile is None:
            app = QtGui.QApplication([])            
            outfile = QtGui.QFileDialog.getSaveFileName(None,'Save tomography data','') 
        
        ext = os.path.splitext(outfile)[1]
        if ext in ['.HDF5','.hdf5','.hd5','.HD5']:
            self.datafile = io.WriteHDF5(self,outfile)
        elif ext in ['.mrc','.MRC','.rec','.REC','.ali','.ALI']:
            self.datafile = io.WriteMRC(self,outfile)
        else:
            raise ValueError('Unknown output file format. Must be .MRC, .REC, .ALI, or .HDF5')
        return
    
    def saveMovie(self,start,stop,axis='XY',fps=15,dpi=100,outfile=None,title='output.avi',clim=None,cmap='afmhot'):
        """
        Method to save the Stack as an AVI movie file.

        Args
        ----------
        start : integer
         Filename for output. If None, a UI will prompt for a filename.
        stop : integer
         Filename for output. If None, a UI will prompt for a filename.
        axis : string
         Projection axis for the output movie. Must be 'XY' (default), 'YZ' , or 'XZ'
        fps : integer
         Number of frames per second at which to create the movie.
        dpi : integer
         Resolution to save the images in the movie.
        outfile : string
         Filename for output.
        title : string
         Title to add at the top of the movie
        clim : tuple
         Upper and lower contrast limit to use for movie
        cmap : string
         Matplotlib colormap to use for movie
        """
        if clim is None:
            clim = [self.data.min(),self.data.max()]
            
        fig,ax = plt.subplots(1,figsize=(8,8))

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if title:
            ax.set_title(title)
            
        if axis == 'XY':
            im = ax.imshow(self.data[:,start,:],interpolation='none',cmap=cmap,clim=clim)
        elif axis == 'XZ':
            im = ax.imshow(self.data[start,:,:],interpolation='none',cmap=cmap,clim=clim)
        elif axis == 'YZ':
            im = ax.imshow(self.data[:,:,start],interpolation='none',cmap=cmap,clim=clim)
        else:
            raise ValueError('Unknown axis!')
        fig.tight_layout()

        def updateXY(n):
            tmp = self.data[:,n,:]
            im.set_data(tmp)
            return im

        def updateXZ(n):
            tmp = self.data[n,:,:]
            im.set_data(tmp)
            return im

        def updateYZ(n):
            tmp = self.data[:,:,n]
            im.set_data(tmp)
            return im       

        if axis == 'XY':
            ani = animation.FuncAnimation(fig,updateXY,np.arange(start,stop,1))
        elif axis == 'XZ':
            ani = animation.FuncAnimation(fig,updateXZ,np.arange(start,stop,1))
        elif axis == 'YZ':
            ani = animation.FuncAnimation(fig,updateYZ,np.arange(start,stop,1))
        else:
            raise ValueError('Axis not understood!')
            
        writer = animation.writers['ffmpeg'](fps=fps)
        ani.save(outfile,writer=writer,dpi=dpi)
        plt.close()
        return

    def show(self):
        """
        Method to show the Stack for visualization with an interactive slice slider using OpenCV"""
        
        def nothing(*arg):
            pass

        def SimpleTrackBar(Image,WindowName):
            TrackbarName='Slice'
            if (np.shape(Image)[1] > 1024) or (np.shape(Image)[2] > 1024):
                new = np.zeros([np.shape(Image)[0],1024,1024],Image.dtype)
                for i in range(0,np.size(Image,0)):
                    new[i,:,:] = cv2.resize(Image[i,:,:], (1024,1024))
                Image = new
            cv2.startWindowThread
            cv2.namedWindow(WindowName)
            cv2.createTrackbar(TrackbarName,WindowName,0,np.size(Image,0)-1,nothing)

            while True:
                TrackbarPos = cv2.getTrackbarPos(TrackbarName,WindowName)
                cv2.imshow(WindowName,Image[TrackbarPos,:,:]/np.max(Image[TrackbarPos,:,:]))
                ch = cv2.waitKey(5)
                if ch == 27:
                    break
            cv2.destroyAllWindows()

        SimpleTrackBar(self.data,'Press "ESC" to exit')
        return

def load(filename=None,reader=None):
    """
    Function to create a Stack object using data from a file

    Args
    ----------
    filename : string
        Name of file that contains data to be read.  Accepted formats (.MRC, .RAW/.RPL pair, .DM3, .DM4)
    reader : string
        File reader to employ for loading the data. If None, Hyperspy's load function is used.

    Returns
    ----------
    stack : Stack object
    """
    if filename is None:
        app = QtGui.QApplication([])
        filename = QtGui.QFileDialog.getOpenFileName(None, 'Choose files',os.getcwd(),'Tilt Series Type (*.mrc *.ali *.rec *.dm3 *.dm4)')
    #if reader is None:
    ext = os.path.splitext(filename)[1]
    if (ext in ['.HDF5','.hdf5','.hd5','.HD5']) or (reader in ['HSPY','hspy']):
        stack=io.LoadHspy(filename)
    elif (ext in ['.MRC','.mrc','.ALI','.ali','.REC','.rec']) or (reader in ['IMOD','imod']):
        stack = io.LoadIMOD(filename)
    else:
        raise ValueError("Unknown file type")
    return(stack)






