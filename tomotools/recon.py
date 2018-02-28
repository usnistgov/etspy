# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:13:35 2016

@author: aherzing
"""
import numpy as np
import astra
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import ThreadPool
from matplotlib import pylab as plt

def run(stack,method,thickness,iterations=None,constrain=None,thresh=None,CUDA=True):
    """
    Function to call appropriate sub-function to perform reconstruction of input tilt series.

    Args
    ----------
    stack :TomoStack object
       TomoStack containing the input tilt series
    method : string
        Reconstruction algorithm to use.  Must be either 'astraWBP' (default) or 'astraSIRT'
    thickness : integer
        Number of pixels in the Z dimension of the reconstructed data
    iterations : integer
        Number of iterations for the SIRT reconstruction (for astraSIRT and astraSIRT_GPU ,methods only)
    constrain : boolean
        If True, output reconstruction is constrained above value given by 'thresh'
    thresh : integer or float
        Value above which to constrain the reconstructed data
    CUDA : boolean
        If True, use the CUDA-accelerated Astra algorithms. Otherwise, use the CPU-based algorithms

    Returns
    ----------
    rec : Numpy array
        Containing the reconstructed volume
    """  
    if method == 'astraWBP':
        if not astra.astra.use_cuda() or not CUDA:
            '''ASTRA weighted-backprojection reconstruction of single slice'''
            print('Reconstructing volume using CPU-based WBP in the Astra Toolbox')
            rec = astra2D_CPU(stack,thickness,method='FBP')
            print('Reconstruction complete')
        elif astra.astra.use_cuda() or CUDA:
            '''ASTRA weighted-backprojection CUDA reconstruction of single slice'''
            print('Reconstructing volume using CUDA Accelerated WBP in the Astra Toolbox')
            rec = astra2D_CUDA(stack,thickness,method='FBP')
            print('Reconstruction complete')
        else:
            raise Exception('Error related to ASTRA Toolbox')
    elif method == 'astraSIRT':
        if not astra.astra.use_cuda() or not CUDA:
            '''ASTRA SIRT reconstruction of single slice'''
            print('Reconstructing volume using CPU-based SIRT in the Astra Toolbox')
            rec = astra2D_CPU(stack,thickness,iterations=iterations,constrain=constrain,thresh=thresh,method='SIRT')
            print('Reconstruction complete')
        elif astra.astra.use_cuda() or CUDA:
            '''ASTRA CUDA-accelerated SIRT reconstruction'''
            print('Reconstructing volume using CUDA Accelerated SIRT in the Astra Toolbox')
            if len(stack.data.shape) == 2:
                rec = astra2D_CUDA(stack,thickness,iterations=iterations,constrain=constrain,thresh=thresh,method='SIRT')
            elif len(stack.data.shape) == 3:
                rec = astraSIRT3D_CUDA(stack,thickness,iterations=iterations,constrain=constrain,thresh=thresh)
            print('Reconstruction complete')
        else:
            raise Exception('Error related to ASTRA Toolbox')
    else:
        raise ValueError('Unknown reconstruction algorithm:' + method)
    return(rec)   

def astra2D_CPU(stack,thickness,method,iterations=None,constrain=None,thresh=None):  
    """
    Perform reconstruction of 2D projections using the CPU-based algorithms in the Astra toolbox.  

    Args
    ----------
    stack :TomoStack object
       TomoStack containing the input tilt series
    thickness : integer
        Size in pixels of the Z-dimension of the output reconstruction.
    method : string
        Algorithm to use for reconstruction. Must be: FBP or SIRT
    iterations : integer
        Number of iterations for the SIRT reconstruction
    constrain : boolean
        If True, the reconstructions are constrained so that the minimum value is thresh
    thresh : integer or float
        Value above which to constrain the reconstruction
        
    Returns
    ---------
    rec : numpy array
        Array containing the reconstruction data.
    """ 

    if len(stack.data.shape) == 2:
        data = np.expand_dims(stack.data,1)
    else:
        data = stack.data
    rec = np.zeros([data.shape[1],thickness,data.shape[2]],data.dtype)
    vol_geom = astra.create_vol_geom(thickness,np.shape(data)[2])
    proj_geom = astra.create_proj_geom('parallel', 1.0, np.shape(data)[2], np.pi/180*stack.tilts) 
    proj_id = astra.create_projector('strip', proj_geom, vol_geom)
    rec_id = astra.data2d.create('-vol', vol_geom)
    sinogram_id = astra.data2d.create('-sino',proj_geom,data[:,0,:])
    cfg = astra.astra_dict(method)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id
    if constrain:
        cfg['option'] = {}    
        cfg['option']['MinConstraint'] = thresh
    
    alg_id = astra.algorithm.create(cfg)
    for i in tqdm(range(0,data.shape[1])):
        astra.data2d.store(sinogram_id,data[:,i,:])
        if method is 'FBP':
            astra.algorithm.run(alg_id)
        elif method is 'SIRT':
            astra.algorithm.run(alg_id,iterations)
        rec[i,:,:] = astra.data2d.get(rec_id)
    
    if rec.shape[0] == 1:
        rec = rec[0,:,:]
        
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    return(rec)
    
def astra2D_CUDA(stack,thickness,method,iterations=None,constrain=None,thresh=None):
    """
    Perform reconstruction of 2D projections using the CUDA-accerlated algorithms in the Astra toolbox.  

    Args
    ----------
    stack :TomoStack object
       TomoStack containing the input tilt series
    thickness : integer
        Size in pixels of the Z-dimension of the output reconstruction.
    method : string
        Algorithm to use for reconstruction. Must be: FBP or SIRT
    iterations : integer
        Number of iterations for the SIRT reconstruction
    constrain : boolean
        If True, the reconstructions are constrained so that the minimum value is thresh
    thresh : integer or float
        Value above which to constrain the reconstruction
        
    Returns
    ---------
    rec : numpy array
        Array containing the reconstruction data.
    """ 
    
    if len(stack.data.shape) == 2:
        data = np.expand_dims(stack.data,1)
    else:
        data = stack.data
    rec = np.zeros([data.shape[1],thickness,data.shape[2]],data.dtype)
    vol_geom = astra.create_vol_geom(thickness,np.shape(data)[2])
    proj_geom = astra.create_proj_geom('parallel', 1.0, np.shape(data)[2], np.pi/180*stack.tilts) 
    proj_id = astra.create_projector('strip', proj_geom, vol_geom)
    rec_id = astra.data2d.create('-vol', vol_geom)
    sinogram_id = astra.data2d.create('-sino',proj_geom,data[:,0,:])
    cfg = astra.astra_dict(method)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id
    if constrain:
        cfg['option'] = {}    
        cfg['option']['MinConstraint'] = thresh
    
    alg_id = astra.algorithm.create(cfg)
    for i in tqdm(range(0,data.shape[1])):
        astra.data2d.store(sinogram_id,data[:,i,:])
        if method is 'FBP':
            astra.algorithm.run(alg_id)
        elif method is 'SIRT':
            astra.algorithm.run(alg_id,iterations)
        rec[i,:,:] = astra.data2d.get(rec_id)
    
    if rec.shape[0] == 1:
        rec = rec[0,:,:]
        
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    return rec
    
def astraSIRT3D_CUDA(stack,thickness=512,iterations=30,chunksize=128,constrain=False,thresh=0,showProgressBar=True):
    """
    Perform reconstruction using the CUDA-accelerated SIRT algorithm in the Astra toolbox.  Data must be processed in 
    chunks so as to not produce memory-limited crashes.  Kernel will also crash if this is called on a machine
    with no CUDA-capable GPU

    Args
    ----------
    stack :TomoStack object
       TomoStack containing the tilt series data
    thickness : integer
        Size in pixels of the Z-dimension of the output reconstruction.
    iterations : integer
        Number of iterations for the SIRT reconstruction
    chunksize : integer
        Size of data chunk to use for each reconstruction run 
    constrain : boolean
        If True, the reconstructions are constrained so that the minimum value is thresh
    thresh : integer or float
        Value above which to constrain the reconstruction
    showProgressBar : boolean
        If True, output a progress bar to the terminal using the tqdm library

    Returns
    ---------
    volume : numpy array
        Array containing the reconstruction data.
    """
    data = np.rollaxis(stack.data,1)    
    rec = np.zeros([np.shape(data)[0],thickness,np.shape(data)[2]],data.dtype)
    nchunks = np.shape(data)[0]/128
    if nchunks < 1:
        nchunks = 1
        chunksize = np.shape(data)[0]

    for i in tqdm(range(0,np.int32(nchunks))):
        chunk = data[i*chunksize:(i+1)*chunksize,:,:]
        vol_geom = astra.create_vol_geom(thickness,np.shape(data)[2],chunksize)
        proj_geom = astra.create_proj_geom('parallel3d', 1, 1, chunksize, np.shape(data)[2], np.pi/180*stack.tilts)
        data_id = astra.data3d.create('-proj3d',proj_geom,chunk)
        rec_id = astra.data3d.create('-vol', vol_geom)
        
        cfg = astra.astra_dict('SIRT3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = data_id
        if constrain:
            cfg['option'] = {}    
            cfg['option']['MinConstraint'] = thresh
        
        alg_id = astra.algorithm.create(cfg)
        
        astra.algorithm.run(alg_id, iterations)
        
        rec[i*chunksize:(i+1)*chunksize,:,:] = astra.data3d.get(rec_id)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(data_id)
    if rec.shape[0] == 1:
        rec = np.flipud(rec[0,:,:])
    return rec

    
def calculateParallel(slices,processors,thickness,tilts,method='astraWBP',iterations=None,constrain=False,thresh=0):
    """
    Perform reconstruction (either WBP or SIRT) using parallel processing

    Args
    ----------
    slices : numpy array
        Array containing the tilt series data
    processors : integer
        Number of processors to use for the calculation. Determined automatically using os.cpu_count()
    thickness : integer
        Size in pixels of the Z-dimension of the output reconstruction.
    tilts : list
        List containing the tilt orientation of each slice of the input data
    method : string
        Reconstruction algorithm to use. Must be either 'astraWBP' or 'astraSIRT'
    iterations : integer
        Number of iterations for the SIRT reconstruction
    constrain : boolean
        If True, the reconstructions are constrained so that the minimum value is thresh
    thresh : integer or float
        Value above which to constrain the reconstruction
        
    Returns
    ---------
    recs : numpy array
        Array containing the reconstruction data from a chunk of data.
    """   
    if method == 'astraWBP':
        astraPartial = partial(astraWBP,thickness=thickness)
    elif method == 'astraSIRT':          
        astraPartial = partial(astraSIRT,thickness=thickness,tilts=tilts,iterations=iterations,constrain=constrain,thresh=thresh)
    else:
        raise ValueError("Unknown reconstruction algorithm.  Must be 'astraSIRT' or 'astraWBP'")
    pool = ThreadPool(processors)
    recs = pool.map(astraPartial,slices)
    pool.close()
    pool.join()
    return(recs)

def errorSIRTCPU(data,thickness,tilts,minError,maxiters=300):
        """
        Function to calculate the number of SIRT iterations required to produce a minimum change
        between successive iterations.  This method uses the CPU-based SIRT reconstruction
        algorithm of the Astra toolbox.

        Args
        ----------
        data : numpy array
            Sinogram extracted from a tilt series at a user defined location or the central Y-slice
            if not provided
        thickness : integer
            Size in pixels of the Z-dimension of the output reconstruction.
        tilts : list
            List of floats indicating the specimen tilt for each image in the stack
        minError : float
            Percentage change between successive iterations that terminates the algorithm
        maxiters : integer
            Maximum number of iterations to perform if minError is not met
        """
        vol_geom = astra.create_vol_geom(thickness,np.shape(data)[1])
        proj_geom = astra.create_proj_geom('parallel', 1.0, np.shape(data)[1], np.pi/180*tilts)     
        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
        rec_id = astra.data2d.create('-vol', vol_geom)

        sinogram_id = astra.data2d.create('-sino',proj_geom,data)

        cfg = astra.astra_dict('SIRT')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ProjectorId'] = proj_id

        for i in range(0,maxiters):
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id,1)
            rec = astra.data2d.get(rec_id)
            sinogram_id,sinogram = astra.create_sino(rec,proj_id)
            astra.data2d.store(sinogram_id,data-sinogram)
            if i == 0:
                error = np.sum((data-sinogram)**2)
            else: 
                error = np.append(error,np.sum((data-sinogram)**2))
                if i == 1:
                    diff = np.array(np.abs((error[i] - error[i-1])/error[i-1]))
                else: 
                    diff = np.append(diff,np.abs(((error[i] - error[i-1])/error[i-1])))
                    if 100*diff[-1:]<minError:
                        print('Error < %s%% after %s iterations' % (str(minError),str(i+1)))
                        break
        
        plt.imshow(rec,clim=[0,rec.max()/20],cmap='afmhot')
        
        fig,ax = plt.subplots(1)
        ax.plot(100*diff)
        ax.set_title('SIRT Error')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('% Change from Previous')
        
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(proj_id)
        return
    
def errorSIRTGPU(stack,thickness,nIters,N):
    """
    Method to calculate the number of SIRT iterations required to produce a minimum change
    between successive iterations.  This method uses the GPU-based SIRT reconstruction
    algorithm of the Astra toolbox.

    Args
    ----------
    stack :TomoStack object
       TomoStack containing the tilt series data
    thickness : integer
        Size in pixels of the Z-dimension of the output reconstruction.
    start : integer
        Number of iterations for the initial reconstruction
    step : integer
        Increment for the number of iterations between each calculation
    N : integer
        Location of the slice to use for error calculation. If None, the middle slice is chosen.
    errormin : float
        Percentage change between successive iterations that terminates the algorithm
    constrain : boolean
        If True, the reconstructions are constrained so that the minimum value is thresh
    thresh : integer or float
        Value above which to constrain the reconstruction

    Returns
    ---------
    error : numpy array
        Array containing the error at each iteration between the projected reconstruction and the
        input sinogram.  Error is given by the sum of the squared difference between the two sinograms.
    diff : numpy array
        Array showing the iterative difference between slices
    rec : numpy array
        Final reconstructed image
    """
    data = stack.data[:,N,:]
    vol_geom = astra.create_vol_geom(thickness,np.shape(data)[1])
    proj_geom = astra.create_proj_geom('parallel', 1.0, np.shape(data)[1], np.pi/180*stack.tilts)     
    proj_id = astra.create_projector('strip', proj_geom, vol_geom)
    rec_id = astra.data2d.create('-vol', vol_geom)
    sinogram_id = astra.data2d.create('-sino',proj_geom,data)
    
    cfg = astra.astra_dict('SIRT_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    
    error = np.zeros(nIters)
    for i in range(nIters):
      # Run a single iteration
      astra.algorithm.run(alg_id, 1)
      error[i] = astra.algorithm.get_res_norm(alg_id)
      rec = astra.data2d.get(rec_id)
      #sino = proj(rec,stack.tilts)
      if i > 0:
          if i == 1:
              diff = np.array(np.abs((error[i] - error[i-1])/error[i-1]))
          else: 
              diff = np.append(diff,np.abs(((error[i] - error[i-1])/error[i-1])))
    diff = 100 * diff
    return(error,diff,rec)
    
def run2(stack,method,thickness,iterations,constrain,thresh,CUDA=True):
    """
    Function to call appropriate sub-function to perform reconstruction of input tilt series.

    Args
    ----------
    data :TomoStack object
       TomoStack containing the input tilt series
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
    rec :TomoStack object
       TomoStack containing the reconstructed volume
    """  
    if len(np.shape(stack.data)) == 2:
        if method == 'astraWBP':
            if not astra.astra.use_cuda() or not CUDA:
                '''ASTRA weighted-backprojection reconstruction of single slice'''
                print('Reconstructing single slice using CPU-based WBP in the Astra Toolbox')
                rec = astra2D_CPU(stack,thickness,method='FBP')
                print('Reconstruction complete')
            elif astra.astra.use_cuda() or CUDA:
                '''ASTRA weighted-backprojection CUDA reconstruction of single slice'''
                print('Reconstructing single-slice using CUDA Accelerated WBP in the Astra Toolbox')
                rec = astra2D_CUDA(stack,thickness,method='FBP')
                print('Reconstruction complete')
            else:
                raise Exception('Error related to ASTRA Toolbox')
        elif method == 'astraSIRT':
            if not astra.astra.use_cuda() or not CUDA:
                '''ASTRA SIRT reconstruction of single slice'''
                print('Reconstructing single-slice using SIRT in the Astra Toolbox')
                rec = astra2D_CPU(stack,thickness,iterations=iterations,constrain=constrain,thresh=thresh,method='SIRT')
                print('Reconstruction complete')
            elif astra.astra.use_cuda():
                '''ASTRA SIRT_CUDA reconstruction of single slice'''
                print('Reconstructing single-slice using CUDA Accelerated SIRT in the Astra Toolbox')
                rec = astra2D_CUDA(stack,thickness,iterations=iterations,constrain=constrain,thresh=thresh,method='SIRT')
                print('Reconstruction complete')
            else:
                raise Exception('Error related to ASTRA Toolbox')
        else:
            raise ValueError('Unknown reconstruction method: "%s"' % method)
    elif len(np.shape(stack.data)) == 3:
        if method == 'astraSIRT':
            if not astra.astra.use_cuda() or not CUDA:
                rec = np.zeros([stack.data.shape[1],thickness,stack.data.shape[2]],np.float32)
                print('Reconstructing using CPU-based SIRT in the Astra Toolbox')
                for i in tqdm(range(np.shape(stack.data)[1])):
                    rec[i,:,:] = astra2D_CPU(stack,thickness,row=i,iterations=iterations,constrain=constrain,thresh=thresh,method='SIRT')
#                processors = np.int(os.cpu_count())
#                nchunks = np.size(stack.data,1)/processors
#                remaining = np.size(stack.data,1)-int(nchunks)*processors
#                if remaining != 0:
#                    totalchunks = int(nchunks)+1
#                else: 
#                    totalchunks = int(nchunks)
#                #rec = np.zeros([np.shape(data.data)[1],thickness,np.shape(data.data)[2]],data.data.dtype)
#                if nchunks < 1:
#                    print('Reconstructing using SIRT in the Astra Toolbox')
#                    for i in tqdm(range(np.shape(stack.data)[1])):
#                        rec[i,:,:] = astraSIRT(stack.data[:,i,:],thickness,stack.tilts,iterations,constrain)
#                else:
#                    print('Reconstructing with MultiThreaded SIRT in the Astra Toolbox')
#                    for j in tqdm(range(0,totalchunks)):
#                        #print('Processing chunk %s of %s' % (j+1,totalchunks))
#                        if j+1 <= nchunks:
#                            slices = [None]*processors
#                        else:
#                            slices = [None]*remaining
#                        for i in range(0,len(slices)):
#                            slices[i] = stack.data[:,processors*j+i,:]
#                        result = calculateParallel(slices,processors=processors,thickness=thickness,tilts=stack.tilts,method='astraSIRT',iterations=iterations,constrain=constrain)
#                        for k in range(0,len(result)):    
#                            rec[processors*j+k,:,:] = result[k]     
                print('Reconstruction complete')
            elif astra.astra.use_cuda() or CUDA:
                print('Reconstructing with GPU-accelerated SIRT in the Astra Toolbox')
                rec = astraSIRT3D_CUDA(stack,thickness,iterations,constrain,thresh)
                print('Reconstruction complete')
            else:
                raise Exception('Error related to ASTRA Toolbox')
        elif method=='astraWBP':
            if not astra.astra.use_cuda() or not CUDA:
                rec = np.zeros([stack.data.shape[1],thickness,stack.data.shape[2]],np.float32)
                print('Reconstructing using CPU-based WBP in the Astra Toolbox')
                for i in tqdm(range(np.shape(stack.data)[1])):
                    rec[i,:,:] = astra2D_CPU(stack,thickness,row=i,method='FBP')
#            processors = np.int(os.cpu_count())
#            nchunks = np.size(stack.data,1)/processors
#            remaining = np.size(stack.data,1)-int(nchunks)*processors
#            if remaining != 0:
#                totalchunks = int(nchunks)+1
#            else: 
#                totalchunks = int(nchunks)
#            if nchunks < 1:
#                print('Reconstructing using WBP in the Astra Toolbox')
#                for i in tqdm(range(np.shape(stack.data)[1])):
#                    rec[i,:,:] = astraWBP(stack,thickness,row=i)
#            else:
#                print('Reconstructing with MultiThreaded WBP in the Astra Toolbox')
#                for j in tqdm(range(0,totalchunks)):
#                    if j+1 <= nchunks:
#                        slices = [None]*processors
#                    else:
#                        slices = [None]*remaining
#                    for i in range(0,len(slices)):
#                        slices[i] = stack.data[:,processors*j+i,:]
#                    result = calculateParallel(slices,processors=processors,thickness=thickness,tilts=stack.tilts,method='astraWBP')
#                    for k in range(0,len(result)):    
#                        rec[processors*j+k,:,:] = result[k]    
                print('Reconstruction complete')
            elif astra.astra.use_cuda() or CUDA:
                rec = np.zeros([stack.data.shape[1],thickness,stack.data.shape[2]],np.float32)
                print('Reconstructing using GPU-based WBP in the Astra Toolbox')
                for i in tqdm(range(np.shape(stack.data)[1])):
                    rec[i,:,:] = astra2D_CUDA(stack,thickness,row=i,method='FBP')
                print('Reconstruction complete')
        else:
            raise ValueError('Unknown reconstruction method: "%s"' % method)
    else:
        raise ValueError('Unknown data structure with dimension: %s' % str(stack.data.shape))
    return(rec) 
