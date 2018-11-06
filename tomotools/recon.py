# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:13:35 2016

@author: aherzing
"""
import tomopy
import numpy as np
import astra


def run(stack, method, rot_center=None, iterations=None, constrain=None, thresh=None, cuda=True):
    """
    Function to call appropriate sub-function to perform reconstruction of input tilt series.

    Args
    ----------
    stack :TomoStack object
       TomoStack containing the input tilt series
    method : string
        Reconstruction algorithm to use.  Must be either 'FBP' (default) or 'SIRT'
    rot_center : float
        Location of the rotation center.  If None, position is assumed to be the
        center of the image.
    iterations : integer (only required for SIRT)
        Number of iterations for the SIRT reconstruction (for SIRT methods only)
    constrain : boolean
        If True, output reconstruction is constrained above value given by 'thresh'
    thresh : integer or float
        Value above which to constrain the reconstructed data
    cuda : boolean
        If True, use the CUDA-accelerated Astra algorithms. Otherwise, use the CPU-based algorithms

    Returns
    ----------
    rec : Numpy array
        Containing the reconstructed volume
    """
    theta = stack.axes_manager[0].axis*np.pi/180
    if method == 'FBP':
        if not astra.astra.use_cuda() or not cuda:
            '''ASTRA weighted-backprojection reconstruction of single slice'''
            print('Reconstructing volume using CPU-based WBP in the Astra Toolbox')
            options = {'proj_type': 'linear', 'method': 'FBP'}
            rec = tomopy.recon(stack.data, theta, center=rot_center, algorithm=tomopy.astra, options=options)
            print('Reconstruction complete')
        elif astra.astra.use_cuda() or cuda:
            '''ASTRA weighted-backprojection CUDA reconstruction of single slice'''
            print('Reconstructing volume using CUDA Accelerated WBP in the Astra Toolbox')
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
            rec = tomopy.recon(stack.data, theta, center=rot_center, algorithm=tomopy.astra, options=options)
            print('Reconstruction complete')
        else:
            raise Exception('Error related to ASTRA Toolbox')
    elif method == 'SIRT':
        if not iterations:
            iterations = 20
        if not astra.astra.use_cuda() or not cuda:
            '''ASTRA SIRT reconstruction of single slice'''
            print('Reconstructing volume using CPU-based SIRT in the Astra Toolbox')
            if constrain:
                if not thresh:
                    thresh = 0
                extra_options = {'MinConstraint': thresh}
                options = {'proj_type': 'linear', 'method': 'SIRT', 'num_iter': iterations,
                           'extra_options': extra_options}
            else:
                options = {'proj_type': 'linear', 'method': 'SIRT', 'num_iter': iterations}
            rec = tomopy.recon(stack.data, theta, center=rot_center, algorithm=tomopy.astra, options=options)
            print('Reconstruction complete')
        elif astra.astra.use_cuda() or cuda:
            '''ASTRA CUDA-accelerated SIRT reconstruction'''
            print('Reconstructing volume using CUDA Accelerated SIRT in the Astra Toolbox')
            if constrain:
                if not thresh:
                    thresh = 0
                extra_options = {'MinConstraint': thresh}
                options = {'proj_type': 'cuda', 'method': 'SIRT_CUDA', 'num_iter': iterations,
                           'extra_options': extra_options}
            else:
                options = {'proj_type': 'cuda', 'method': 'SIRT_CUDA', 'num_iter': iterations}
            rec = tomopy.recon(stack.data, theta, center=rot_center, algorithm=tomopy.astra, options=options)
            print('Reconstruction complete')
        else:
            raise Exception('Error related to ASTRA Toolbox')
    else:
        raise ValueError('Unknown reconstruction algorithm:' + method)
    return rec

# def errorSIRTCPU(data,thickness,tilts,minError,maxiters=300):
#         """
#         Function to calculate the number of SIRT iterations required to produce a minimum change
#         between successive iterations.  This method uses the CPU-based SIRT reconstruction
#         algorithm of the Astra toolbox.
#
#         Args
#         ----------
#         data : numpy array
#             Sinogram extracted from a tilt series at a user defined location or the central Y-slice
#             if not provided
#         thickness : integer
#             Size in pixels of the Z-dimension of the output reconstruction.
#         tilts : list
#             List of floats indicating the specimen tilt for each image in the stack
#         minError : float
#             Percentage change between successive iterations that terminates the algorithm
#         maxiters : integer
#             Maximum number of iterations to perform if minError is not met
#         """
#         vol_geom = astra.create_vol_geom(thickness,np.shape(data)[1])
#         proj_geom = astra.create_proj_geom('parallel', 1.0, np.shape(data)[1], np.pi/180*tilts)
#         proj_id = astra.create_projector('strip', proj_geom, vol_geom)
#         rec_id = astra.data2d.create('-vol', vol_geom)
#
#         sinogram_id = astra.data2d.create('-sino',proj_geom,data)
#
#         cfg = astra.astra_dict('SIRT')
#         cfg['ReconstructionDataId'] = rec_id
#         cfg['ProjectionDataId'] = sinogram_id
#         cfg['ProjectorId'] = proj_id
#
#         for i in range(0,maxiters):
#             alg_id = astra.algorithm.create(cfg)
#             astra.algorithm.run(alg_id,1)
#             rec = astra.data2d.get(rec_id)
#             sinogram_id,sinogram = astra.create_sino(rec,proj_id)
#             astra.data2d.store(sinogram_id,data-sinogram)
#             if i == 0:
#                 error = np.sum((data-sinogram)**2)
#             else:
#                 error = np.append(error,np.sum((data-sinogram)**2))
#                 if i == 1:
#                     diff = np.array(np.abs((error[i] - error[i-1])/error[i-1]))
#                 else:
#                     diff = np.append(diff,np.abs(((error[i] - error[i-1])/error[i-1])))
#                     if 100*diff[-1:]<minError:
#                         print('Error < %s%% after %s iterations' % (str(minError),str(i+1)))
#                         break
#
#         plt.imshow(rec,clim=[0,rec.max()/20],cmap='afmhot')
#
#         fig,ax = plt.subplots(1)
#         ax.plot(100*diff)
#         ax.set_title('SIRT Error')
#         ax.set_xlabel('Iteration')
#         ax.set_ylabel('% Change from Previous')
#
#         astra.algorithm.delete(alg_id)
#         astra.data2d.delete(rec_id)
#         astra.data2d.delete(sinogram_id)
#         astra.data2d.delete(proj_id)
#         return
#
# def errorSIRTGPU(stack,thickness,nIters,N):
#     """
#     Method to calculate the number of SIRT iterations required to produce a minimum change
#     between successive iterations.  This method uses the GPU-based SIRT reconstruction
#     algorithm of the Astra toolbox.
#
#     Args
#     ----------
#     stack :TomoStack object
#        TomoStack containing the tilt series data
#     thickness : integer
#         Size in pixels of the Z-dimension of the output reconstruction.
#     start : integer
#         Number of iterations for the initial reconstruction
#     step : integer
#         Increment for the number of iterations between each calculation
#     N : integer
#         Location of the slice to use for error calculation. If None, the middle slice is chosen.
#     errormin : float
#         Percentage change between successive iterations that terminates the algorithm
#     constrain : boolean
#         If True, the reconstructions are constrained so that the minimum value is thresh
#     thresh : integer or float
#         Value above which to constrain the reconstruction
#
#     Returns
#     ---------
#     error : numpy array
#         Array containing the error at each iteration between the projected reconstruction and the
#         input sinogram.  Error is given by the sum of the squared difference between the two sinograms.
#     diff : numpy array
#         Array showing the iterative difference between slices
#     rec : numpy array
#         Final reconstructed image
#     """
#     tilts = stack.axes_manager[0].axis*np.pi/180
#     data = stack.data[:,N,:]
#     vol_geom = astra.create_vol_geom(thickness,np.shape(data)[1])
#     proj_geom = astra.create_proj_geom('parallel', 1.0, np.shape(data)[1], tilts)
#     proj_id = astra.create_projector('strip', proj_geom, vol_geom)
#     rec_id = astra.data2d.create('-vol', vol_geom)
#     sinogram_id = astra.data2d.create('-sino',proj_geom,data)
#
#     cfg = astra.astra_dict('SIRT_CUDA')
#     cfg['ReconstructionDataId'] = rec_id
#     cfg['ProjectionDataId'] = sinogram_id
#     cfg['ProjectorId'] = proj_id
#     alg_id = astra.algorithm.create(cfg)
#
#     error = np.zeros(nIters)
#     for i in range(nIters):
#       # Run a single iteration
#       astra.algorithm.run(alg_id, 1)
#       error[i] = astra.algorithm.get_res_norm(alg_id)
#       rec = astra.data2d.get(rec_id)
#       #sino = proj(rec,stack.tilts)
#       if i > 0:
#           if i == 1:
#               diff = np.array(np.abs((error[i] - error[i-1])/error[i-1]))
#           else:
#               diff = np.append(diff,np.abs(((error[i] - error[i-1])/error[i-1])))
#     diff = 100 * diff
#     return error,diff,rec
