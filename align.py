import numpy as np
import cv2
import copy
from scipy import optimize,ndimage
import pylab as plt
import warnings
import astra
import tqdm

def getPoints(data,numpoints=3):
    """
    Function to display the central image in a stack and prompt the user to choose
    three locations by mouse click.  Once three locations have been clicked, the
    window closes and the function returns the coordinates

    Args
    ----------
    data : Numpy array
        Tilt series datastack
    numpoints : integer
        Number of points to use in fitting the tilt axis
    
    Returns
    ----------
    coords : Numpy array
        array containing the XY coordinates selected interactively by the user   
    """
    warnings.filterwarnings('ignore')
    plt.figure(num='Align Tilt',frameon=False)            
    if len(data.shape) == 3:
        plt.imshow(data[np.int(data.shape[0]/2),:,:],cmap='gray')
    else:
        plt.imshow(data,cmap='gray')
    plt.title('Choose %s points for tilt axis alignment....' % str(numpoints))
    coords = np.array(plt.ginput(numpoints,timeout=0,show_clicks=True))     
    plt.close()
    return coords

def proj(data,tilts):
    """
    Function to project a sinogram using the CUDA accelerated radon transform algorithm in the ASTRA toolbox

    Args
    ----------
    data : Numpy array
        Volumetric data to be projected into a sinogram
    tilts : list
        List of floats providing the tilts to use for projection.
    
    Returns
    ----------
    sinogram : Numpy array
        Sinogram calculated using the ASTRA radon transform algorithm   
    """
    vol_geom = astra.create_vol_geom(np.shape(data)[0],np.shape(data)[1])
    proj_geom = astra.create_proj_geom('parallel', 1, np.shape(data)[1], np.pi/180*tilts)
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
    sinogram_id,sinogram = astra.create_sino(data,proj_id)
    return(sinogram)
    
def recon(data,tilts,thickness,method='SIRT_CUDA',iterations=10,constrain=True,thresh=0):
    """
    Function to reconstruct a sinogram via SIRT_CUDA algorithm in the ASTRA toolbox'''    

    Args
    ----------
    data : Numpy array
        Sinogram
    tilts : list
        List of floats providing the tilts to use for projection.
    thickness : integer
        Number of pixels in the Z-dimension of the volume to receive the reconstruction
    method : list
        Reconstruction algorithm to provide Astra toolbox.
    iterations : integer
        Number of SIRT iterations to run.
    constrain : boolean
        If True, output reconstruction is constrained above value given by 'thresh'
    thresh : integer or float
        Value above which to constrain the reconstructed data

    Returns
    ----------
    volume : numpy array
        Array containing the reconstructed volume
    """
    vol_geom = astra.create_vol_geom(thickness,np.shape(data)[1])
    proj_geom = astra.create_proj_geom('parallel', 1, np.shape(data)[1], np.pi/180*tilts)
    data_id = astra.data2d.create('-sino',proj_geom,data)
    rec_id = astra.data2d.create('-vol', vol_geom)

    cfg = astra.astra_dict(method)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = data_id
    cfg['option'] = {}    
    if method == 'SIRT_CUDA':
        if constrain:
            cfg['option']['MinConstraint'] = thresh
        #cfg['option']['GPUindex'] = 0
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)
    else:
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

    volume = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(data_id)
    return(volume)

def recon3D(slices,thickness,tilts,iterations=10,thresh=0):
    """
    Function to reconstruct a series sinogram via SIRT_CUDA algorithm in the ASTRA toolbox'''    

    Args
    ----------
    slices : Numpy array
        3-D array containing a series of sinograms
    tilts : list
        List of floats providing the tilts to use for projection.
    thickness : integer
        Number of pixels in the Z-dimension of the volume to receive the reconstruction
    iterations : integer
        Number of SIRT iterations to run.
    constrain : boolean
        If True, output reconstruction is constrained above value given by 'thresh'
    thresh : integer or float
        Value above which to constrain the reconstructed data

    Returns
    ----------
    volume : numpy array
        Array containing the reconstructed volume
    """   
    vol_geom = astra.create_vol_geom(thickness,slices.shape[2],slices.shape[0])
    proj_geom = astra.create_proj_geom('parallel3d', 1, 1, slices.shape[0], slices.shape[2], np.pi/180*tilts)
    data_id = astra.data3d.create('-proj3d',proj_geom,slices)
    rec_id = astra.data3d.create('-vol', vol_geom)

    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = data_id
    cfg['option'] = {}    
    cfg['option']['MinConstraint'] = thresh
    #cfg['option']['GPUindex'] = 0

    alg_id = astra.algorithm.create(cfg)

    astra.algorithm.run(alg_id, iterations)

    volume = astra.data3d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(data_id)
    return(volume)

def proj3D(volume,tilts):
    """
    Function to project a series of sinogram using the CUDA accelerated radon transform algorithm in the ASTRA toolbox

    Args
    ----------
    volume : Numpy array
        Volumetric data to be projected into a sinogram
    tilts : list
        List of floats providing the tilts to use for projection.
    
    Returns
    ----------
    sinogram : Numpy array
        Sinograms calculated using the ASTRA radon transform algorithm   
    """
    vol_geom = astra.create_vol_geom(volume.shape[1],volume.shape[2],volume.shape[0])
    proj_geom = astra.create_proj_geom('parallel3d', 1, 1, volume.shape[0] , volume.shape[2], np.pi/180*tilts)
    sinogram_id,sinogram = astra.create_sino3d_gpu(volume,proj_geom,vol_geom)
    return(sinogram)

def rigidECC(stack,start):
    """
    Function to compute the shifts necessary to spatially register a stack of images.
    Shifts are determined by the OpenCV findTransformECC algorithm.  Shifts are then
    applied and the aligned stack is returned.
    
    Args
    ----------
    stack : Numpy array
        3-D numpy array containing the tilt series data
    start : integer
        Position in tilt series to use as starting point for the alignment. If None, the central projection is used.

    Returns
    ----------
    out : Stack object
        Spatially registered copy of the input stack
    """
    number_of_iterations = 1000
    termination_eps = 1e-3    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    old_trans = np.array([[1.,0,0],[0,1.,0]])
    out = copy.deepcopy(stack)
    out.data = np.zeros(stack.data.shape,stack.data.dtype)
    out.shifts = out.data.shape[0]*[None]
    if start is None:
        start = np.argmin(np.abs(stack.tilts))
    out.data[start,:,:] = stack.data[start,:,:]
    out.shifts[start] = old_trans
    
    for i in tqdm.tqdm(range(start+1,stack.data.shape[0])):
        warp_matrix = np.eye(2, 3, dtype=np.float32)        
        (cc,trans) = cv2.findTransformECC(stack.data[i,:,:],stack.data[i-1,:,:],warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        trans[:,2] = trans[:,2] + old_trans[:,2]
        out.data[i,:,:] = cv2.warpAffine(stack.data[i,:,:],trans,stack.data[i,:,:].T.shape,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0.0)
        out.shifts[i] = trans
        old_trans = trans
    
    if start != 0:
        old_trans = np.array([[1.,0,0],[0,1.,0]])
        for i in tqdm.tqdm(range(start-1,-1,-1)):
            warp_matrix = np.eye(2, 3, dtype=np.float32)        
            (cc,trans) = cv2.findTransformECC(stack.data[i,:,:],stack.data[i+1,:,:],warp_matrix, cv2.MOTION_TRANSLATION, criteria)
            trans[:,2] = trans[:,2] + old_trans[:,2]
            out.data[i,:,:] = cv2.warpAffine(stack.data[i,:,:],trans,stack.data[i,:,:].T.shape,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0.0)
            out.shifts[i] = trans
            old_trans = trans
    print('Spatial registration by ECC complete')
    return(out)
    
def rigidPC(stack,start):
    """
    Function to compute the shifts necessary to spatially register a stack of images.
    Shifts are determined by the OpenCV phaseCorrelate algorithm.  Shifts are then
    applied and the aligned stack is returned.
    
    Args
    ----------
    stack : Numpy array
        3-D numpy array containing the tilt series data
    start : integer
        Position in tilt series to use as starting point for the alignment. If None, the central projection is used.

    Returns
    ----------
    out : Stack object
        Spatially registered copy of the input stack
    """      
    old_trans = np.array([[1.,0,0],[0,1.,0]])
    out = copy.deepcopy(stack)
    out.data = np.zeros(stack.data.shape,stack.data.dtype)
    out.shifts = out.data.shape[0]*[None]
    if start is None:
        start = np.argmin(np.abs(stack.tilts))
    out.data[start,:,:] = stack.data[start,:,:]
    out.shifts[start] = old_trans
    
    for i in tqdm.tqdm(range(start+1,stack.data.shape[0])):
        trans = np.array([[1.,0,0],[0,1.,0]])       
        trans[:,2] = cv2.phaseCorrelate(stack.data[i,:,:],stack.data[i-1,:,:])[0] + old_trans[:,2]       
        out.data[i,:,:] = cv2.warpAffine(stack.data[i,:,:],trans,stack.data[i,:,:].T.shape,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0.0)
        out.shifts[i] = trans
        old_trans = trans
    
    if start != 0:
        old_trans = np.array([[1.,0,0],[0,1.,0]])
        for i in tqdm.tqdm(range(start-1,-1,-1)):
            trans = np.array([[1.,0,0],[0,1.,0]])       
            trans[:,2] = cv2.phaseCorrelate(stack.data[i,:,:],stack.data[i+1,:,:])[0] + old_trans[:,2] 
            out.data[i,:,:] = cv2.warpAffine(stack.data[i,:,:],trans,stack.data[i,:,:].T.shape,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0.0)
            out.shifts[i] = trans
            old_trans = trans
    print('Spatial registration by PC complete')
    return(out)
    
def tiltCorrect(stack,offset = 0):
    """
    Function to perform automated determination of the tilt axis of a Stack by tracking the 
    center of mass (CoM) and comparing it to the path expected for an ideal cylinder
    
    Args
    ----------
    stack : Stack object
        3-D numpy array containing the tilt series data
    offset : integer
        Not currently used
        
    Returns
    ----------
    out : Stack object
        Copy of the input stack after rotation and translation to center and make the tilt
        axis vertical
    """
    def sinocalc(data,y):
        """
        Function to extract sinograms at stack positions chosen by user via getPoints() function
        and track the center of mass (CoM) as a function of angle for each.

        Args
        ----------
        data : Numpy array
            3-D numpy array containing the tilt series data
        y : Numpy array
            Array containing the coordinates selected by the user in getPoints()
            
        Returns
        ----------
        outvals : Numpy array
            Array containing the center of mass as a function of tilt for the selected sinograms
        """
        def CoM(row):
            """
            Compute the center of mass for a row of pixels

            Args
            ----------
            row : Numpy array
                Row of pixels extracted from a sinogram
                
            Returns
            ----------
            value : float
                Center of mass of the input row
            """
            size = np.size(row)
            value = 0.0    
            for i in range (0,size):
                value = value + row[i]*(i+1)
            value = value/np.sum(row)
            return value
            
        outvals = np.zeros([np.size(data,axis=0),3])
        sinotop = data[:,:,y[0]]
        sinomid = data[:,:,y[1]]    
        sinobot = data[:,:,y[2]]
        
        for i in range (0,np.size(data,0)):
            outvals[i][0] = CoM(sinotop[i,:])
            outvals[i][1] = CoM(sinomid[i,:])
            outvals[i][2] = CoM(sinobot[i,:])
            
        return outvals
    
    def fitCoMs(tilts,CoMs):
        """
        Function to fit the motion of calculated centers-of-mass in a sinogram to a 
        sinusoidal function: (r0-A*cos(tilt)-B*sin(tilt)) as would be expected 
        for an ideal cylinder. Return the coefficient of the fit equation for use
        in fitTiltAxis

        Args
        ----------
        tilts : Numpy array
            Array containing the stage tilt at each row in the sinogram
        CoMs : Numpy array
            Array containing the calculated center of mass as a function of tilt
            for the sinogram
            
        Returns
        ----------
        coeffs : Numpy array
            Coefficients (r0 , A , and B) resulting from the fit
        """
        guess = [0.0,0.0,0.0]
        def func(x,r0,A,B):
            return(r0 - A*np.cos(x)-B*np.sin(x))
        coeffs,covars = optimize.curve_fit(func, tilts, np.int16(CoMs), guess)
        return coeffs
    
    def fitTiltAxis(y,vals):
        """
        Function to fit the coefficients calculated by fitCoMs() at each of the three user 
        chosen positions to a linear function to determine the necessary rotation
        to vertically align the tilt axis

        Args
        ----------
        y : Numpy array
            Horixonal coordinates from which the sinograms were extracted
        vals : Numpy array
            Array containing the r0 coefficient calculated for each sinogram by fitCoMs
            
        Returns
        ----------
        coeffs : Numpy array
            Coefficients (m and b) resulting from the fit
        """
        guess = [0.0,0.0]        
        def func(x,m,b):
            return(m*x+b)
        coeffs,covars = optimize.curve_fit(func,y,vals,guess)
        return coeffs
        
    data = stack.deepcopy()
    y = np.int16(np.sort(getPoints(stack.data)[:,0]))
    print('\nCorrecting tilt axis....')
    tilts = stack.tilts*np.pi/180
    xshift = tiltaxis = 0
    totaltilt = totalshift = 0
    count = 1

    while abs(tiltaxis) >= 1 or abs(xshift) >= 1 or count == 1:
        centers = sinocalc(data.data,y)
    
        coeffs = np.zeros([3,3])
        coeffs[0,:] = fitCoMs(tilts,centers[:,0])
        coeffs[1,:] = fitCoMs(tilts,centers[:,1])
        coeffs[2,:] = fitCoMs(tilts,centers[:,2])
        
        r = np.zeros(3)
        r[:] = coeffs[:,0]
        
        coeffsaxis = np.zeros([1,3])
        coeffsaxis = fitTiltAxis(y,r)
        tiltaxis = 180/np.pi*np.tanh(coeffsaxis[0])
        xshift = (coeffsaxis[1]/coeffsaxis[0]*np.sin(np.pi/180*tiltaxis))
        xshift = (data.data.shape[1]/2)-xshift - offset
        totaltilt += tiltaxis
        totalshift += xshift
        
        print(('Iteration #%s' % count))    
        print(('Calculated tilt correction is: %s' % str(tiltaxis)))
        print(('Calculated shift value is: %s' % str(xshift)))
        count += 1

        data = data.transStack(xshift=0,yshift=xshift,angle=tiltaxis)
        
    out = copy.deepcopy(data)
    out.data = np.transpose(data.data,(0,2,1))
    print('\nTilt axis alignment complete')    
    out.tiltaxis = totaltilt
    out.xshift = totalshift
    return(out)

def tiltAnalyze(data,limit=10,delta=0.3):
    """
    Perform automated determination of the tilt axis of a Stack by measuring
    the rotation of the projected maximum image.  Maximum image is rotated postively
    and negatively, filterd using a Hamming window, and the rotation angle is
    determined by iterative histogram analysis
    
    Args
    ----------
    data : Stack object
        3-D numpy array containing the tilt series data
    limit : integer or float
        Maximum rotation angle to use for MaxImage calculation
    delta : float
        Angular increment for MaxImage calculation
        
    Returns
    ----------
    opt_angle : Stack object
        Calculated rotation to set the tilt axis vertical
    """

    def hamming(image):
        """
        Function to apply hamming window to the image to remove edge effects

        Args
        ----------
        image : Numpy array
            Input image
        Returns
        ----------
        out : Numpy array
            Filtered image
        """
        if image.shape[0] < image.shape[1]:
            image = image[:,np.int32((image.shape[1]-image.shape[0])/2):-np.int32(((image.shape[1]-image.shape[0])/2))]
            if image.shape[0] != image.shape[1]:
                image = image[:,0:-1]
            h = np.hamming(image.shape[0])
            ham2d = np.sqrt(np.outer(h,h))
        elif image.shape[1] < image.shape[0]:
            image = image[np.int32((image.shape[0]-image.shape[1])/2):-np.int32(((image.shape[0]-image.shape[1])/2)),:]
            if image.shape[0] != image.shape[1]:
                image = image[0:-1,:]
            h = np.hamming(image.shape[1])
            ham2d = np.sqrt(np.outer(h,h))
        else:
            h = np.hamming(image.shape[0])
            ham2d = np.sqrt(np.outer(h,h))
        out = ham2d*image
        return(out)

    def find_score(im, angle):
        """
        Function to perform histogram analysis to measure the rotation angle

        Args
        ----------
        image : Numpy array
            Input image
        angle : float
            Angle by which to rotate the input image before analysis
            
        Returns
        ----------
        hist : Numpy array
            Result of integrating image along the vertical axis
        score : numpy array
            Score calculated from hist
        """
        im = ndimage.rotate(im,angle,reshape=False,order=3)
        hist = np.sum(im, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return(hist, score)
    
    image = np.max(data.data,0)
    rot_pos = ndimage.rotate(hamming(image),limit/2,reshape=False,order=3)
    rot_neg = ndimage.rotate(hamming(image),-limit/2,reshape=False,order=3)
    angles = np.arange(-limit, limit+delta, delta)
    scores_pos = []
    scores_neg = []
    for angle in tqdm.tqdm(angles):
        hist_pos, score_pos = find_score(rot_pos, angle)
        hist_neg, score_neg = find_score(rot_neg, angle)
        scores_pos.append(score_pos)
        scores_neg.append(score_neg)

    best_score_pos = max(scores_pos)
    best_score_neg = max(scores_neg)
    pos_angle = -angles[scores_pos.index(best_score_pos)]
    neg_angle = -angles[scores_neg.index(best_score_neg)]
    opt_angle = (pos_angle+neg_angle)/2
    print('Optimum positive rotation angle: {}'.format(pos_angle))
    print('Optimum negative rotation angle: {}'.format(neg_angle))
    print('Optimum positive rotation angle: {}'.format(opt_angle))
    return(opt_angle)
    
def alignToOther(stack,other):
    """
    Function to spatially register a Stack using a seres of shifts previously calculated
    on a separate data stack of the same size.

    Args
    ----------
    stack : Stack object
        Stack which was previously aligned
    other : Stack object
        Stack to be aligned
        
    Returns
    ----------
    out : Stack object
        Aligned copy of other Stack
    """
    out = copy.deepcopy(other)
    out.data = np.zeros(np.shape(other.data),dtype=other.data.dtype)    
    out.shifts = stack.shifts
    out.tiltaxis = stack.tiltaxis
    out.xshift = stack.xshift
    if stack.shifts:   
        for i in range(0,out.data.shape[0]):
            out.data[i,:,:] = cv2.warpAffine(other.data[i,:,:],stack.shifts[i],other.data[i,:,:].T.shape,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0.0)
    out = out.transStack(xshift=other.xshift,yshift=0.0,angle=other.tiltaxis)
    print('Stack alignment applied')
    return(out)
