# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:38:16 2016

@author: aherzing
"""

import numpy as np
from tomotools import recon, align
import copy
import os
import cv2
import pylab as plt
import matplotlib.animation as animation
from hyperspy.signals import Signal2D
from scipy import ndimage
from tempfile import TemporaryDirectory


class TomoStack(Signal2D):
    """
    Create a TomoStack object for tomography data.

    Note: All attributes are initialized with values of None or 0.0 in
    __init__.

    #TODO create __init__ function
    # Attributes
    # ----------
    # stack : Hyperspy Signal2D
    #     The tilt series or reconstruction data array. It can be either 2 or 3
    #     dimensional. For a tilt series, the first dimension must be the tilt
    #     increment axis (e.g. [theta,X] or [theta,X,Y]). Prior to
    #     reconstruction, the third dimension must describe the tilt axis
    #     orientation. For a reconstruction, the dimensions will be [Y,X,Z].
    # shifts : numpy array
    #     X,Y shifts calculated for each image for stack registration
    # tiltaxis : float
    #      Angular orientation (in degrees) by which data is rotated to
           orient the
    #      stack so that the tilt axis is vertical
    # xshift : float
    #     Lateral shift of the tilt axis from the center of the stack.
    """

    def __repr__(self):
        string = '<TomoStack, '

        if 'title' in self.metadata['General'].keys():
            string += "title: %s" % self.metadata['General']['title']
        else:
            string += "title: "
        if self.data is None:
            string += ", Empty"
        elif len(self.data.shape) == 2:
            string += ", dimensions: (|%s, %s)" % (str(self.data.shape[1]),
                                                   str(self.data.shape[0]))
        elif len(self.data.shape) == 3:
            string += ", dimensions: (%s|%s, %s)" % (str(self.data.shape[0]),
                                                     str(self.data.shape[2]),
                                                     str(self.data.shape[1]))
        string += '>'
        return string

    def align_other(self, other):
        """
        Method to apply the alignment calculated for one dataset to another.
        This will include the spatial registration, tilt axis, and tilt axis
        shift if they have been previously calculated.

        Args
        ----------
        other : TomoStack object
            The tilt series which is to be aligned using the previously
            calculated parameters. The data array in the TomoStack must be of
            the same size as that in self.data

        Returns
        ----------
        out : TomoStack object
            The result of applying the alignment to other
        """

        if self.original_metadata.shifts is None:
            raise ValueError('Spatial registration has not been calculated '
                             'for this stack')

        out = align.align_to_other(self, other)
        return out

    def invert(self):
        """
        Method to invert the contrast levels of an entire TomoStack

        Args
        ----------

        Returns
        ----------
        inverted : TomoStack object
            Copy of the input stack with contrast inverted

        Examples
        --------
        >>> import tomotools
        >>> s = tomotools.load('tomotools/tests/test_data/HAADF.mrc')
        Tilts found in metadata
        >>> s_inverted = s.invert()
        """
        maxvals = self.data.max(2).max(1)
        maxvals = maxvals.reshape([self.data.shape[0], 1, 1])
        minvals = self.data.min(2).min(1)
        minvals = minvals.reshape([self.data.shape[0], 1, 1])
        ranges = maxvals-minvals

        inverted = self.deepcopy()
        inverted.data = inverted.data - np.reshape(inverted.data.mean(
            2).mean(1), [self.data.shape[0], 1, 1])
        inverted.data = (inverted.data - minvals) / ranges

        inverted.data = inverted.data - 1
        inverted.data = np.sqrt(inverted.data ** 2)

        inverted.data = (inverted.data * ranges) + minvals

        return inverted

    def stack_register(self, method='ECC', start=None, show_progressbar=False):
        """
        Method which calls a function in the align module to spatially
        register a stack using one of two OpenCV based algorithms: Phase
        Correlation (PC) or Enhanced Correlation Coefficient (ECC)
        maximization.

        Args
        ----------
        method : string
            Algorithm to use for registration calculation. Must be either
            'PC' or 'ECC'
        start : integer
            Position in tilt series to use as starting point for the
            alignment. If None, the central projection is used.
        show_progressbar : boolean
            Enable/disable progress bar

        Returns
        ----------
        out : TomoStack object
            Spatially registered copy of the input stack

        Examples
        --------
        Registration with enhanced correlation coefficient algorithm (ECC)
        >>> import tomotools
        >>> s = tomotools.load('tomotools/tests/test_data/HAADF.mrc')
        Tilts found in metadata
        >>> s.inav[0:10].stack_register('ECC',show_progressbar=False)
        Spatial registration by ECC complete
        <TomoStack, title: , dimensions: (10|256, 256)>

        Registration with phase correlation algorithm (PC)
        >>> import tomotools
        >>> s = tomotools.load('tomotools/tests/test_data/HAADF.mrc')
        Tilts found in metadata
        >>> s.inav[0:10].stack_register('PC',show_progressbar=False)
        Spatial registration by PC complete
        <TomoStack, title: , dimensions: (10|256, 256)>
        """

        if method == 'ECC':
            out = align.rigid_ecc(self, start, show_progressbar)
        elif method == 'PC':
            out = align.rigid_pc(self, start, show_progressbar)
        else:
            print("Unknown registration method.  Must use 'ECC' or 'PC'")
            return ()
        return out

    def tilt_align(self, method, limit=10, delta=0.3, offset=0.0, locs=None,
                   output=True, show_progressbar=False):
        """
        Method to call one of two tilt axis calculation functions in the
        align module ('CoM' and 'MaxImage')
        and apply the calculated rotation.

        Available options are 'CoM' and 'Error'

        CoM: track the center of mass (CoM) of the projections at three
        locations.  Fit the motion of the CoM as a function of tilt to that
        expected for an ideal cylinder to calculate an X-shift at each
        location. Perform a  linear fit of the three X-shifts to calculate an
        ideal rotation.

        MaxImage: Perform automated determination of the tilt axis of a
        TomoStack by measuring the rotation of the projected maximum image.
        Maximum image is rotated positively and negatively, filtered using a
        Hamming window, and the rotation angle is determined by iterative
        histogram analysis

        Args
        ----------
        method : string
            Algorithm to use for registration alignment. Must be either 'CoM'
            or 'MaxImage'
        limit : integer
            Position in tilt series to use as starting point for the
            alignment. If None, the central projection is used.
        delta : integer
            Position i
        offset : integer
            Not currently used
        limit : integer or float
            Maximum rotation angle to use for MaxImage calculation
        delta : float
            Angular increment for MaxImage calculation
        locs : list
            Image coordinates indicating the locations at which to calculate
            the alignment
        output : boolean
            Output alignment results to console after each iteration
        show_progressbar : boolean
            Enable/disable progress bar

        Returns
        ----------
        out : TomoStack object
            Copy of the input stack rotated by calculated angle

        Examples
        ----------
        Align tilt axis using the center of mass (CoM) method
        >>> import tomotools
        >>> s = tomotools.load('tomotools/tests/test_data/HAADF.mrc')
        Tilts found in metadata
        >>> reg = s.stack_register('ECC',show_progressbar=False)
        Spatial registration by ECC complete
        >>> ali = reg.tilt_align(method='CoM', locs=[50,100,160], output=False)

        Align tilt axis using the maximum image method
        >>> import tomotools
        >>> s = tomotools.load('tomotools/tests/test_data/HAADF.mrc')
        Tilts found in metadata
        >>> reg = s.stack_register('ECC',show_progressbar=False)
        Spatial registration by ECC complete
        >>> ali = reg.tilt_align(method='MaxImage', output=False,\
        show_progressbar=False)

        """
        if method == 'CoM':
            out = align.tilt_correct(self, offset, locs, output)
        elif method == 'MaxImage':
            angle = align.tilt_analyze(self, limit, delta, output,
                                       show_progressbar)
            if angle > 0.1:
                out = self.rotate(angle, True)
            else:
                out = self.deepcopy()
            out.tiltaxis = angle
        else:
            print('Invalid alignment method: Enter either "CoM" or "MaxImage"')
            return
        return out

    def reconstruct(self, method='FBP', rot_center=None, iterations=None,
                    constrain=False, thresh=0, cuda=None):
        """
        Function to reconstruct a stack using one of the available methods:
        astraWBP, astraSIRT, astraSIRT_GPU

        Args
        ----------
        method : string
            Reconstruction algorithm to use.  Must be either 'FBP' (default)
            or 'SIRT'
        rot_center : float
            Location of the rotation center.  If None, position is assumed to
            be the center of the image.
        iterations : integer
            Number of iterations for the SIRT reconstruction (for astraSIRT
            and astraSIRT_GPU, methods only)
        constrain : boolean
            If True, output reconstruction is constrained above value given
            by 'thresh'
        thresh : integer or float
            Value above which to constrain the reconstructed data
        cuda : boolean
            If True, use the CUDA-accelerated reconstruction algorithm

        Returns
        ----------
        out : TomoStack object
            TomoStack containing the reconstructed volume

        Examples
        ----------
        Filtered backprojection (FBP) reconstruction
        >>> import tomotools
        >>> file = 'tomotools/tests/test_data/HAADF_Aligned.hdf5'
        >>> stack = tomotools.load(file)
        Tilts found in metadata
        >>> slices = stack.isig[:, 120:121].deepcopy()
        >>> rec = slices.reconstruct('FBP')
        Reconstruction complete

        Simultaneous iterative reconstruction technique (SIRT) reconstruction
        >>> import tomotools
        >>> file = 'tomotools/tests/test_data/HAADF_Aligned.hdf5'
        >>> stack = tomotools.load(file)
        Tilts found in metadata
        >>> slices = stack.isig[:, 120:121].deepcopy()
        >>> rec = slices.reconstruct('SIRT',iterations=5)
        Reconstruction complete

        Simultaneous iterative reconstruction technique (SIRT) reconstruction
        with positivity constraint
        >>> import tomotools
        >>> file = 'tomotools/tests/test_data/HAADF_Aligned.hdf5'
        >>> stack = tomotools.load(file)
        Tilts found in metadata
        >>> slices = stack.isig[:, 120:121].deepcopy()
        >>> rec = slices.reconstruct('SIRT',iterations=5, constrain=True, \
        thresh=0)
        Reconstruction complete

        """
        if cuda is None:
            if 'CUDA_Path' in os.environ.keys():
                cuda = True
            else:
                cuda = False

        out = copy.deepcopy(self)
        out.data = recon.run(self, method, rot_center, iterations, constrain,
                             thresh, cuda)

        out.axes_manager[0].name = 'y'
        out.axes_manager[0].size = out.data.shape[0]
        out.axes_manager[0].offset = self.axes_manager['y'].offset
        out.axes_manager[0].scale = self.axes_manager['y'].scale
        out.axes_manager[0].units = self.axes_manager['y'].units

        out.axes_manager[2].name = 'z'
        out.axes_manager[2].size = out.data.shape[1]
        out.axes_manager[2].offset = self.axes_manager['x'].offset
        out.axes_manager[2].scale = self.axes_manager['x'].scale
        out.axes_manager[2].units = self.axes_manager['x'].units

        out.axes_manager[1].name = 'x'
        out.axes_manager[1].size = out.data.shape[2]
        out.axes_manager[1].offset = self.axes_manager['x'].offset
        out.axes_manager[1].scale = self.axes_manager['x'].scale
        out.axes_manager[1].units = self.axes_manager['x'].units
        return out

    def rotate(self, angle, resize=True):
        """
        Method to rotate the stack by a given angle using the SciPy
        ndimage.rotate function

        Args
        ----------
        angle : float
            Angle by which to rotate the data in the TomoStack about the XY
            plane
        resize : boolean
            If True, output stack size is increased relative to input so that
            no pixels are lost.
            If False, output stack is the same size as the input.

        Returns
        ----------
        rot : TomoStack object
            Rotated copy of the input stack

        Examples
        ----------
        >>> import tomotools
        >>> stack = tomotools.load('tomotools/tests/test_data/HAADF.mrc')
        Tilts found in metadata
        >>> stack.isig[100:156,:]
        <TomoStack, title: , dimensions: (77|56, 256)>
        >>> rotated = stack.isig[100:156,:].rotate(90)
        >>> rotated
        <TomoStack, title: , dimensions: (77|256, 56)>
        """

        rot = self.deepcopy()
        rot.data = ndimage.rotate(rot.data, angle, axes=(1, 2), reshape=resize)

        rot.axes_manager[1].size = rot.data.shape[2]
        rot.axes_manager[2].size = rot.data.shape[1]
        return rot

    def test_align(self, xshift=0.0, angle=0.0, slices=None):
        """
        Method to produce quickly reconstruct three slices from the input
        data for inspection of the quality of the alignment.

        Args
        ----------
        xshift : float
            Number of pixels by which to shift the input data.
        angle : float
            Angle by which to rotate stack prior to reconstruction
        slices : list
            Position of slices to use for the reconstruction.  If None,
            positions at 1/4, 1/2, and 3/4 of the full size of the stack are
            chosen.
        """
        if slices is None:
            mid = np.int32(self.data.shape[1] / 2)
            slices = np.int32([mid / 2, mid, mid + mid / 2])

        temp = self.deepcopy()
        if angle != 0:
            shifted = temp.trans_stack(xshift, 0, angle)
        elif angle == 0 and xshift != 0:
            shifted = self.deepcopy()
            shifted.data = shifted.data[:, slices, :]
            shifted = shifted.trans_stack(xshift, 0, 0)
        else:
            shifted = self.deepcopy()
            shifted.data = shifted.data[:, slices, :]

        rec = recon.run(shifted, method='FBP', cuda=False)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
        ax1.imshow(rec[0, :, :], cmap='afmhot')
        ax1.set_title('Slice %s' % str(slices[0]))
        ax1.set_axis_off()

        ax2.imshow(rec[1, :, :], cmap='afmhot')
        ax2.set_title('Slice %s' % str(slices[1]))
        ax2.set_axis_off()

        ax3.imshow(rec[2, :, :], cmap='afmhot')
        ax3.set_title('Slice %s' % str(slices[2]))
        ax3.set_axis_off()

        return

    def trans_stack(self, xshift=0.0, yshift=0.0, angle=0.0):
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
        out : TomoStack object
            Transformed copy of the input stack

        Examples
        ----------
        >>> import tomotools
        >>> stack = tomotools.load('tomotools/tests/test_data/HAADF.mrc')
        Tilts found in metadata
        >>> transformed = stack.trans_stack(xshift=10.0, yshift=3.5, \
        angle=-15.2)
        >>> transformed
        <TomoStack, title: , dimensions: (77|256, 256)>

        """
        out = self.deepcopy()
        if angle:
            image_center = tuple(np.array(out.data[0, :, :].shape) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale=1.0)
            for i in range(0, out.data.shape[0]):
                out.data[i, :, :] = \
                    cv2.warpAffine(out.data[i, :, :],
                                   rot_mat,
                                   out.data[i, :, :].T.shape,
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0.0)
        if xshift != 0.0 or yshift != 0.0:
            trans_mat = np.array([[1., 0, xshift], [0, 1., yshift]])
            for i in range(0, out.data.shape[0]):
                out.data[i, :, :] = \
                    cv2.warpAffine(out.data[i, :, :],
                                   trans_mat,
                                   out.data[i, :, :].T.shape,
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0.0)
        return out

    # noinspection PyTypeChecker
    def savemovie(self, start, stop, axis='XY', fps=15, dpi=100,
                  outfile=None, title='output.avi', clim=None, cmap='afmhot'):
        """
        Method to save the TomoStack as an AVI movie file.

        Args
        ----------
        start : integer
         Filename for output. If None, a UI will prompt for a filename.
        stop : integer
         Filename for output. If None, a UI will prompt for a filename.
        axis : string
         Projection axis for the output movie.
         Must be 'XY' (default), 'YZ' , or 'XZ'
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
            clim = [self.data.min(), self.data.max()]

        fig, ax = plt.subplots(1, figsize=(8, 8))

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if title:
            ax.set_title(title)

        if axis == 'XY':
            im = ax.imshow(self.data[:, start, :], interpolation='none',
                           cmap=cmap, clim=clim)
        elif axis == 'XZ':
            im = ax.imshow(self.data[start, :, :], interpolation='none',
                           cmap=cmap, clim=clim)
        elif axis == 'YZ':
            im = ax.imshow(self.data[:, :, start], interpolation='none',
                           cmap=cmap, clim=clim)
        else:
            raise ValueError('Unknown axis!')
        fig.tight_layout()

        def updatexy(n):
            tmp = self.data[:, n, :]
            im.set_data(tmp)
            return im

        def updatexz(n):
            tmp = self.data[n, :, :]
            im.set_data(tmp)
            return im

        def updateyz(n):
            tmp = self.data[:, :, n]
            im.set_data(tmp)
            return im

        frames = np.arange(start, stop, 1)

        if axis == 'XY':
            ani = animation.FuncAnimation(fig, updatexy, frames)
        elif axis == 'XZ':
            ani = animation.FuncAnimation(fig, updatexz, frames)
        elif axis == 'YZ':
            ani = animation.FuncAnimation(fig, updateyz, frames)
        else:
            raise ValueError('Axis not understood!')

        writer = animation.writers['ffmpeg'](fps=fps)
        ani.save(outfile, writer=writer, dpi=dpi)
        plt.close()
        return

    def show(self):
        """
        Method to show the TomoStack for visualization with an interactive
        slice slider using OpenCV
        """

        def nothing(x):
            pass

        def simpletrackbar(image, windowname):
            trackbarname = 'Slice'
            if (np.shape(image)[1] > 1024) or (np.shape(image)[2] > 1024):
                new = np.zeros([np.shape(image)[0], 1024, 1024], image.dtype)
                for i in range(0, np.size(image, 0)):
                    new[i, :, :] = cv2.resize(image[i, :, :], (1024, 1024))
                image = new
            cv2.startWindowThread()
            cv2.namedWindow(windowname)
            cv2.createTrackbar(trackbarname, windowname, 0,
                               np.size(image, 0) - 1, nothing)

            while True:
                trackbarpos = cv2.getTrackbarPos(trackbarname, windowname)
                cv2.imshow(windowname,
                           image[trackbarpos, :, :]
                           / np.max(image[trackbarpos, :, :]))
                ch = cv2.waitKey(5)
                if ch == 27:
                    break
            cv2.destroyAllWindows()

        simpletrackbar(self.data, 'Press "ESC" to exit')
        return

    def align_imod(self, diameter=7, markers=10):
        # TODO Automatic dection of IMOD presence and path
        # if 'IMOD' in os.environ["PATH"]:
        #     imod_path = [s for s in os.environ["PATH"].split(';') if "IMOD"
        #                  in s][0]
        #     imod_path = imod_path.replace("\\", "/")
        #     print('IMOD found in %s' % imod_path)
        # else:
        #     print('IMOD does not appear to be installed. Cannot run RAPTOR')
        #     return

        imod_path = 'c:/progra~1/imod/bin/'

        ali = self.deepcopy()
        shape = self.data.shape
        orig_path = os.getcwd()
        tmp_dir = TemporaryDirectory()
        os.chdir(tmp_dir.name)

        with open('stack.raw', 'wb') as h:
            self.data.tofile(h)

        mrc_cmd = 'raw2mrc -x %s -y %s -z %s -t float ' % \
                  (str(shape[2]), str(shape[1]), str(shape[0])) + \
                  'stack.raw stack.mrc'

        os.system(mrc_cmd)
        angles = self.axes_manager[0].axis
        with open('stack.rawtlt', 'w') as h:
            np.savetxt(h, angles, fmt='%.1f')

        raptor_cmd = 'raptor -exec %s ' % imod_path + \
                     '-path . -inp stack.mrc -out raptor ' \
                     '-diam %s -mark %s stack.mrc' % \
                     (str(diameter), str(markers))
        os.system(raptor_cmd)

        file = 'raptor/align/stack.ali'
        with open(file, 'rb') as h:
            _ = np.fromfile(h, np.uint8, 1024)
            temp = np.fromfile(h, np.float32)

        ali.data = temp.reshape([shape[0], shape[1], shape[2]])

        os.chdir(orig_path)
        return ali
