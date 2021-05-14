# -*- coding: utf-8 -*-
#
# This file is part of TomoTools

"""
Primary module for TomoTools package.

Contains the TomoStack class and its methods.

@author: Andrew Herzing
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
import matplotlib as mpl
import logging
import astra

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TomoStack(Signal2D):
    """
    Create a TomoStack object for tomography data.

    Note: All attributes are initialized with values of None or 0.0
    in __init__ unless they are already defined

    # Attributes
    # ----------
    # shifts : numpy array
    #     X,Y shifts calculated for each image for stack registration
    # tiltaxis : float
    #      Angular orientation (in degrees) by which data is rotated to
           orient the
    #      stack so that the tilt axis is vertical
    # xshift : float
    #     Lateral shift of the tilt axis from the center of the stack.
    """

    def __init__(self, *args, **kwargs):
        """Initialize TomoStack class."""
        super().__init__(*args, **kwargs)

        if not self.metadata.has_item("Tomography"):
            self.metadata.add_node("Tomography")
            self.metadata.Tomography.set_item("tilts", None)
            self.metadata.Tomography.set_item("tiltaxis", 0)
            self.metadata.Tomography.set_item("xshift", 0)
            self.metadata.Tomography.set_item("yshift", 0)
            self.metadata.Tomography.set_item("shifts", None)
            self.metadata.Tomography.set_item("cropped", False)
        else:
            if not self.metadata.Tomography.has_item("tilts"):
                self.metadata.Tomography.set_item("tilts", None)
            if not self.metadata.Tomography.has_item("tiltaxis"):
                self.metadata.Tomography.set_item("tiltaxis", 0)
            if not self.metadata.Tomography.has_item("xshift"):
                self.metadata.Tomography.set_item("xshift", 0)
            if not self.metadata.Tomography.has_item("yshift"):
                self.metadata.Tomography.set_item("yshift", 0)
            if not self.metadata.Tomography.has_item("shifts"):
                self.metadata.Tomography.set_item("shifts", None)
            if not self.metadata.Tomography.has_item("cropped"):
                self.metadata.Tomography.set_item("cropped", False)

    def test_correlation(self, images=None):
        """
        Test output of cross-correlation prior to alignment.

        Args
        ----------
        images : list
            List of two numbers indicating which projections to cross-correlate

        Returns
        ----------
        fig : Matplotlib Figure
            Figure showing the results

        """
        if not images:
            images = [0, 1]
        im1 = self.data[images[0], :, :]
        im2 = self.data[images[1], :, :]

        image_product = np.fft.fft2(im1) * np.fft.fft2(im2).conj()
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))

        fig = plt.figure(figsize=(8, 3))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
        ax3 = plt.subplot(1, 3, 3)

        ax1.imshow(im1, cmap='gray')
        ax1.set_axis_off()
        ax1.set_title('Reference image')

        ax2.imshow(im2, cmap='gray')
        ax2.set_axis_off()
        ax2.set_title('Offset image')

        ax3.imshow(cc_image.real, cmap='inferno')
        ax3.set_axis_off()
        ax3.set_title("Cross-correlation")
        return fig

    def align_other(self, other):
        """
        Apply the alignment calculated for one dataset to another.

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
        # Check if any transformations have been applied to the current stack
        if (self.metadata.Tomography.shifts is None) and\
           any([self.metadata.Tomography.xshift is None,
                self.metadata.Tomography.xshift == 0.0]) and\
           any([self.metadata.Tomography.yshift is None,
                self.metadata.Tomography.yshift == 0.0]) and\
           any([self.metadata.Tomography.tiltaxis is None,
                self.metadata.Tomography.tiltaxis == 0.0]):
            raise ValueError('No transformation have been applied '
                             'to this stack')

        out = align.align_to_other(self, other)

        return out

    def filter(self, method='median', size=5, taper=0.1):
        """
        Apply one of several image filters to an entire TomoStack.

        Args
        ----------
        method : string
            Type of filter to apply. Must be 'median' or 'sobel'.
        size : integer
            Size of filtering neighborhood.
        taper : float
            Fraction of image size to pad to the mean.

        Returns
        ----------
        filtered : TomoStack object
            Filtered copy of the input stack

        Examples
        --------
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> filtered = stack.filter(method='median')

        """
        filtered = self.deepcopy()
        if method == 'median':
            filtered.data = ndimage.median_filter(filtered.data,
                                                  size=(1, size, size))
        elif method == 'sobel':
            for i in range(0, filtered.data.shape[0]):
                dx = ndimage.sobel(filtered.data[i, :, :], 0)
                dy = ndimage.sobel(filtered.data[i, :, :], 1)
                filtered.data[i, :, :] = np.hypot(dx, dy)
        elif method == 'both':
            filtered.data = ndimage.median_filter(filtered.data,
                                                  size=(1, size, size))
            for i in range(0, filtered.data.shape[0]):
                dx = ndimage.sobel(filtered.data[i, :, :], 0)
                dy = ndimage.sobel(filtered.data[i, :, :], 1)
                filtered.data[i, :, :] = np.hypot(dx, dy)
        elif method == 'bpf':
            lp_freq = 0.1
            hp_freq = 0.05
            lp_sigma = 1.5
            hp_sigma = 1.5
            [nprojs, rows, cols] = self.data.shape

            F = np.fft.fftshift(np.fft.fft2(self.data))

            x = (np.arange(0, cols) - np.fix(cols / 2)) / cols
            y = (np.arange(0, rows) - np.fix(rows / 2)) / rows
            xx, yy = np.meshgrid(x, y)
            r = np.sqrt(xx**2 + yy**2)
            lpf = 1 / (1.0 + (r / lp_freq)**(2 * lp_sigma))

            hpf = 1 - (1 / (1.0 + (r / hp_freq)**(2 * hp_sigma)))
            bpf = lpf * hpf
            F_filtered = F * bpf

            filtered.data = np.fft.ifft2(np.fft.ifftshift(F_filtered)).real

            h = np.hamming(rows)
            ham2d = np.sqrt(np.outer(h, h))
            filtered.data = filtered.data * ham2d
        elif method is None:
            pass
        else:
            raise ValueError("Unknown filter method. Must be 'median', "
                             "'sobel', 'both', 'bpf', or None")
        if taper:
            taper_size = np.int32(np.array(taper) * self.data.shape[1:])
            filtered.data = np.pad(filtered.data,
                                   [(0, 0),
                                    (taper_size[0], taper_size[0]),
                                    (taper_size[1], taper_size[1])],
                                   mode='constant')
        return filtered

    def normalize(self, width=3):
        """
        Normalize the contrast levels of an entire TomoStack.

        Args
        ----------
        width : integer
            Number of standard deviations from the mean to set
            as maximum intensity level.

        Returns
        ----------
        normalized : TomoStack object
            Copy of the input stack with intensities normalized

        Examples
        --------
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> s_normalized = stack.normalize()

        """
        normalized = self.deepcopy()
        minvals = np.reshape((normalized.data.min(2).min(1)),
                             [self.data.shape[0], 1, 1])
        normalized.data = normalized.data - minvals
        meanvals = np.reshape((normalized.data.mean(2).mean(1)),
                              [self.data.shape[0], 1, 1])
        stdvals = np.reshape((normalized.data.std(2).std(1)),
                             [self.data.shape[0], 1, 1])
        normalized.data = normalized.data / (meanvals + width * stdvals)
        return normalized

    def invert(self):
        """
        Invert the contrast levels of an entire TomoStack.

        Returns
        ----------
        inverted : TomoStack object
            Copy of the input stack with contrast inverted

        Examples
        --------
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> s_inverted = stack.invert()

        """
        maxvals = self.data.max(2).max(1)
        maxvals = maxvals.reshape([self.data.shape[0], 1, 1])
        minvals = self.data.min(2).min(1)
        minvals = minvals.reshape([self.data.shape[0], 1, 1])
        ranges = maxvals - minvals

        inverted = self.deepcopy()
        inverted.data = inverted.data - np.reshape(inverted.data.mean(
            2).mean(1), [self.data.shape[0], 1, 1])
        inverted.data = (inverted.data - minvals) / ranges

        inverted.data = inverted.data - 1
        inverted.data = np.sqrt(inverted.data ** 2)

        inverted.data = (inverted.data * ranges) + minvals

        return inverted

    def stats(self):
        """Print basic stats about TomoStack data to terminal."""
        print('Mean: %.1f' % self.data.mean())
        print('Std: %.2f' % self.data.std())
        print('Max: %.1f' % self.data.max())
        print('Min: %.1f\n' % self.data.min())
        return

    def stack_register(self, method='ECC', start=None, crop=False,
                       show_progressbar=False, nslice=None, ratio=0.5,
                       cl_resolution=0.05, cl_div_factor=8, com_ref_index=None,
                       cl_ref_index=None):
        """
        Register stack spatially.

        Options are phase correlation (PC) maximization, enhanced
        correlation coefficient (ECC) maximization, StackReg, center of
        mass ('COM'), or combined center of mass and common line methods.
        See docstring for tomotools.align.align_stack for details.

        Args
        ----------
        method : string
            Algorithm to use for registration calculation. Must be either
            'PC', 'ECC', 'StackReg', 'COM', or 'COM-CL'.
        start : integer
            Position in tilt series to use as starting point for the
            alignment. If None, the central projection is used.
        crop : boolean
            If True, crop aligned stack to eliminate border pixels. Default is
            False.
        show_progressbar : boolean
            Enable/disable progress bar
        nslice : int
            Location of slice to use for alignment.  Only used for 'COM' method
        ratio : float
            Value between 0 and 1 used to assess quality of projections.
            Only used for 'COM' method.
        com_ref_index : integer
            Reference slice for center of mass alignment.  All other slices
            will be aligned to this reference.  If not provided, the midpoint
            of the stack will be chosen.
        cl_ref_index : integer
            Reference slice for common line alignment.  All other slices
            will be aligned to this reference.  If not provided, the midpoint
            of the stack will be chosen.
        cl_resolution : float
            Resolution for subpixel common line alignment. Default is 0.05.
            Should be less than 0.5.
        cl_div_factor : integer
            Factor which determines the number of iterations of common line
            alignment to perform.  Default is 8.

        Returns
        ----------
        out : TomoStack object
            Spatially registered copy of the input stack

        Examples
        --------
        Registration with enhanced correlation coefficient algorithm (ECC)
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> regECC = stack.inav[0:10].stack_register('ECC')

        Registration with phase correlation algorithm (PC)
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> regPC = stack.inav[0:10].stack_register('PC')

        Registration with center of mass tracking (COM)
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> regCOM = stack.stack_register('COM')

        Registration with StackReg
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> regSR = stack.inav[0:10].stack_register('StackReg')

        Registration with center of mass and common line (COM-CL)
        >>> import tomotools.api as tomotools
        >>> s = tomotools.load('tomotools/tests/test_data/HAADF.mrc')
        >>> regCOMCL = s.inav[0:10].stack_register('COM-CL')

        """
        method = method.lower()
        if method in ['ecc', 'pc', 'com', 'stackreg', 'com-cl']:
            out = align.align_stack(self, method, start, show_progressbar,
                                    ratio=ratio, nslice=nslice,
                                    cl_resolution=cl_resolution,
                                    cl_div_factor=cl_div_factor,
                                    com_ref_index=com_ref_index,
                                    cl_ref_index=cl_ref_index)
        else:
            raise ValueError(
                "Unknown registration method: "
                "%s. Must be ECC, PC, StackReg, or COM" % method)

        if crop:
            out = align.shift_crop(out)
        return out

    def tilt_align(self, method, limit=10, delta=0.3, locs=None,
                   axis=0, show_progressbar=False):
        """
        Align the tilt axis of a TomoStack.

        Uses either a center-of-mass approach or a maximum image approach

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

        minimize: Perform automated determination of the tilt axis of a
        TomoStack by minimizing the difference between the reconstruction and
        the input dataset usign scipy.optimize.differential_evolution.

        Args
        ----------
        method : string
            Algorithm to use for registration alignment. Must be either 'CoM',
            'MaxImage', or 'minimize'.
        limit : integer
            Position in tilt series to use as starting point for the
            alignment. If None, the central projection is used.
        delta : integer
            Position i
        limit : integer or float
            Maximum rotation angle to use for MaxImage calculation
        delta : float
            Angular increment for MaxImage calculation
        locs : list
            Image coordinates indicating the locations at which to calculate
            the alignment
        axis : integer
            Axis along which to extract sinograms. Value of 0 means tilt axis
            is horizontally oriented.  1 means vertically oriented.
        show_progressbar : boolean
            Enable/disable progress bar

        Returns
        ----------
        out : TomoStack object
            Copy of the input stack rotated by calculated angle

        Examples
        ----------
        Align tilt axis using the center of mass (CoM) method
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> reg = stack.stack_register('ECC',show_progressbar=False)
        >>> method = 'CoM'
        >>> ali = reg.tilt_align(method, locs=[50,100,160])

        Align tilt axis using the maximum image method
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> reg = stack.stack_register('ECC',show_progressbar=False)
        >>> method = 'MaxImage'
        >>> ali = reg.tilt_align(method, show_progressbar=False)

        """
        method = method.lower()
        if axis == 1:
            self = self.swap_axes(1, 2)
        if method == 'com':
            out = align.tilt_com(self, locs)
        elif method == 'maximage':
            out = align.tilt_maximage(self, limit, delta, show_progressbar)
        elif method == 'minimize':
            out = align.tilt_minimize(self, cuda=astra.use_cuda())
        else:
            raise ValueError(
                "Invalid alignment method: %s."
                "Must be 'CoM', 'MaxImage', or 'Minimize'" % method)

        if axis == 1:
            self = self.swap_axes(2, 1)
        return out

    def reconstruct(self, method='FBP', iterations=None, constrain=False,
                    thresh=0, cuda=None, thickness=None):
        """
        Reconstruct a TomoStack series using one of the available methods.

        astraWBP, astraSIRT, astraSIRT_GPU

        Args
        ----------
        method : string
            Reconstruction algorithm to use.  Must be either 'FBP' (default)
            or 'SIRT'
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
        thickness : integer
            Size of the output volume (in pixels) in the projection direction.

        Returns
        ----------
        out : TomoStack object
            TomoStack containing the reconstructed volume

        Examples
        ----------
        Filtered backprojection (FBP) reconstruction
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data(True)
        >>> slices = stack.isig[:, 120:121].deepcopy()
        >>> rec = slices.reconstruct('FBP')

        Simultaneous iterative reconstruction technique (SIRT) reconstruction
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data(True)
        >>> slices = stack.isig[:, 120:121].deepcopy()
        >>> rec = slices.reconstruct('SIRT',iterations=5)

        Simultaneous iterative reconstruction technique (SIRT) reconstruction
        with positivity constraint
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data(True)
        >>> slices = stack.isig[:, 120:121].deepcopy()
        >>> iterations = 5
        >>> constrain = True
        >>> thresh = 0
        >>> rec = slices.reconstruct('SIRT',iterations, constrain, thresh)

        """
        if cuda is None:
            if 'CUDA_Path' in os.environ.keys():
                cuda = True
            else:
                cuda = False

        out = copy.deepcopy(self)
        out.data = recon.run(self, method, iterations, constrain, thresh, cuda)

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

        if thickness:
            offset = np.int32(np.floor((out.data.shape[1] - thickness) / 2))
            if offset < 0:
                pass
            else:
                out = out.isig[:, offset:-offset]

        return out

    def test_align(self, xshift=0.0, angle=0.0, slices=None, thickness=None,
                   method='FBP', iterations=50, constrain=True):
        """
        Reconstruct three slices from the input data for visual inspection.

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
        thickness : integer
            Size of the output volume (in pixels) in the projection direction.

        """
        if slices is None:
            mid = np.int32(self.data.shape[1] / 2)
            slices = np.int32([mid / 2, mid, mid + mid / 2])

        if (xshift != 0.0) or (angle != 0.0):
            shifted = self.trans_stack(xshift=xshift, yshift=0, angle=angle)
        else:
            shifted = self.deepcopy()
        shifted.data = shifted.data[:, slices, :]

        shifted.axes_manager[0].axis = self.axes_manager[0].axis
        rec = recon.run(shifted, method=method, iterations=iterations,
                        constrain=constrain)

        if thickness:
            offset = np.int32(np.floor((rec.shape[1] - thickness) / 2))
            if offset < 0:
                pass
            else:
                rec = rec[:, offset:-offset, :]

        if mpl.get_backend() == 'nbAgg':
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        if method == 'FBP':
            minvals = rec.mean((1, 2)) - 2 * rec.std((1, 2))
            minvals[minvals < 0] = 0
            maxvals = rec.mean((1, 2)) + 2 * rec.std((1, 2))
        else:
            minvals = np.zeros(3)
            maxvals = rec.max((1, 2))
        for i in range(0, 3):
            if maxvals[i] > rec[i].max():
                maxvals[i] = rec[i].max()
        ax1.imshow(rec[0, :, :], cmap='afmhot', vmin=minvals[0],
                   vmax=maxvals[0])
        ax1.set_title('Slice %s' % str(slices[0]))
        ax1.set_axis_off()

        ax2.imshow(rec[1, :, :], cmap='afmhot', vmin=minvals[1],
                   vmax=maxvals[1])
        ax2.set_title('Slice %s' % str(slices[1]))
        ax2.set_axis_off()

        ax3.imshow(rec[2, :, :], cmap='afmhot', vmin=minvals[2],
                   vmax=maxvals[2])
        ax3.set_title('Slice %s' % str(slices[2]))
        ax3.set_axis_off()
        fig.tight_layout()
        return

    def trans_stack(self, xshift=0.0, yshift=0.0, angle=0.0,
                    interpolation='cubic'):
        """
        Transform the stack using the OpenCV warpAffine function.

        Args
        ----------
        xshift : float
            Number of pixels by which to shift in the X dimension
        yshift : float
            Number of pixels by which to shift the stack in the Y dimension
        angle : float
            Angle in degrees by which to rotate the stack about the X-Y plane
        interpolation : str
            Mode of interpolation to employ. Must be either 'linear',
            'cubic', 'nearest' or 'none'.  Note that 'nearest' and 'none'
            are equivalent.  Default is 'linear'.

        Returns
        ----------
        out : TomoStack object
            Transformed copy of the input stack

        Examples
        ----------
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> xshift = 10.0
        >>> yshift = 3.5
        >>> angle = -15.2
        >>> transformed = stack.trans_stack(xshift, yshift, angle)
        >>> transformed
        <TomoStack, title: , dimensions: (77|256, 256)>

        """
        transformed = self.deepcopy()
        theta = np.pi * angle / 180.
        center_y, center_x = np.float32(np.array(transformed.data.shape[1:])/2)

        rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])

        trans_mat = np.array([[1, 0, center_x],
                              [0, 1, center_y],
                              [0, 0, 1]])

        rev_mat = np.array([[1, 0, -center_x],
                            [0, 1, -center_y],
                            [0, 0, 1]])

        rotation_mat = np.dot(np.dot(trans_mat, rot_mat), rev_mat)

        shift = np.array([[1, 0, np.float32(xshift)],
                          [0, 1, np.float32(-yshift)],
                          [0, 0, 1]])

        full_transform = np.dot(shift, rotation_mat)

        if interpolation == 'linear':
            mode = cv2.INTER_LINEAR
        elif interpolation == 'cubic':
            mode = cv2.INTER_LINEAR
        elif interpolation == 'none' or interpolation == 'nearest':
            mode = cv2.INTER_NEAREST
        else:
            raise ValueError("Interpolation method %s unknown. "
                             "Must be 'linear', 'cubic', 'nearest' "
                             "or 'none'" % interpolation)

        for i in range(0, self.data.shape[0]):
            transformed.data[i, :, :] = \
                cv2.warpAffine(transformed.data[i, :, :],
                               full_transform[:2, :],
                               transformed.data.shape[1:][::-1],
                               flags=mode)

        transformed.metadata.Tomography.xshift =\
            self.metadata.Tomography.xshift + xshift

        transformed.metadata.Tomography.yshift =\
            self.metadata.Tomography.yshift + yshift

        transformed.metadata.Tomography.tiltaxis =\
            self.metadata.Tomography.tiltaxis + angle
        return transformed

    # noinspection PyTypeChecker
    def savemovie(self, start, stop, axis='XY', fps=15, dpi=100,
                  outfile=None, title='output.avi', clim=None, cmap='afmhot'):
        """
        Save the TomoStack as an AVI movie file.

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
        """Display the TomoStack for interactive visualization."""
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

                if image.max() == 1.0:
                    cv2.imshow(windowname,
                               image[trackbarpos, :, :])

                elif image.dtype == '<f4' or 'float32':
                    cv2.imshow(windowname,
                               np.uint8(255 * image[trackbarpos, :, :] /
                                        image[trackbarpos, :, :].max()))

                else:
                    cv2.imshow(windowname,
                               image[trackbarpos, :, :]
                               / np.max(image[trackbarpos, :, :]))
                ch = cv2.waitKey(5)
                if ch == 27:
                    break
            cv2.destroyAllWindows()

        simpletrackbar(self.data, 'Press "ESC" to exit')
        return

    def align_imod(self, diameter=7, markers=10, white=False):
        """
        Align the stack using IMODs RAPTOR algorithm.

        Args
        ----------
        diameter : float
            Diameter in pixels of the fiducial markers

        markers : integer
            Number of markers to include in the model

        white : boolean
            If True, the markers are bright compared to the background (i.e.
            dark-field).

        Returns
        ----------
        ali : TomoStack object
            Aligned copy of the input stack

        """
        # Automatic detection of IMOD presence and path
        imod_path = False
        split_path = os.environ["PATH"].split(';')
        for i in split_path:
            if 'IMOD' in i:
                imod_path = i
        if imod_path:
            logger.info('IMOD found in %s' % imod_path)
        else:
            raise RuntimeError('IMOD does not appear to be '
                               'installed. Cannot run RAPTOR')

        # imod_path = 'c:/progra~1/imod/bin/'
        if self.data.dtype == '<f8':
            self.data = np.float32(self.data)

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

        np.savetxt('stack.rawtlt', angles, fmt='%.1f')

        if white:
            raptor_cmd = 'raptor -exec %s ' % imod_path + \
                '-path . -inp stack.mrc -out raptor ' \
                '-diam %s -mark %s -white stack.mrc' % \
                (str(diameter), str(markers))
        else:
            raptor_cmd = 'raptor -exec %s ' % imod_path + \
                '-path . -inp stack.mrc -out raptor ' \
                '-diam %s -mark %s stack.mrc' % \
                (str(diameter), str(markers))
        os.system(raptor_cmd)

        alifile = 'raptor/align/stack.ali'
        logfile = 'raptor/align/stack_RAPTOR.log'
        if os.path.isfile(alifile):
            with open(alifile, 'rb') as h:
                np.fromfile(h, np.uint8, 1024)
                temp = np.fromfile(h, np.float32)

            if np.mod(len(temp), shape[0]) != 0:
                logger.info('************************************'
                            '**************')
                logger.info('RAPTOR alignment was unable to fit all images.')
                logger.info('Improve rough alignment or image quality.\n')
                logger.info('*************************************'
                            '*************\n\n')
                logger.info('RAPTOR log file contents:')
                logger.info('*************************************'
                            '*************')
                with open(logfile, 'r') as h:
                    logger.info(h.read())

            else:
                ali.data = temp.reshape([shape[0], shape[1], shape[2]])
                logger.info('**************************************'
                            '************')
                logger.info('RAPTOR alignment complete.\n')
                logger.info('**************************************'
                            '************')
                logger.info('\n\n')
                logger.info('RAPTOR log file contents:')
                logger.info('***************************************'
                            '***********')
                with open(logfile, 'r') as h:
                    logger.info(h.read())
        else:
            logger.info('*******************************************'
                        '*******')
            logger.info('RAPTOR alignment failed.')
            logger.info('Improve rough alignment or image quality.')
            logger.info('*******************************************'
                        '*******\n\n')
            logger.info('RAPTOR log file contents:')
            logger.info('*******************************************'
                        '*******')
            with open(logfile, 'r') as h:
                logger.info(h.read())

        os.chdir(orig_path)
        return ali

    def set_tilts(self, start, increment):
        """
        Calibrate the tilt axis of the image stack.

        Args
        ----------
        start : float or integer
            Tilt angle of first image in stack

        increment : float or integer
            Tilt increment between images

        """
        nimages = self.data.shape[0]
        self.axes_manager[0].name = 'Tilt'
        self.axes_manager[0].units = 'degrees'
        self.axes_manager[0].scale = increment
        self.axes_manager[0].offset = start
        tilts = np.arange(start, nimages * increment + start, increment)

        if not self.metadata.has_item("Tomography"):
            self.metadata.add_node("Tomography")
            self.metadata.Tomography.set_item("tilts", tilts)
            self.metadata.Tomography.set_item("tiltaxis", 0)
            self.metadata.Tomography.set_item("xshift", 0)
            self.metadata.Tomography.set_item("yshift", 0)
            self.metadata.Tomography.set_item("shifts", None)
            self.metadata.Tomography.set_item("cropped", False)
        else:
            self.metadata.Tomography.set_item("tilts", tilts)
        return

    def manual_align(self, nslice, xshift=0, yshift=0, display=False):
        """
        Manually shift one portion of a stack with respect to the other.

        Args
        ----------
        nslice : integer
            Slice position at which to implement shift
        xshift : integer
            Number of pixels with which to shift the second portion of the
            stack relative to the first in the X dimension.
        yshift : integer
            Number of pixels with which to shift the second portion of the
            stack relative to the first in the Y dimension.
        display : bool
            If True, display the result.

        """
        output = self.deepcopy()
        if yshift == 0:
            if xshift > 0:
                output.data = output.data[:, :, :-xshift]
                output.data[0:nslice, :, :] = self.data[0:nslice, :, xshift:]
                output.data[nslice:, :, :] = self.data[nslice:, :, :-xshift]
            elif xshift < 0:
                output.data = output.data[:, :, :xshift]
                output.data[0:nslice, :, :] = self.data[0:nslice, :, :xshift]
                output.data[nslice:, :, :] = self.data[nslice:, :, -xshift:]
            else:
                pass

        elif xshift == 0:
            if yshift > 0:
                output.data = output.data[:, :-yshift, :]
                output.data[0:nslice, :, :] = self.data[0:nslice, yshift:, :]
                output.data[nslice:, :, :] = self.data[nslice:, :-yshift, :]
            elif yshift < 0:
                output.data = output.data[:, :yshift, :]
                output.data[0:nslice, :, :] = self.data[0:nslice, :yshift, :]
                output.data[nslice:, :, :] = self.data[nslice:, -yshift:, :]
            else:
                pass
        else:
            if (xshift > 0) and (yshift > 0):
                output.data = output.data[:, :-yshift, :-xshift]
                output.data[0:nslice, :, :] = \
                    self.data[0:nslice, yshift:, xshift:]
                output.data[nslice:, :, :] = \
                    self.data[nslice:, :-yshift, :-xshift]
            elif (xshift > 0) and (yshift < 0):
                output.data = output.data[:, :yshift, :-xshift]
                output.data[0:nslice, :, :] = \
                    self.data[0:nslice, :yshift, xshift:]
                output.data[nslice:, :, :] = \
                    self.data[nslice:, -yshift:, :-xshift]
            elif (xshift < 0) and (yshift > 0):
                output.data = output.data[:, :-yshift, :xshift]
                output.data[0:nslice, :, :] = \
                    self.data[0:nslice, yshift:, :xshift]
                output.data[nslice:, :, :] = \
                    self.data[nslice:, :-yshift, -xshift:]
            elif (xshift < 0) and (yshift < 0):
                output.data = output.data[:, :yshift, :xshift]
                output.data[0:nslice, :, :] = \
                    self.data[0:nslice, :yshift, :xshift]
                output.data[nslice:, :, :] = \
                    self.data[nslice:, -yshift:, -xshift:]
            else:
                pass
        if display:
            old_im1 = self.data[nslice - 1, :, :]
            old_im2 = self.data[nslice, :, :]
            new_im1 = output.data[nslice - 1, :, :]
            new_im2 = output.data[nslice, :, :]
            old_im1 = old_im1 - old_im1.min()
            old_im1 = old_im1 / old_im1.max()
            old_im2 = old_im2 - old_im2.min()
            old_im2 = old_im2 / old_im2.max()
            new_im1 = new_im1 - new_im1.min()
            new_im1 = new_im1 / new_im1.max()
            new_im2 = new_im2 - new_im2.min()
            new_im2 = new_im2 / new_im2.max()

            fig, ax = plt.subplots(2, 3)
            ax[0, 0].imshow(old_im1)
            ax[0, 1].imshow(old_im2)
            ax[0, 2].imshow(old_im1 - old_im2, clim=[-0.5, 0.5])

            ax[1, 0].imshow(new_im1)
            ax[1, 1].imshow(new_im2)
            ax[1, 2].imshow(new_im1 - new_im2, clim=[-0.5, 0.5])

        return output

    def save_raw(self, filename=None):
        """
        Save TomoStack data as a .raw/.rpl file pair.

        Args
        ----------
        filname : string (optional)
            Name of file to receive data. If not specified, the metadata will
            be used. Data dimensions and data type will be appended.

        """
        datashape = self.data.shape

        if filename is None:
            filename = self.metadata.General.title
        else:
            filename, ext = os.path.splitext(filename)

        filename = filename + '_%sx%sx%s_%s.rpl' % (str(datashape[0]),
                                                    str(datashape[1]),
                                                    str(datashape[2]),
                                                    self.data.dtype.name)
        self.save(filename)
        return

    def recon_error(self, nslice=None, iterations=50, constrain=True,
                    cuda=None):
        """
        Determine the optimum number of iterations for reconstruction.

        Evaluates the difference between reconstruction and input data
        at each iteration and terminates when the change between iterations is
        below tolerance.

        Args
        ----------
        algorithm : str
            Reconstruction algorithm use.
        nslice : int
            Location at which to perform the evaluation.
        constrain : boolean
            If True, perform SIRT reconstruction with a non-negativity
            constraint.  Default is True
        cuda : boolean
            If True, perform reconstruction using the GPU-accelrated algorithm.
            Default is True

        Returns
        ----------
        error : Numpy array
            Sum of squared difference between the forward-projected
            reconstruction and the input sinogram at each iteration

        rec_stack : Hyperspy Signal2D
            Signal containing the SIRT reconstruction at each iteration
            for visual inspection.

        Examples
        ----------
        >>> import tomotools.datasets as ds
        >>> stack = ds.get_needle_data(True)
        >>> error, rec_stack = s.recon_error(iterations=5)

        """
        if not nslice:
            nslice = np.int32(self.data.shape[1] / 2)

        if not cuda:
            cuda = astra.use_cuda()
        sinogram = self.isig[:, nslice:nslice+1].deepcopy()
        angles = self.metadata.Tomography.tilts
        rec_stack, error = recon.astra_sirt_error(sinogram, angles,
                                                  iterations=iterations,
                                                  constrain=constrain,
                                                  thresh=0, cuda=False)
        return rec_stack, error
