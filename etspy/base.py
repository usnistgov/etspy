# -*- coding: utf-8 -*-
#
# This file is part of ETSpy

"""
Primary module for ETSpy package.

Contains the TomoStack class and its methods.

@author: Andrew Herzing
"""

import logging
import os

import astra
import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np
import pylab as plt
from hyperspy.signals import Signal1D, Signal2D
from scipy import ndimage
from skimage import transform

from etspy import align, recon

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CommonStack(Signal2D):
    """
    Create a CommonStack object for tomography data.

    Note: All attributes are initialized with values of None or 0.0
    in __init__ unless they are already defined
    """

    def plot(self, navigator="slider", *args, **kwargs):
        """Plot function to set default navigator to 'slider'."""
        super().plot(navigator, *args, **kwargs)

    def change_data_type(self, dtype):
        """
        Change data type.

        Use instead of the inherited change_dtype function of Hyperspy which results in
        conversion of the Stack to a Signal2D.

        """
        self.data = self.data.astype(dtype)

    def invert(self):
        """
        Invert the contrast levels of an entire Stack.

        Returns
        ----------
        inverted : CommonStack object
            Copy of the input stack with contrast inverted

        Examples
        --------
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> s_inverted = stack.invert()
        """
        maxvals = self.data.max(2).max(1)
        maxvals = maxvals.reshape([self.data.shape[0], 1, 1])
        minvals = self.data.min(2).min(1)
        minvals = minvals.reshape([self.data.shape[0], 1, 1])
        ranges = maxvals - minvals

        inverted = self.deepcopy()
        inverted.data = inverted.data - np.reshape(
            inverted.data.mean(2).mean(1), [self.data.shape[0], 1, 1]
        )
        inverted.data = (inverted.data - minvals) / ranges

        inverted.data = inverted.data - 1
        inverted.data = np.sqrt(inverted.data**2)

        inverted.data = (inverted.data * ranges) + minvals

        return inverted

    def normalize(self, width=3):
        """
        Normalize the contrast levels of an entire Stack.

        Args
        ----------
        width : integer
            Number of standard deviations from the mean to set
            as maximum intensity level.

        Returns
        ----------
        normalized : CommonStack object
            Copy of the input stack with intensities normalized

        Examples
        --------
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> s_normalized = stack.normalize()

        """
        normalized = self.deepcopy()
        minvals = np.reshape(
            (normalized.data.min(2).min(1)), [self.data.shape[0], 1, 1]
        )
        normalized.data = normalized.data - minvals
        meanvals = np.reshape(
            (normalized.data.mean(2).mean(1)), [self.data.shape[0], 1, 1]
        )
        stdvals = np.reshape(
            (normalized.data.std(2).std(1)), [self.data.shape[0], 1, 1]
        )
        normalized.data = normalized.data / (meanvals + width * stdvals)
        return normalized

    # noinspection PyTypeChecker
    def savemovie(
        self,
        start,
        stop,
        axis="XY",
        fps=15,
        dpi=100,
        outfile=None,
        title="output.avi",
        clim=None,
        cmap="afmhot",
    ):
        """
        Save the Stack as an AVI movie file.

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

        if axis == "XY":
            im = ax.imshow(
                self.data[:, start, :], interpolation="none", cmap=cmap, clim=clim
            )
        elif axis == "XZ":
            im = ax.imshow(
                self.data[start, :, :], interpolation="none", cmap=cmap, clim=clim
            )
        elif axis == "YZ":
            im = ax.imshow(
                self.data[:, :, start], interpolation="none", cmap=cmap, clim=clim
            )
        else:
            raise ValueError("Unknown axis!")
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

        if axis == "XY":
            ani = animation.FuncAnimation(fig, updatexy, frames)
        elif axis == "XZ":
            ani = animation.FuncAnimation(fig, updatexz, frames)
        elif axis == "YZ":
            ani = animation.FuncAnimation(fig, updateyz, frames)
        else:
            raise ValueError("Axis not understood!")

        writer = animation.writers["ffmpeg"](fps=fps)
        ani.save(outfile, writer=writer, dpi=dpi)
        plt.close()
        return

    def save_raw(self, filename=None):
        """
        Save Stack data as a .raw/.rpl file pair.

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

        filename = filename + "_%sx%sx%s_%s.rpl" % (
            str(datashape[0]),
            str(datashape[1]),
            str(datashape[2]),
            self.data.dtype.name,
        )
        self.save(filename)
        return

    def stats(self):
        """Print basic stats about Stack data to terminal."""
        print("Mean: %.1f" % self.data.mean())
        print("Std: %.2f" % self.data.std())
        print("Max: %.1f" % self.data.max())
        print("Min: %.1f\n" % self.data.min())
        return

    def trans_stack(self, xshift=0.0, yshift=0.0, angle=0.0, interpolation="linear"):
        """
        Transform the stack using the skimage Affine transform.

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
        out : CommonStack object
            Transformed copy of the input stack

        Examples
        ----------
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> xshift = 10.0
        >>> yshift = 3.5
        >>> angle = -15.2
        >>> transformed = stack.trans_stack(xshift, yshift, angle)
        >>> transformed
        <TomoStack, title: , dimensions: (77|256, 256)>

        """
        transformed = self.deepcopy()
        theta = np.pi * angle / 180.0
        center_y, center_x = np.float32(np.array(transformed.data.shape[1:]) / 2)

        rot_mat = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        trans_mat = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])

        rev_mat = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])

        rotation_mat = np.dot(np.dot(trans_mat, rot_mat), rev_mat)

        shift = np.array(
            [[1, 0, np.float32(xshift)], [0, 1, np.float32(-yshift)], [0, 0, 1]]
        )

        full_transform = np.dot(shift, rotation_mat)
        tform = transform.AffineTransform(full_transform)

        if interpolation.lower() == "nearest" or interpolation.lower() == "none":
            interpolation_order = 0
        elif interpolation.lower() == "linear":
            interpolation_order = 1
        elif interpolation.lower() == "cubic":
            interpolation_order = 3
        else:
            raise ValueError(
                "Interpolation method %s unknown. "
                "Must be 'nearest', 'linear', or 'cubic'" % interpolation
            )

        for i in range(0, self.data.shape[0]):
            transformed.data[i, :, :] = transform.warp(
                transformed.data[i, :, :],
                inverse_map=tform.inverse,
                order=interpolation_order,
            )

        transformed.metadata.Tomography.xshift = (
            self.metadata.Tomography.xshift + xshift
        )

        transformed.metadata.Tomography.yshift = (
            self.metadata.Tomography.yshift + yshift
        )

        transformed.metadata.Tomography.tiltaxis = (
            self.metadata.Tomography.tiltaxis + angle
        )
        return transformed


class TomoStack(CommonStack):
    """
    Create a TomoStack object for tomography data.

    Parameters
    ----------
    CommonStack : CommonStack
        CommonStack class
    """

    def plot_sinos(self, *args, **kwargs):
        """Plot the TomoStack in sinogram orientation."""
        self.swap_axes(1, 0).swap_axes(1, 2).plot(navigator="slider", *args, **kwargs)
        return

    def remove_projections(self, projections=None):
        """
        Remove projections from tilt series.

        Args
        ----------
        projections : list
            List of projection indices in integers to remove

        Returns
        ----------
        s_new : TomoStack
            Copy of self with indicated projections removed

        """
        if projections is None:
            raise ValueError("No projections provided")
        nprojs = len(projections)
        s_new = self.deepcopy()
        s_new.axes_manager[0].size -= nprojs
        mask = np.ones(self.data.shape[0], dtype=bool)
        mask[projections] = False
        s_new.data = self.data[mask]
        s_new.metadata.Tomography.shifts = s_new.metadata.Tomography.shifts[mask]
        s_new.metadata.Tomography.tilts = s_new.metadata.Tomography.tilts[mask]
        return s_new

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

        ax1.imshow(im1, cmap="gray")
        ax1.set_axis_off()
        ax1.set_title("Reference image")

        ax2.imshow(im2, cmap="gray")
        ax2.set_axis_off()
        ax2.set_title("Offset image")

        ax3.imshow(cc_image.real, cmap="inferno")
        ax3.set_axis_off()
        ax3.set_title("Cross-correlation")
        return fig

    # TODO: allow a list of signals for 'other'
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
        no_shifts = np.all(self.metadata.Tomography.shifts == 0)
        no_xshift = any(
            [
                self.metadata.Tomography.xshift is None,
                self.metadata.Tomography.xshift == 0.0,
            ]
        )
        no_yshift = any(
            [
                self.metadata.Tomography.xshift is None,
                self.metadata.Tomography.xshift == 0.0,
            ]
        )
        no_rotation = any(
            [
                self.metadata.Tomography.tiltaxis is None,
                self.metadata.Tomography.tiltaxis == 0.0,
            ]
        )

        if all([no_shifts, no_xshift, no_yshift, no_rotation]):
            raise ValueError("No transformations have been applied to this stack")

        out = align.align_to_other(self, other)

        return out

    def filter(self, method="median", size=5, taper=0.1):
        """
        Apply one of several image filters to an entire TomoStack.

        Args
        ----------
        method : string
            Type of filter to apply. Must be 'median', 'bpf', 'both', or 'sobel'.
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
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> filtered = stack.filter(method='median')

        """
        filtered = self.deepcopy()
        if method == "median":
            filtered.data = ndimage.median_filter(filtered.data, size=(1, size, size))
        elif method == "sobel":
            for i in range(0, filtered.data.shape[0]):
                dx = ndimage.sobel(filtered.data[i, :, :], 0)
                dy = ndimage.sobel(filtered.data[i, :, :], 1)
                filtered.data[i, :, :] = np.hypot(dx, dy)
        elif method == "both":
            filtered.data = ndimage.median_filter(filtered.data, size=(1, size, size))
            for i in range(0, filtered.data.shape[0]):
                dx = ndimage.sobel(filtered.data[i, :, :], 0)
                dy = ndimage.sobel(filtered.data[i, :, :], 1)
                filtered.data[i, :, :] = np.hypot(dx, dy)
        elif method == "bpf":
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
            lpf = 1 / (1.0 + (r / lp_freq) ** (2 * lp_sigma))

            hpf = 1 - (1 / (1.0 + (r / hp_freq) ** (2 * hp_sigma)))
            bpf = lpf * hpf
            F_filtered = F * bpf

            filtered.data = np.fft.ifft2(np.fft.ifftshift(F_filtered)).real

            h = np.hamming(rows)
            ham2d = np.sqrt(np.outer(h, h))
            filtered.data = filtered.data * ham2d
        else:
            raise ValueError(
                "Unknown filter method. Must be 'median', 'sobel', 'both', or 'bpf'"
            )
        if taper:
            taper_size = np.int32(np.array(taper) * self.data.shape[1:])
            filtered.data = np.pad(
                filtered.data,
                [
                    (0, 0),
                    (taper_size[0], taper_size[0]),
                    (taper_size[1], taper_size[1]),
                ],
                mode="constant",
            )
        return filtered

    def stack_register(
        self, method="PC", start=None, show_progressbar=False, crop=False, **kwargs
    ):
        """
        Register stack spatially.

        Options are phase correlation (PC) maximization, StackReg, center of
        mass ('COM'), or combined center of mass and common line methods.
        See docstring for etspy.align.align_stack for details.

        Args
        ----------
        method : string
            Algorithm to use for registration calculation. Must be either
            'PC', 'StackReg', 'COM', or 'COM-CL'.
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
        Registration with phase correlation algorithm (PC)
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> regPC = stack.stack_register('PC')

        Registration with center of mass tracking (COM)
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> regCOM = stack.stack_register('COM')

        Registration with StackReg
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> regSR = stack.stack_register('StackReg')

        Registration with center of mass and common line (COM-CL)
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> regCOMCL = stack.stack_register('COM-CL')

        """
        method = method.lower()
        if method in ["pc", "com", "stackreg", "com-cl"]:
            out = align.align_stack(self, method, start, show_progressbar, **kwargs)
        else:
            raise ValueError(
                "Unknown registration method: "
                "%s. Must be PC, StackReg, or COM" % method
            )

        if crop:
            out = align.shift_crop(out)
        return out

    def tilt_align(self, method, **kwargs):
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
        TomoStack by analyzing features in the projected maximum image.  A combination
        of edge detection and Hough transform analysis is used to determine the global
        rotation of the stack.  Optionally, the global shift of the tilt axis can also
        be calculated by minimization of the sum of the reconstruction.

        Args
        ----------
        method : string
            Algorithm to use for registration alignment. Must be either 'CoM' or
            'MaxImage'.

        **kwargs: Additional keyword arguments. Possible keys include:
                - nslices (int): Number of slices to use for the center of mass tilt alignment.
                - locs (list): Location along tilt axis to use for center of mass tilt alignment.
                - limit (integer or float): Maximum rotation angle to use for MaxImage calculation
                - delta (float): Angular increment in degrees for MaxImage calculation
                - plot_results (bool): if True, plot results of Hough line analysis
                - also_shift (bool): if True, also calculate global shift of tilt axis
                - shift_limit (int): Search range for global shift of tilt axis

        Returns
        ----------
        out : TomoStack object
            Copy of the input stack rotated by calculated angle

        Examples
        ----------
        Align tilt axis using the center of mass (CoM) method
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> reg = stack.stack_register('PC',show_progressbar=False)
        >>> method = 'CoM'
        >>> ali = reg.tilt_align(method, locs=[50,100,160])

        Align tilt axis using the maximum image method
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data()
        >>> reg = stack.stack_register('PC',show_progressbar=False)
        >>> method = 'MaxImage'
        >>> ali = reg.tilt_align(method)

        """
        method = method.lower()

        if method == "com":
            nslices = kwargs.get("nslices", 20)
            locs = kwargs.get("locs", None)
            out = align.tilt_com(self, locs, nslices)
        elif method == "maximage":
            limit = kwargs.get("limit", 10)
            delta = kwargs.get("delta", 0.3)
            plot_results = kwargs.get("plot_results", False)
            also_shift = kwargs.get("also_shift", False)
            shift_limit = kwargs.get("shift_limit", 20)
            out = align.tilt_maximage(
                self, limit, delta, plot_results, also_shift, shift_limit
            )
        else:
            raise ValueError(
                "Invalid alignment method: %s." "Must be 'CoM' or 'MaxImage'" % method
            )
        return out

    def reconstruct(
        self,
        method="FBP",
        iterations=None,
        constrain=False,
        thresh=0,
        cuda=None,
        thickness=None,
        show_progressbar=True,
        **kwargs
    ):
        """
        Reconstruct a TomoStack series using one of the available methods.

        Args
        ----------
        method : string
            Reconstruction algorithm to use.  Must be'FBP' (default), 'SIRT', 'SART', or 'DART'
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
        show_progressbar : bool
            If True, show a progress bar for the reconstruction. Default is True.
        **kwargs: Additional keyword arguments. Possible keys include:
        - ncores (int): Number of cores to use for multithreaded reconstructions.
        - sino_filter (str): Filter to apply for filtered backprojection.  Default is shepp-logan.
        - dart_iterations (int): Number of iterations to employ for DART reconstruction.

        Returns
        ----------
        out : TomoStack object
            TomoStack containing the reconstructed volume

        Examples
        ----------
        Filtered backprojection (FBP) reconstruction
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data(True)
        >>> slices = stack.isig[:, 120:121].deepcopy()
        >>> rec = slices.reconstruct('FBP', cuda=False, show_progressbar=False)

        Simultaneous iterative reconstruction technique (SIRT) reconstruction
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data(True)
        >>> slices = stack.isig[:, 120:121].deepcopy()
        >>> rec = slices.reconstruct('SIRT',iterations=5, cuda=False, show_progressbar=False)

        Simultaneous iterative reconstruction technique (SIRT) reconstruction
        with positivity constraint
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data(True)
        >>> slices = stack.isig[:, 120:121].deepcopy()
        >>> iterations = 5
        >>> constrain = True
        >>> thresh = 0
        >>> rec = slices.reconstruct('SIRT',iterations, constrain, thresh, cuda=False, show_progressbar=False)

        Discreate algebraice reconstruction technique (DART) reconstruction
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data(True)
        >>> slices = stack.isig[:, 120:121].deepcopy()
        >>> gray_levels = [0., slices.data.max()/2, slices.data.max()]
        >>> rec = slices.reconstruct('DART',iterations=5, cuda=False, gray_levels=gray_levels, p=0.99, dart_iterations=5, show_progressbar=False)

        """
        if method.lower() not in [
            "fbp",
            "sirt",
            "sart",
            "dart",
        ]:
            raise ValueError("Unknown reconstruction algorithm: %s" % method)
        if cuda is None:
            if astra.use_cuda():
                logger.info("CUDA detected with Astra")
                cuda = True
            else:
                cuda = False
                logger.info("CUDA not detected with Astra")

        ncores = kwargs.get("ncores", None)
        sino_filter = kwargs.get("sino_filter", "shepp-logan")
        if method.lower() == "dart":
            dart_iterations = kwargs.get("dart_iterations", 5)
            p = kwargs.get("p", 0.99)
            gray_levels = kwargs.get("gray_levels", None)
            if not isinstance(gray_levels, (np.ndarray, list)):
                raise ValueError(
                    "Unknown type (%s) for gray_levels" % type(gray_levels)
                )
            elif gray_levels is None:
                raise ValueError("gray_levels must be provided for DART")
        else:
            dart_iterations = None
            p = None
            gray_levels = None
        rec = recon.run(
            self,
            method,
            iterations,
            constrain,
            thresh,
            cuda,
            thickness,
            ncores,
            sino_filter,
            gray_levels,
            dart_iterations,
            p,
            show_progressbar,
        )

        axes_dict = self.axes_manager.as_dictionary()
        rec_axes_dict = [
            axes_dict["axis-2"],
            dict(axes_dict["axis-1"]),
            axes_dict["axis-1"],
        ]
        rec_axes_dict[1]["name"] = "z"
        rec_axes_dict[1]["size"] = rec.shape[1]
        rec = RecStack(rec, axes=rec_axes_dict)
        return rec

    def test_align(
        self,
        tilt_shift=0.0,
        tilt_rotation=0.0,
        slices=None,
        thickness=None,
        method="FBP",
        iterations=50,
        constrain=True,
        cuda=None,
        thresh=0,
        vmin_std=0.1,
        vmax_std=10,
    ):
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
        method : str
            Reconstruction algorithm to use.  Must be 'FBP', 'SIRT', or 'SART'.
        cuda : bool
            If True, use CUDA-accelerated Astra algorithms.  If None, use
            CUDA if astra.use_cuda() is True.
        thresh : float
            Minimum value for reconstruction
        vmin_std, vmax_std : float
            Number of standard deviations from mean to use for scaling the displayed slices
        """
        if slices is None:
            mid = np.int32(self.data.shape[2] / 2)
            slices = np.int32([mid / 2, mid, mid + mid / 2])

        if (tilt_shift != 0.0) or (tilt_rotation != 0.0):
            shifted = self.trans_stack(xshift=0, yshift=tilt_shift, angle=tilt_rotation)
        else:
            shifted = self.deepcopy()
        shifted.data = shifted.data[:, :, slices]

        shifted.axes_manager[0].axis = self.axes_manager[0].axis
        if cuda is None:
            if astra.use_cuda():
                logger.info("CUDA detected with Astra")
                cuda = True
            else:
                cuda = False
                logger.info("CUDA not detected with Astra")
        rec = shifted.reconstruct(
            method=method,
            iterations=iterations,
            constrain=constrain,
            thickness=thickness,
            cuda=cuda,
            thresh=thresh,
            show_progressbar=False,
        )

        if "ipympl" in mpl.get_backend().lower():
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 3))
        elif "nbagg" in mpl.get_backend().lower():
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        minvals = rec.data.mean((1, 2)) - vmin_std * rec.data.std((1, 2))
        minvals[minvals < 0] = 0
        maxvals = rec.data.mean((1, 2)) + vmax_std * rec.data.std((1, 2))

        ax1.imshow(rec.data[0, :, :], cmap="afmhot", vmin=minvals[0], vmax=maxvals[0])
        ax1.set_title("Slice %s" % str(slices[0]))
        ax1.set_axis_off()

        ax2.imshow(rec.data[1, :, :], cmap="afmhot", vmin=minvals[1], vmax=maxvals[1])
        ax2.set_title("Slice %s" % str(slices[1]))
        ax2.set_axis_off()

        ax3.imshow(rec.data[2, :, :], cmap="afmhot", vmin=minvals[2], vmax=maxvals[2])
        ax3.set_title("Slice %s" % str(slices[2]))
        ax3.set_axis_off()
        fig.tight_layout()
        return

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
        self.axes_manager[0].name = "Tilt"
        self.axes_manager[0].units = "degrees"
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
            if (xshift > 0) and (yshift > 0):
                output.data = output.data[:, :-yshift, :-xshift]
                output.data[0:nslice, :, :] = self.data[0:nslice, yshift:, xshift:]
                output.data[nslice:, :, :] = self.data[nslice:, :-yshift, :-xshift]
            elif (xshift > 0) and (yshift < 0):
                output.data = output.data[:, :yshift, :-xshift]
                output.data[0:nslice, :, :] = self.data[0:nslice, :yshift, xshift:]
                output.data[nslice:, :, :] = self.data[nslice:, -yshift:, :-xshift]
            elif (xshift < 0) and (yshift > 0):
                output.data = output.data[:, :-yshift, :xshift]
                output.data[0:nslice, :, :] = self.data[0:nslice, yshift:, :xshift]
                output.data[nslice:, :, :] = self.data[nslice:, :-yshift, -xshift:]
            elif (xshift < 0) and (yshift < 0):
                output.data = output.data[:, :yshift, :xshift]
                output.data[0:nslice, :, :] = self.data[0:nslice, :yshift, :xshift]
                output.data[nslice:, :, :] = self.data[nslice:, -yshift:, -xshift:]
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

    def recon_error(
        self,
        nslice=None,
        algorithm="SIRT",
        iterations=50,
        constrain=True,
        cuda=None,
        thresh=0,
    ):
        """
        Determine the optimum number of iterations for reconstruction.

        Evaluates the difference between reconstruction and input data
        at each iteration and terminates when the change between iterations is
        below tolerance.

        Args
        ----------
        algorithm : str
            Reconstruction algorithm use.  Must be 'SIRT' (default) or 'SART'.
        nslice : int
            Location at which to perform the evaluation.
        constrain : boolean
            If True, perform SIRT reconstruction with a non-negativity
            constraint.  Default is True
        cuda : boolean
            If True, perform reconstruction using the GPU-accelrated algorithm.
            Default is True
        thresh : integer or float
            Value above which to constrain the reconstructed data

        Returns
        ----------
        rec_stack : Hyperspy Signal2D
            Signal containing the SIRT reconstruction at each iteration
            for visual inspection.
        error : Hyperspy Signal1D
            Sum of squared difference between the forward-projected
            reconstruction and the input sinogram at each iteration

        Examples
        ----------
        >>> import etspy.datasets as ds
        >>> stack = ds.get_needle_data(True)
        >>> rec_stack, error = stack.recon_error(iterations=5)

        """
        if self.metadata.Tomography.tilts is None:
            raise ValueError("Tilt angles not defined")

        if not nslice:
            nslice = int(self.data.shape[2] / 2)

        if cuda is None:
            if astra.use_cuda():
                logger.info("CUDA detected with Astra")
                cuda = True
            else:
                cuda = False
                logger.info("CUDA not detected with Astra")
        sinogram = self.isig[nslice, :].data
        angles = self.metadata.Tomography.tilts
        rec_stack, error = recon.astra_error(
            sinogram,
            angles,
            method=algorithm,
            iterations=iterations,
            constrain=constrain,
            thresh=thresh,
            cuda=cuda,
        )
        rec_stack = Signal2D(rec_stack)
        rec_stack.axes_manager[0].name = algorithm.upper() + " iteration"
        rec_stack.axes_manager[0].scale = 1
        rec_stack.axes_manager[1].name = self.axes_manager[2].name
        rec_stack.axes_manager[1].scale = self.axes_manager[2].scale
        rec_stack.axes_manager[1].units = self.axes_manager[2].units
        rec_stack.axes_manager[2].name = "z"
        rec_stack.axes_manager[2].scale = self.axes_manager[2].scale
        rec_stack.axes_manager[2].units = self.axes_manager[2].units
        rec_stack.navigator = "slider"

        error = Signal1D(error)
        error.axes_manager[0].name = algorithm.upper() + " Iteration"
        error.metadata.Signal.quantity = "Sum of Squared Difference"
        return rec_stack, error


class RecStack(CommonStack):
    """
    Create a RecStack object for tomography data.

    Parameters
    ----------
    CommonStack : CommonStack
        CommonStack class
    """

    def plot_slices(
        self, xslice=None, yslice=None, zslice=None, vmin_std=0.1, vmax_std=5
    ):
        """
        Plot slices along all three axes of a reconstruction stack.

        Args
        ----------
        yslice, zslice, xslice : int
            Indices of slices to plot

        vmin_std, vmax_std : float
            Number of standard deviations from mean to use for scaling the displayed slices

        Returns
        ----------
        fig : Matplotlib Figure

        """
        if xslice is None:
            xslice = np.uint16(self.data.shape[0] / 2)
        if yslice is None:
            yslice = np.uint16(self.data.shape[1] / 2)
        if zslice is None:
            zslice = np.uint16(self.data.shape[2] / 2)

        if "ipympl" in mpl.get_backend().lower():
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 3))
        elif "nbagg" in mpl.get_backend().lower():
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        slices = [
            self.data[xslice, :, :],
            self.data[:, zslice, :],
            self.data[:, :, yslice],
        ]
        minvals = [slices[i].mean() - vmin_std * slices[i].std() for i in range(3)]
        minvals = [x if x >= 0 else 0 for x in minvals]
        maxvals = [slices[i].mean() + vmax_std * slices[i].std() for i in range(3)]

        ax1.imshow(slices[0], cmap="afmhot", vmin=minvals[0], vmax=maxvals[0])
        ax1.set_title("Z-Y Slice %s" % str(xslice))
        ax1.set_ylabel("Z")
        ax1.set_xlabel("Y")

        ax2.imshow(slices[1], cmap="afmhot", vmin=minvals[1], vmax=maxvals[1])
        ax2.set_title("Y-X Slice %s" % str(zslice))
        ax2.set_ylabel("Y")
        ax2.set_xlabel("X")

        ax3.imshow(slices[2].T, cmap="afmhot", vmin=minvals[2], vmax=maxvals[2])
        ax3.set_title("Z-X Slice %s" % str(yslice))
        ax3.set_ylabel("Z")
        ax3.set_xlabel("X")
        fig.tight_layout()

        [i.set_xticks([]) for i in [ax1, ax2, ax3]]
        [i.set_yticks([]) for i in [ax1, ax2, ax3]]
        return fig
