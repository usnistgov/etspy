"""
Primary module for ETSpy package.

Contains the TomoStack class and its methods.
"""

import logging  # noqa: I001
from abc import ABC
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Self, Tuple, Union, cast

import astra
import matplotlib as mpl
import numpy as np
import pylab as plt
from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D
from hyperspy.axes import UniformDataAxis as Uda
from hyperspy.misc.utils import DictionaryTreeBrowser as Dtb
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from scipy import ndimage
from skimage import transform

from etspy import AlignmentMethod, AlignmentMethodType, FbpMethodType, ReconMethodType
from etspy import _format_choices as _fmt
from etspy import _get_literal_hint_values as _get_lit
from etspy import align, recon

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CommonStack(Signal2D, ABC):
    """
    An abstract base class for tomography data.

    .. abstract::

       This class is intended to be subclassed (*e.g.* by
       :py:class:`~etspy.base.TomoStack` and :py:class:`~etspy.base.RecStack`) and
       should not be instantiated directly. Doing so will raise a
       :py:exc:`NotImplementedError`.

    All arguments are passed to the :py:class:`~hyperspy.api.signals.Signal2D`
    constructor and should be used as documented for that method.

    Group
    -----
    signals

    Order
    -----
    3
    """

    def __init__(self, *args, **kwargs):
        """
        Create an ETSpy signal instance.

        Raises
        ------
        NotImplementedError
            :py:class:`~etspy.base.CommonStack` is not intended to be used directly.
            One of its sub-classes (:py:class:`~etspy.base.TomoStack` or
            :py:class:`~etspy.base.RecStack`) should be used instead.
        """
        if type(self) is CommonStack:
            msg = (
                "CommonStack should not be instantiated directly. Use one of its "
                "sub-classes instead (TomoStack or RecStack)"
            )
            raise NotImplementedError(msg)

        super().__init__(*args, **kwargs)

    def plot(self, navigator: str = "slider", *args, **kwargs):
        """
        Override of plot function to set default HyperSpy navigator to 'slider'.

        Any other arguments (keyword and non-keyword) are passed to
        :py:meth:`hyperspy.api.signals.Signal2D.plot`
        """
        super().plot(navigator=navigator, *args, **kwargs)  # noqa: B026

    def change_data_type(self, dtype: Union[str, np.dtype]):
        """
        Change the data type of a stack.

        Use instead of the inherited change_dtype function of Hyperspy which results in
        conversion of the Stack to a Signal2D.

        Parameters
        ----------
        dtype
            A string that represents a NumPy data type, or a specific data type
        """
        self.data = self.data.astype(dtype)

    def invert(self) -> Self:
        """
        Create a copy of a Stack with inverted contrast levels.

        Returns
        -------
        inverted : Self
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
            inverted.data.mean(2).mean(1),
            [self.data.shape[0], 1, 1],
        )
        inverted.data = (inverted.data - minvals) / ranges

        inverted.data = inverted.data - 1
        inverted.data = np.sqrt(inverted.data**2)

        inverted.data = (inverted.data * ranges) + minvals

        return inverted

    def normalize(self, width: int = 3) -> Self:
        """
        Create a copy of a stack with normalized contrast levels.

        Parameters
        ----------
        width
            Number of standard deviations from the mean to set
            as maximum intensity level.

        Returns
        -------
        normalized : Self
            Copy of the input stack with intensities normalized

        Examples
        --------
            >>> import etspy.datasets as ds
            >>> stack = ds.get_needle_data()
            >>> s_normalized = stack.normalize()
        """
        normalized = self.deepcopy()
        minvals = np.reshape(
            (normalized.data.min(2).min(1)),
            [self.data.shape[0], 1, 1],
        )
        normalized.data = normalized.data - minvals
        meanvals = np.reshape(
            (normalized.data.mean(2).mean(1)),
            [self.data.shape[0], 1, 1],
        )
        stdvals = np.reshape(
            (normalized.data.std(2).std(1)),
            [self.data.shape[0], 1, 1],
        )
        normalized.data = normalized.data / (meanvals + width * stdvals)
        return normalized

    def save_movie(
        self,
        start: int,
        stop: int,
        axis: Literal["XY", "YZ", "XZ"] = "XY",
        fps: int = 15,
        dpi: int = 100,
        outfile: str = "output.avi",
        title: str = "output.avi",
        clim: Optional[Tuple[float, float]] = None,
        cmap: str = "afmhot",
    ):
        """
        Save the Stack as an AVI movie file.

        Parameters
        ----------
        start
            Starting slice number for animation
        stop
            Ending slice number for animation
        axis
            Projection axis for the output movie.
            Must be ``'XY'`` (default), ``'YZ'`` , or ``'XZ'``
        fps
            Number of frames per second at which to create the movie.
        dpi
            Resolution to save the images in the movie.
        outfile
            Filename for output.
        title
            Title to add at the top of the movie
        clim
            Upper and lower contrast limit to use for movie
        cmap
            Matplotlib colormap to use for movie
        """
        if clim is None:
            clim = (self.data.min(), self.data.max())

        fig, ax = plt.subplots(1, figsize=(8, 8))

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if title:
            ax.set_title(title)

        if axis == "XY":
            im = ax.imshow(
                self.data[:, start, :],
                interpolation="none",
                cmap=cmap,
                clim=clim,
            )
        elif axis == "XZ":
            im = ax.imshow(
                self.data[start, :, :],
                interpolation="none",
                cmap=cmap,
                clim=clim,
            )
        elif axis == "YZ":
            im = ax.imshow(
                self.data[:, :, start],
                interpolation="none",
                cmap=cmap,
                clim=clim,
            )
        else:
            msg = (
                f'Invalid axis "{axis}". Must be one of '
                f"{_fmt(_get_lit(self.save_movie, 'axis'))}."
            )
            raise ValueError(msg)
        fig.tight_layout()

        def updatexy(n) -> Iterable[Artist]:
            tmp = self.data[:, n, :]
            im.set_data(tmp)
            return [im]

        def updatexz(n) -> Iterable[Artist]:
            tmp = self.data[n, :, :]
            im.set_data(tmp)
            return [im]

        def updateyz(n) -> Iterable[Artist]:
            tmp = self.data[:, :, n]
            im.set_data(tmp)
            return [im]

        frames = np.arange(start, stop, 1)

        if axis == "XY":
            ani = animation.FuncAnimation(fig=fig, func=updatexy, frames=frames)
        elif axis == "XZ":
            ani = animation.FuncAnimation(fig, updatexz, frames)
        elif axis == "YZ":
            ani = animation.FuncAnimation(fig, updateyz, frames)
        else:
            msg = "Axis not understood!"
            raise ValueError(msg)

        writer = animation.writers["ffmpeg"](fps=fps)
        ani.save(outfile, writer=writer, dpi=dpi)
        plt.close()

    def save_raw(self, filename: Optional[Union[str, Path]] = None):
        """
        Save Stack data as a .raw/.rpl file pair.

        Parameters
        ----------
        filename
            Name of file to receive data. If not specified, the metadata will
            be used. Data dimensions and data type will be appended.
        """
        datashape = self.data.shape

        if filename is None:
            filename = Path(str(cast(Dtb, self.metadata.General).title))
        elif isinstance(filename, str):
            filename = Path(filename)

        filename = filename.parent / (
            filename.stem + f"_{datashape[0]}x"
            f"{datashape[1]}x"
            f"{datashape[2]}_"
            f"{self.data.dtype.name}.rpl"
        )
        self.save(filename)

    def stats(self):
        """Print some basic statistics about Stack data."""
        print(f"Mean: {self.data.mean():.1f}")  # noqa: T201
        print(f"Std: {self.data.std():.2f}")  # noqa: T201
        print(f"Max: {self.data.max():.1f}")  # noqa: T201
        print(f"Min: {self.data.min():.1f}\n")  # noqa: T201

    def trans_stack(
        self,
        xshift: float = 0.0,
        yshift: float = 0.0,
        angle: float = 0.0,
        interpolation: Literal["linear", "cubic", "nearest", "none"] = "linear",
    ) -> Self:
        """
        Create a copy of a Stack, transformed using the ``skimage`` Affine transform.

        Parameters
        ----------
        xshift
            Number of pixels by which to shift in the X dimension
        yshift
            Number of pixels by which to shift the stack in the Y dimension
        angle
            Angle in degrees by which to rotate the stack about the X-Y plane
        interpolation
            Mode of interpolation to employ. Must be either ``'linear'``,
            ``'cubic'``, ``'nearest'`` or ``'none'``.  Note that ``'nearest'``
            and ``'none'`` are equivalent.  Default is ``'linear'``.

        Returns
        -------
        out : Self
            Transformed copy of the input stack

        Examples
        --------
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
        center_y, center_x = np.array(
            np.array(transformed.data.shape[1:]) / 2,
            dtype=np.float32,
        )

        rot_mat = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
        )

        trans_mat = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])

        rev_mat = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])

        rotation_mat = np.dot(np.dot(trans_mat, rot_mat), rev_mat)

        shift = np.array(
            [[1, 0, np.float32(xshift)], [0, 1, np.float32(-yshift)], [0, 0, 1]],
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
            msg = (
                f'Invalid interpolation method "{interpolation}". Must be one of '
                f"{_fmt(_get_lit(self.trans_stack, 'interpolation'))}."
            )
            raise ValueError(msg)

        for i in range(self.data.shape[0]):
            transformed.data[i, :, :] = transform.warp(
                transformed.data[i, :, :],
                inverse_map=tform.inverse,
                order=interpolation_order,
            )

        trans_tomo_meta = cast(Dtb, transformed.metadata.Tomography)
        self_tomo_meta = cast(Dtb, self.metadata.Tomography)
        trans_tomo_meta.xshift = cast(float, self_tomo_meta.xshift) + xshift
        trans_tomo_meta.yshift = cast(float, self_tomo_meta.yshift) + yshift
        trans_tomo_meta.tiltaxis = cast(float, self_tomo_meta.tiltaxis) + angle
        return transformed


class TomoStack(CommonStack):
    """
    Create a TomoStack instance, used to represent tomographic tilt series data.

    All arguments are passed to the :py:class:`~hyperspy.api.signals.Signal2D`
    constructor and should be used as documented for that method.

    Group
    -----
    signals

    Order
    -----
    1
    """

    def __init__(self, *args, **kwargs):
        """
        Create a TomoStack signal.

        Parameters
        ----------
        args
            Additional non-keyword arguments passed to
            :py:class:`~hyperspy.api.signals.Signal2D`
        kwargs
            Additional keyword arguments passed to
            :py:class:`~hyperspy.api.signals.Signal2D`
        """
        super().__init__(*args, **kwargs)

    def plot_sinos(self, *args: Tuple, **kwargs: Dict):
        """
        Plot the TomoStack in sinogram orientation.

        Parameters
        ----------
        args
            Additional non-keyword arguments passed to
            :py:meth:`~hyperspy.api.signals.Signal2D.plot`
        kwargs
            Additional keyword arguments passed to
            :py:meth:`~hyperspy.api.signals.Signal2D.plot`
        """
        self.swap_axes(1, 0).swap_axes(1, 2).plot(
            navigator="slider",
            *args,  # noqa: B026
            **kwargs,
        )

    def remove_projections(self, projections: Optional[List] = None) -> "TomoStack":
        """
        Return a copy of the TomoStack with certain projections removed from the series.

        Parameters
        ----------
        projections
            List of projection indices in integers to remove

        Returns
        -------
        s_new : TomoStack
            Copy of self with indicated projections removed

        Raises
        ------
        ValueError
            If no projections are provided
        """
        if projections is None:
            msg = "No projections provided"
            raise ValueError(msg)
        nprojs = len(projections)
        s_new = self.deepcopy()
        cast(Uda, s_new.axes_manager[0]).size -= nprojs
        mask = np.ones(self.data.shape[0], dtype=bool)
        mask[projections] = False
        s_new.data = self.data[mask]
        s_new_tomo_meta = cast(Dtb, s_new.metadata.Tomography)
        s_new_tomo_meta.shifts = s_new_tomo_meta.shifts[mask]
        s_new_tomo_meta.tilts = s_new_tomo_meta.tilts[mask]
        return s_new

    def test_correlation(
        self,
        images: Optional[Union[List[int], Tuple[int, int]]] = None,
    ) -> Figure:
        """
        Test output of cross-correlation prior to alignment.

        Parameters
        ----------
        images
            List of two numbers indicating which projections to cross-correlate.
            If ``None``, the first two images will be used.

        Returns
        -------
        fig : ~matplotlib.figure.Figure
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
    def align_other(self, other: "TomoStack") -> "TomoStack":
        """
        Apply the alignment calculated for one dataset to another.

        This will include the spatial registration, tilt axis, and tilt axis
        shift if they have been previously calculated.

        Parameters
        ----------
        other
            The tilt series which is to be aligned using the previously
            calculated parameters. The data array in the TomoStack must be of
            the same size as that in ``self.data``

        Returns
        -------
        out : TomoStack
            The result of applying the alignment to other
        """
        # Check if any transformations have been applied to the current stack
        tomo_meta = cast(Dtb, self.metadata.Tomography)
        no_shifts = np.all(tomo_meta.shifts == 0)
        no_xshift = any(
            [
                tomo_meta.xshift is None,
                tomo_meta.xshift == 0.0,
            ],
        )
        no_yshift = any(
            [
                tomo_meta.xshift is None,
                tomo_meta.xshift == 0.0,
            ],
        )
        no_rotation = any(
            [
                tomo_meta.tiltaxis is None,
                tomo_meta.tiltaxis == 0.0,
            ],
        )

        if all([no_shifts, no_xshift, no_yshift, no_rotation]):
            msg = "No transformations have been applied to this stack"
            raise ValueError(msg)

        out = align.align_to_other(self, other)

        return out

    def filter(
        self,
        method: Literal["median", "bpf", "both", "sobel"] = "median",
        size: int = 5,
        taper: float = 0.1,
    ) -> "TomoStack":
        """
        Apply one of several image filters to an entire TomoStack.

        Parameters
        ----------
        method
            Type of filter to apply. Must be ``'median'``, ``'bpf'``, ``'both'``, or
            ``'sobel'``.
        size
            Size of filtering neighborhood.
        taper
            Fraction of image size to pad to the mean.

        Returns
        -------
        filtered : TomoStack
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
            for i in range(filtered.data.shape[0]):
                dx = ndimage.sobel(filtered.data[i, :, :], 0)
                dy = ndimage.sobel(filtered.data[i, :, :], 1)
                filtered.data[i, :, :] = np.hypot(dx, dy)
        elif method == "both":
            filtered.data = ndimage.median_filter(filtered.data, size=(1, size, size))
            for i in range(filtered.data.shape[0]):
                dx = ndimage.sobel(filtered.data[i, :, :], 0)
                dy = ndimage.sobel(filtered.data[i, :, :], 1)
                filtered.data[i, :, :] = np.hypot(dx, dy)
        elif method == "bpf":
            lp_freq = 0.1
            hp_freq = 0.05
            lp_sigma = 1.5
            hp_sigma = 1.5
            [nprojs, rows, cols] = self.data.shape

            fft = np.fft.fftshift(np.fft.fft2(self.data))

            x = (np.arange(0, cols) - np.fix(cols / 2)) / cols
            y = (np.arange(0, rows) - np.fix(rows / 2)) / rows
            xx, yy = np.meshgrid(x, y)
            r = np.sqrt(xx**2 + yy**2)
            lpf = 1 / (1.0 + (r / lp_freq) ** (2 * lp_sigma))

            hpf = 1 - (1 / (1.0 + (r / hp_freq) ** (2 * hp_sigma)))
            bpf = lpf * hpf
            fft_filtered = fft * bpf

            filtered.data = np.fft.ifft2(np.fft.ifftshift(fft_filtered)).real

            h = np.hamming(rows)
            ham2d = np.sqrt(np.outer(h, h))
            filtered.data = filtered.data * ham2d
        else:
            msg = (
                f'Invalid filter method "{method}". Must be one of '
                f"{_fmt(_get_lit(self.filter, 'method'))}."
            )
            raise ValueError(msg)
        if taper:
            taper_size = np.array(np.array(taper) * self.data.shape[1:], dtype=np.int32)
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
        self,
        method: AlignmentMethodType = AlignmentMethod.PC,
        start: Optional[int] = None,
        show_progressbar: bool = False,
        crop: bool = False,
        nslices: int = 20,
        com_ref_index: Optional[int] = None,
        cl_ref_index: Optional[int] = None,
        cl_resolution: float = 0.05,
        cl_div_factor: int = 8,
    ) -> "TomoStack":
        """
        Register stack spatially.

        Options are phase correlation (PC) maximization, StackReg, center of
        mass ('COM'), or combined center of mass and common line methods.
        See docstring for :py:func:`etspy.align.align_stack` for details.

        Parameters
        ----------
        method
            Algorithm to use for registration calculation. Must be one of
            the values specified by the :py:class:`etspy.AlignmentMethod` enum.
        start
            Position in tilt series to use as starting point for the
            alignment. If ``None``, the central projection is used.
        show_progressbar
            Enable/disable progress bar
        crop
            If True, crop aligned stack to eliminate border pixels. Default is
            False.
        nslices
            (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.COM`)
            Number of slices to return. See
            :py:func:`~etspy.align.calculate_shifts_com` for more details.
        com_ref_index
            (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.COM_CL`)
            Reference slice for center of mass alignment.  All other slices
            will be aligned to this reference.  If not provided, the midpoint
            of the stack will be chosen. See :py:func:`~etspy.align.calc_shifts_com_cl`
            for more details.
        cl_ref_index
            (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.COM_CL`)
            Reference slice for common line alignment.  All other slices
            will be aligned to this reference.  If not provided, the midpoint
            of the stack will be chosen.
        cl_resolution
            (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.COM_CL`)
            Resolution for subpixel common line alignment. Default is 0.05.
            Should be less than 0.5. See
            :py:func:`~etspy.align.calc_shifts_com_cl` for more details.
        cl_div_factor
            (Only used when ``method ==``:py:attr:`~etspy.AlignmentMethod.COM_CL`)
            Factor which determines the number of iterations of common line
            alignment to perform.  Default is 8. See
            :py:func:`~etspy.align.calc_shifts_com_cl` for more details.

        Returns
        -------
        out : TomoStack
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
        if AlignmentMethod.is_valid_value(method):
            out = align.align_stack(
                self,
                method,
                start,
                show_progressbar,
                nslices=nslices,
                com_ref_index=com_ref_index,
                cl_ref_index=cl_ref_index,
                cl_resolution=cl_resolution,
                cl_div_factor=cl_div_factor,
            )
        else:
            msg = (
                f'Invalid registration method "{method}". '
                f"Must be one of {_fmt(AlignmentMethod.values())}."
            )
            raise TypeError(msg)

        if crop:
            out = align.shift_crop(out)
        return out

    def tilt_align(
        self,
        method: Literal["CoM", "MaxImage"],
        slices: Optional[np.ndarray] = None,
        nslices: Optional[int] = None,
        limit: float = 10,
        delta: float = 0.1,
        plot_results: bool = False,
        also_shift: bool = False,
        shift_limit: int = 20,
    ):
        """
        Align the tilt axis of a TomoStack.

        Uses either a center-of-mass approach or a maximum image approach

        Available methods are ``'CoM'`` and ``'MaxImage'``:

        **CoM:**
        Track the center of mass (CoM) of the projections at three
        locations.  Fit the motion of the CoM as a function of tilt to that
        expected for an ideal cylinder to calculate an X-shift at each
        location. Perform a  linear fit of the three X-shifts to calculate an
        ideal rotation.

        **MaxImage:**
        Perform automated determination of the tilt axis of a
        TomoStack by analyzing features in the projected maximum image.  A combination
        of edge detection and Hough transform analysis is used to determine the global
        rotation of the stack.  Optionally, the global shift of the tilt axis can also
        be calculated by minimization of the sum of the reconstruction.

        Parameters
        ----------
        method
            Algorithm to use for registration alignment. Must be either ``'CoM'`` or
            ``'MaxImage'``.
        slices
            (Only used when ``method == "CoM"``)
            Locations at which to perform the Center of Mass analysis. If not
            provided, an appropriate list of slices will be automatically determined.
        nslices
            (Only used when ``method == "CoM"``)
            Nubmer of slices to use for the center of mass analysis (only used if the
            ``slices`` parameter is not specified). If ``None``, a value of 10% of the
            x-axis size will be used, clamped to the range [3, 50], as calculated in
            the :py:func:`~etspy.align.tilt_com` function.
        limit
            (Only used when ``method == "MaxImage"``)
            Maximum rotation angle for MaxImage calculation
        delta
            (Only used when ``method == "MaxImage"``)
            Angular increment in degrees for MaxImage calculation
        plot_results
            (Only used when ``method == "MaxImage"``)
            If ``True``, plot the maximum image along with the lines determined
            by Hough analysis
        also_shift
            (Only used when ``method == "MaxImage"``)
            If ``True``, also calculate and apply the global shift perpendicular to the
            tilt by minimizing the sum of the reconstruction
        shift_limit
            (Only used when ``method == "MaxImage"``)
            The limit of shifts applied if ``also_shift`` is set to ``True``

        Returns
        -------
        out : TomoStack
            Copy of the input stack rotated by calculated angle

        Examples
        --------
        Align tilt axis using the center of mass (CoM) method:
            >>> import etspy.datasets as ds
            >>> import numpy as np
            >>> stack = ds.get_needle_data()
            >>> reg = stack.stack_register('PC', show_progressbar=False)
            >>> method = 'CoM'
            >>> ali = reg.tilt_align(method, slices=np.array([50,100,160]))

        Align tilt axis using the maximum image method:
            >>> import etspy.datasets as ds
            >>> stack = ds.get_needle_data()
            >>> reg = stack.stack_register('PC', show_progressbar=False)
            >>> method = 'MaxImage'
            >>> ali = reg.tilt_align(method)
        """
        if method == "CoM":
            out = align.tilt_com(self, slices, nslices)
        elif method == "MaxImage":
            out = align.tilt_maximage(
                self,
                limit,
                delta,
                plot_results,
                also_shift,
                shift_limit,
            )
        else:
            msg = (
                f'Invalid alignment method "{method}". Must be one of '
                f"{_fmt(_get_lit(self.tilt_align, 'method'))}."
            )
            raise ValueError(msg)
        return out

    def reconstruct(  # noqa: PLR0913
        self,
        method: ReconMethodType = "FBP",
        iterations: int = 5,
        constrain: bool = False,
        thresh: float = 0,
        cuda: Optional[bool] = None,
        thickness: Optional[int] = None,
        show_progressbar: bool = True,
        p: float = 0.99,
        ncores: Optional[int] = None,
        sino_filter: FbpMethodType = "shepp-logan",
        dart_iterations: Optional[int] = 5,
        gray_levels: Optional[Union[List, np.ndarray]] = None,
    ) -> "RecStack":
        """
        Reconstruct a TomoStack series using one of the available methods.

        Parameters
        ----------
        method
            Reconstruction algorithm to use.  Must be one of ``"FBP"`` (default),
            ``"SIRT"``, ``"SART"``, or ``"DART"``
        iterations
            Number of iterations for the SIRT reconstruction (used with ``SIRT``,
            ``SART``, and ``DART`` methods) (default: 5)
        constrain
            If ``True``, output reconstruction is constrained above value given
            by ``thresh``
        thresh
            Value above which to constrain the reconstructed data
        cuda
            Whether or not to use CUDA-accelerated reconstruction algorithms. If
            ``None`` (the default), the decision to use CUDA will be left to
            :py:func:`astra.astra.use_cuda`.
        thickness
            Size of the output volume (in pixels) in the projection direction. If
            ``None``, the y-size of the stack is used.
        show_progressbar
            If ``True``, show a progress bar for the reconstruction. Default is
            ``True``.
        p
            Probability for setting free pixels in DART reconstruction (only used
            if the reconstruction method is DART, default: 0.99)
        ncores
            Number of cores to use for multithreaded reconstructions.
        sino_filter
            Filter for filtered backprojection. Default is ``"shepp-logan"``.
            Available options are detailed in the Astra Toolbox documentation
            under the ``cfg.FilterType`` option of
            :external+astra:doc:`docs/algs/FBP_CUDA`.
        dart_iterations
            Number of iterations to employ for DART reconstruction
        gray_levels
            List of gray levels to use for DART reconstruction

        Returns
        -------
        rec : RecStack
            RecStack containing the reconstructed volume

        Examples
        --------
        Filtered backprojection (FBP) reconstruction:
            >>> import etspy.datasets as ds
            >>> stack = ds.get_needle_data(aligned=True)
            >>> slices = stack.isig[:, 120:121].deepcopy()
            >>> rec = slices.reconstruct('FBP', cuda=False, show_progressbar=False)

        Simultaneous iterative reconstruction technique (SIRT) reconstruction:
            >>> import etspy.datasets as ds
            >>> stack = ds.get_needle_data(aligned=True)
            >>> slices = stack.isig[:, 120:121].deepcopy()
            >>> rec = slices.reconstruct('SIRT',iterations=5,
            ...                          cuda=False, show_progressbar=False)

        SIRT reconstruction with positivity constraint:
            >>> import etspy.datasets as ds
            >>> stack = ds.get_needle_data(aligned=True)
            >>> slices = stack.isig[:, 120:121].deepcopy()
            >>> iterations = 5
            >>> constrain = True
            >>> thresh = 0
            >>> rec = slices.reconstruct('SIRT', iterations, constrain, thresh,
            ...                          cuda=False, show_progressbar=False)

        Discreate algebraice reconstruction technique (DART) reconstruction:
            >>> import etspy.datasets as ds
            >>> stack = ds.get_needle_data(aligned=True)
            >>> slices = stack.isig[:, 120:121].deepcopy()
            >>> gray_levels = [0., slices.data.max()/2, slices.data.max()]
            >>> rec = slices.reconstruct('DART', iterations=5, cuda=False,
            ...                          gray_levels=gray_levels, p=0.99,
            ...                          dart_iterations=5, show_progressbar=False)
        """
        if method.lower() not in [
            "fbp",
            "sirt",
            "sart",
            "dart",
        ]:
            msg = (
                f'Invalid reconstruction algorithm "{method}". Must be one of '
                f"{_fmt(_get_lit(self.reconstruct, 'method'))}."
            )
            raise ValueError(msg)
        if cuda is None:
            if astra.use_cuda():
                logger.info("CUDA detected with Astra")
                cuda = True
            else:
                logger.info("CUDA not detected with Astra")
                cuda = False

        if method.lower() == "dart":
            if gray_levels is None:
                msg = "gray_levels must be provided for DART"
                raise ValueError(msg)
            if not isinstance(gray_levels, (np.ndarray, list)):
                msg = f"Unknown type ({type(gray_levels)}) for gray_levels"
                raise ValueError(msg)
            if dart_iterations is None:
                logger.info("Using default number of DART iterations (5)")
                dart_iterations = 5
        else:
            dart_iterations = None
            gray_levels = None

        stack_tilts = cast(np.ndarray, cast(Dtb, self.metadata.Tomography).tilts)
        rec = recon.run(
            stack=self.data,
            tilts=stack_tilts,
            method=method,
            niterations=iterations,
            constrain=constrain,
            thresh=thresh,
            cuda=cuda,
            thickness=thickness,
            ncores=ncores,
            bp_filter=sino_filter,
            gray_levels=gray_levels,
            dart_iterations=cast(int, dart_iterations),
            p=p,
            show_progressbar=show_progressbar,
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

    def test_align(  # noqa: PLR0913
        self,
        tilt_shift: float = 0.0,
        tilt_rotation: float = 0.0,
        slices: Optional[np.ndarray] = None,
        thickness: Optional[int] = None,
        method: Literal["FBP", "SIRT", "SART"] = "FBP",
        iterations: int = 50,
        constrain: bool = True,
        cuda: Optional[bool] = None,
        thresh: float = 0,
        vmin_std: float = 0.1,
        vmax_std: float = 10,
    ):
        """
        Perform a reconstruction with limited slices for visual inspection.

        This method is useful to quickly test the alignment of the stack prior
        to a full reconstruction attempt.

        Parameters
        ----------
        tilt_shift
            Number of pixels by which to shift the stack prior to reconstruction
        tilt_rotation
            Angle by which to rotate stack prior to reconstruction
        slices
            Position of slices to use for the reconstruction.  If ``None`` (default),
            three positions at 1/4, 1/2, and 3/4 of the full size of the stack are
            automatically chosen.
        thickness
            Size of the output volume (in pixels) in the projection direction.
            If ``None`` (default), the y-size of the stack is used.
        method
            Reconstruction algorithm to use.  Must be one of ``"FBP"`` (default),
            ``"SIRT"``, or ``"SART"`` (inapplicable for the ``"DART"`` algorithm)
        iterations
            Number of iterations for the SIRT reconstruction (used with ``SIRT``,
            and ``SART`` methods) (default: 50)
        cuda
            Whether or not to use CUDA-accelerated reconstruction algorithms. If
            ``None`` (the default), the decision to use CUDA will be left to
            :py:func:`astra.astra.use_cuda`.
        thresh
            Value above which to constrain the reconstructed data
        vmin_std
            Number of standard deviations from mean (lower bound) to use for scaling the
            displayed slices
        vmax_std
            Number of standard deviations from mean (upper bound) to use for scaling the
            displayed slices
        """
        if slices is None:
            mid = np.array(self.data.shape[2] / 2, dtype=np.int32)
            slices = np.array([mid / 2, mid, mid + mid / 2], dtype=np.int32)

        if (tilt_shift != 0.0) or (tilt_rotation != 0.0):
            shifted = self.trans_stack(xshift=0, yshift=tilt_shift, angle=tilt_rotation)
        else:
            shifted = self.deepcopy()
        shifted.data = shifted.data[:, :, slices]

        cast(Uda, shifted.axes_manager[0]).axis = cast(Uda, self.axes_manager[0]).axis
        if cuda is None:
            if astra.use_cuda():
                logger.info("CUDA detected with Astra")
                cuda = True
            else:
                cuda = False
                logger.info("CUDA not detected with Astra")
        rec = cast(TomoStack, shifted).reconstruct(
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
        ax1.set_title(f"Slice {slices[0]}")
        ax1.set_axis_off()

        ax2.imshow(rec.data[1, :, :], cmap="afmhot", vmin=minvals[1], vmax=maxvals[1])
        ax2.set_title(f"Slice {slices[1]}")
        ax2.set_axis_off()

        ax3.imshow(rec.data[2, :, :], cmap="afmhot", vmin=minvals[2], vmax=maxvals[2])
        ax3.set_title(f"Slice {slices[2]}")
        ax3.set_axis_off()
        fig.tight_layout()

    def set_tilts(self, start: float, increment: float):
        """
        Calibrate the tilt axis of the image stack.

        Parameters
        ----------
        start
            Tilt angle of first image in stack

        increment
            Tilt increment between images
        """
        nimages = self.data.shape[0]
        ax = cast(Uda, self.axes_manager[0])
        ax.name = "Tilt"
        ax.units = "degrees"
        ax.scale = increment
        ax.offset = start
        tilts = np.arange(start, nimages * increment + start, increment)

        if not self.metadata.has_item("Tomography"):
            self.metadata.add_node("Tomography")
            tomo_meta = cast(Dtb, self.metadata.Tomography)
            tomo_meta.set_item("tilts", tilts)
            tomo_meta.set_item("tiltaxis", 0)
            tomo_meta.set_item("xshift", 0)
            tomo_meta.set_item("yshift", 0)
            tomo_meta.set_item("shifts", None)
            tomo_meta.set_item("cropped", value=False)
        else:
            cast(Dtb, self.metadata.Tomography).set_item("tilts", tilts)

    def manual_align(  # noqa: PLR0915
        self,
        nslice: int,
        xshift: int = 0,
        yshift: int = 0,
        display: bool = False,
    ) -> "TomoStack":
        """
        Manually shift part of a stack with respect to another and return it as a copy.

        Parameters
        ----------
        nslice
            Slice position at which to implement shift
        xshift
            Number of pixels with which to shift the second portion of the
            stack relative to the first in the X dimension.
        yshift
            Number of pixels with which to shift the second portion of the
            stack relative to the first in the Y dimension.
        display
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
        elif (xshift > 0) and (yshift > 0):
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
        nslice: Optional[int] = None,
        algorithm: Literal["SIRT", "SART"] = "SIRT",
        iterations: int = 50,
        constrain: bool = True,
        cuda: Optional[bool] = None,
        thresh: float = 0,
    ) -> Tuple[Signal2D, Signal1D]:
        """
        Determine the optimum number of iterations for reconstruction.

        Evaluates the difference between reconstruction and input data
        at each iteration and terminates when the change between iterations is
        below tolerance.

        Parameters
        ----------
        nslice
            Slice location at which to perform the evaluation.
        algorithm
            Reconstruction algorithm use.  Must be either ``'SIRT'`` (default)
            or ``'SART'`` (this method is inapplicable for ``'FBP'`` and ``'DART'``).
        constrain
            If True, perform SIRT reconstruction with a non-negativity
            constraint.  Default is ``True``
        cuda
            Whether or not to use CUDA-accelerated reconstruction algorithms. If
            ``None`` (the default), the decision to use CUDA will be left to
            :py:func:`astra.astra.use_cuda`.
        thresh
            Value above which to constrain the reconstructed data

        Returns
        -------
        rec_stack : :py:class:`~hyperspy.api.signals.Signal2D`
            Signal containing the SIRT reconstruction at each iteration
            for visual inspection.
        error : :py:class:`~hyperspy.api.signals.Signal1D`
            Sum of squared difference between the forward-projected
            reconstruction and the input sinogram at each iteration

        Examples
        --------
            >>> import etspy.datasets as ds
            >>> stack = ds.get_needle_data(aligned=True)
            >>> rec_stack, error = stack.recon_error(iterations=5)
        """
        if cast(Dtb, self.metadata.Tomography).tilts is None:
            msg = "Tilt angles not defined"
            raise ValueError(msg)

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
        angles = cast(np.ndarray, cast(Dtb, self.metadata.Tomography).tilts)
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
        rec_ax0, rec_ax1, rec_ax2 = (
            cast(Uda, rec_stack.axes_manager[i]) for i in range(3)
        )
        self_ax2 = cast(Uda, self.axes_manager[2])
        rec_ax0.name = algorithm.upper() + " iteration"
        rec_ax0.scale = 1
        rec_ax1.name = self_ax2.name
        rec_ax1.scale = self_ax2.scale
        rec_ax1.units = self_ax2.units
        rec_ax2.name = "z"
        rec_ax2.scale = self_ax2.scale
        rec_ax2.units = self_ax2.units
        rec_stack.navigator = "slider"

        error = Signal1D(error)
        cast(Uda, error.axes_manager[0]).name = algorithm.upper() + " Iteration"
        cast(Dtb, error.metadata.Signal).quantity = "Sum of Squared Difference"
        return rec_stack, error


class RecStack(CommonStack):
    """
    Create a RecStack instance, used to hold the results of a reconstructed volume.

    All arguments are passed to the :py:class:`~hyperspy.api.signals.Signal2D`
    constructor and should be used as documented for that method.

    Group
    -----
    signals

    Order
    -----
    2
    """

    def __init__(self, *args, **kwargs):
        """
        Create a RecStack signal.

        Parameters
        ----------
        args
            Additional non-keyword arguments passed to
            :py:class:`~hyperspy.api.signals.Signal2D`
        kwargs
            Additional keyword arguments passed to
            :py:class:`~hyperspy.api.signals.Signal2D`
        """
        super().__init__(*args, **kwargs)

    def plot_slices(
        self,
        xslice: Optional[int] = None,
        yslice: Optional[int] = None,
        zslice: Optional[int] = None,
        vmin_std: float = 0.1,
        vmax_std: float = 5,
    ):
        """
        Plot slices along all three axes of a reconstruction stack.

        Parameters
        ----------
        xslice, yslice, zslice
            Indices of slices to plot. If ``None`` (default), the middle
            most slice will be used.

        vmin_std, vmax_std
            Number of standard deviations from mean to use for
            scaling the displayed slices

        Returns
        -------
        fig : ~matplotlib.figure.Figure
            The figure containing a view of the three slices
        """
        if xslice is None:
            xslice = self.data.shape[0] // 2
        if yslice is None:
            yslice = self.data.shape[1] // 2
        if zslice is None:
            zslice = self.data.shape[2] // 2

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
        ax1.set_title(f"Z-Y Slice {xslice}")
        ax1.set_ylabel("Z")
        ax1.set_xlabel("Y")

        ax2.imshow(slices[1], cmap="afmhot", vmin=minvals[1], vmax=maxvals[1])
        ax2.set_title(f"Y-X Slice {zslice}")
        ax2.set_ylabel("Y")
        ax2.set_xlabel("X")

        ax3.imshow(slices[2].T, cmap="afmhot", vmin=minvals[2], vmax=maxvals[2])
        ax3.set_title(f"Z-X Slice {yslice}")
        ax3.set_ylabel("Z")
        ax3.set_xlabel("X")
        fig.tight_layout()

        [i.set_xticks([]) for i in [ax1, ax2, ax3]]
        [i.set_yticks([]) for i in [ax1, ax2, ax3]]
        return fig
