"""
Visualization module for ETSpy package.

Contains the TomoStack class and its methods.
"""

from typing import TYPE_CHECKING

import hyperspy.api as hs
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

if TYPE_CHECKING:
    from etspy.base import TomoStack


class VolumeSlicer:
    """Class for interactive slicing of a volume."""

    def __init__(
        self,
        stack: "TomoStack",
        vmin_std: float = 0.1,
        vmax_std: float = 10,
        figsize: tuple = (10, 4),
    ):
        """Initialize the VolumeSlicer Class.

        Parameters
        ----------
        stack : RecStack
            Volume to rotate
        vmin_std : float
            Number of standard deviations from mean (lower bound) to use for scaling the
            displayed slices
        vmax_std : float
            Number of standard deviations from mean (upper bound) to use for scaling the
            displayed slices
        figsize : tuple
            Size of matplotlib figure to use
        """
        if type(vmin_std) in [float, int]:
            vmin_std = vmin_std * np.ones(3)
        if type(vmax_std) in [float, int]:
            vmax_std = vmax_std * np.ones(3)

        nx, nz, ny = stack.data.shape
        self.stackZY = stack.deepcopy()
        self.minvalsZY = self.stackZY.data.mean((1, 2)) - vmin_std[
            0
        ] * self.stackZY.data.std((1, 2))
        self.maxvalsZY = self.stackZY.data.mean((1, 2)) + vmax_std[
            0
        ] * self.stackZY.data.std((1, 2))

        self.stackXZ = stack.swap_axes(0, 1)
        self.minvalsXZ = self.stackXZ.data.mean((1, 2)) - vmin_std[
            1
        ] * self.stackXZ.data.std((1, 2))
        self.maxvalsXZ = self.stackXZ.data.mean((1, 2)) + vmax_std[
            1
        ] * self.stackXZ.data.std((1, 2))

        self.stackXY = stack.swap_axes(0, 2)
        self.minvalsXY = self.stackXY.data.mean((1, 2)) - vmin_std[
            2
        ] * self.stackXY.data.std((1, 2))
        self.maxvalsXY = self.stackXY.data.mean((1, 2)) + vmax_std[
            2
        ] * self.stackXY.data.std((1, 2))

        self.figsize = figsize
        self.output = widgets.Output()
        self.sliderX = widgets.IntSlider(
            value=0,
            min=0,
            max=nx - 1,
            step=1,
            description="X",
        )
        self.sliderY = widgets.IntSlider(
            value=0,
            min=0,
            max=ny - 1,
            step=1,
            description="Y",
        )
        self.sliderZ = widgets.IntSlider(
            value=0,
            min=0,
            max=nz - 1,
            step=1,
            description="Z",
        )
        self.create_plot()
        self.link_widgets()

    def create_plot(self):
        """Create display for interactive rotation."""
        with self.output:
            fig = plt.figure(figsize=self.figsize)
            self.axes = hs.plot.plot_images(
                [self.stackZY.inav[0], self.stackXZ.inav[0], self.stackXY.inav[0]],
                cmap="inferno",
                colorbar=None,
                scalebar=[
                    0,
                ],
                per_row=3,
                fig=fig,
                tight_layout=True,
            )

    def update_plot(self, change):
        """Update the plot based on the changed slider value.

        Retrieves the new slider value, updates the corresponding image in the plot,
        and redraws the canvas.

        Args:
            change: A dictionary containing information about the change, including the
            new slider value and the slider object that triggered the change.
        """
        with self.output:
            if change["owner"] == self.sliderX:
                self.axes[0].images[0].set_data(self.stackZY.inav[change["new"]])
                self.axes[0].images[0].set_clim(
                    self.minvalsZY[change["new"]],
                    self.maxvalsZY[change["new"]],
                )
            if change["owner"] == self.sliderY:
                self.axes[1].images[0].set_data(self.stackXZ.inav[change["new"]])
                self.axes[1].images[0].set_clim(
                    self.minvalsXZ[change["new"]],
                    self.maxvalsXZ[change["new"]],
                )
            if change["owner"] == self.sliderZ:
                self.axes[2].images[0].set_data(self.stackXY.inav[change["new"]])
                self.axes[2].images[0].set_clim(
                    self.minvalsXY[change["new"]],
                    self.maxvalsXY[change["new"]],
                )

    def link_widgets(self):
        """Links the slider widgets and button to their respective event handlers.

        Observes changes to the slider widgets and calls the update_plot method when a
        change occurs. Also links the button to the on_button_click method.
        """
        self.sliderX.observe(self.update_plot, names="value")
        self.sliderY.observe(self.update_plot, names="value")
        self.sliderZ.observe(self.update_plot, names="value")

    def display(self):
        """Display the interactive plot and widgets.

        Creates a VBox containing an HBox with the slider widgets, the output widget,
        and the button, and displays it.
        """
        box = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.sliderX,
                    ],
                ),
                widgets.HBox(
                    [
                        self.sliderY,
                    ],
                ),
                widgets.HBox(
                    [
                        self.sliderZ,
                    ],
                ),
                self.output,
            ],
        )
        display(box)
