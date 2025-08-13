"""Tranformation module for ETSpy package."""

import logging

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from scipy import ndimage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ImageRotator:
    """_Class for interactive rotation of a volume."""

    def __init__(self, stack, slices=None):
        """Initialize the ImageRotator Class.

        Parameters
        ----------
        stack : RecStack
            Volume to rotate
        slices : list, optional
            Locations along X, Y, and Z to use for visualizing rotation. If None, the
            central slice along each dimension is used.
        """
        if slices is None:
            slices = np.array(stack.data.shape) // 2
        self.img1 = stack.data[slices[0], :, :]
        self.img2 = stack.data[:, slices[1], :]
        self.img3 = stack.data[:, :, slices[2]]
        self.bckg1 = self.img1[0:5, 0:5].mean()
        self.bckg2 = self.img2[0:5, 0:5].mean()
        self.bckg3 = self.img3[0:5, 0:5].mean()
        self.angle1 = 0
        self.angle2 = 0
        self.angle3 = 0
        self.output = widgets.Output()
        self.slider1 = widgets.IntSlider(
            value=0,
            min=-30,
            max=30,
            step=0.5,
            description="Theta:",
        )
        self.slider2 = widgets.IntSlider(
            value=0,
            min=-30,
            max=30,
            step=0.5,
            description="Phi:",
        )
        self.slider3 = widgets.IntSlider(
            value=0,
            min=-30,
            max=30,
            step=0.5,
            description="Psi:",
        )
        self.button = widgets.Button(description="Get Slider Values")
        self.slider_values = {"angle1": None, "angle2": None, "angle3": None}
        self.create_plot()
        self.link_widgets()

    def create_plot(self):
        """Create display for interactive rotation."""
        with self.output:
            self.fig, axs = plt.subplots(1, 3, figsize=(12, 6))
            self.im1 = axs[0].imshow(
                ndimage.rotate(self.img1, self.angle1, reshape=False, cval=self.bckg1),
                cmap="inferno",
            )
            axs[0].axis("off")
            self.im2 = axs[1].imshow(
                ndimage.rotate(self.img2, self.angle2, reshape=False, cval=self.bckg2),
                cmap="inferno",
            )
            axs[1].axis("off")
            self.im3 = axs[2].imshow(
                ndimage.rotate(self.img3, self.angle3, reshape=False, cval=self.bckg3),
                cmap="inferno",
            )
            axs[2].axis("off")
            plt.show()

    def update_plot(self, change):
        """Update the plot based on the changed slider value.

        Retrieves the new slider value, updates the corresponding image in the plot,
        and redraws the canvas.

        Args:
            change: A dictionary containing information about the change, including the
            new slider value and the slider object that triggered the change.
        """
        with self.output:
            if change["owner"] == self.slider1:
                self.im1.set_data(
                    ndimage.rotate(
                        self.img1,
                        change["new"],
                        reshape=False,
                        cval=self.bckg1,
                    ),
                )
            elif change["owner"] == self.slider2:
                self.im2.set_data(
                    ndimage.rotate(
                        self.img2,
                        change["new"],
                        reshape=False,
                        cval=self.bckg2,
                    ),
                )
            elif change["owner"] == self.slider3:
                self.im3.set_data(
                    ndimage.rotate(
                        self.img3,
                        change["new"],
                        reshape=False,
                        cval=self.bckg3,
                    ),
                )
            self.fig.canvas.draw_idle()

    def on_button_click(self):
        """Handle the button click event.

        Retrieves the current slider values, prints them, closes the plot, and clears
        the output widget. Also disables the slider widgets and the button to prevent
        further interactions.
        """
        self.slider_values["angle1"] = self.slider1.value
        self.slider_values["angle2"] = self.slider2.value
        self.slider_values["angle3"] = self.slider3.value
        plt.close(self.fig)

        self.slider1.disabled = True
        self.slider2.disabled = True
        self.button.disabled = True
        logger.info("Rotation angles retrieved from ImageRotator")

    def link_widgets(self):
        """Links the slider widgets and button to their respective event handlers.

        Observes changes to the slider widgets and calls the update_plot method when a
        change occurs. Also links the button to the on_button_click method.
        """
        self.slider1.observe(self.update_plot, names="value")
        self.slider2.observe(self.update_plot, names="value")
        self.slider3.observe(self.update_plot, names="value")
        self.button.on_click(self.on_button_click)

    def display(self):
        """Display the interactive plot and widgets.

        Creates a VBox containing an HBox with the slider widgets, the output widget,
        and the button, and displays it.
        """
        box = widgets.VBox(
            [
                widgets.HBox([self.slider1, self.slider2, self.slider3]),
                self.output,
                self.button,
            ],
        )
        display(box)
