"""Tranformation module for ETSpy package."""

import logging

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from scipy import ndimage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VolumeRotator:
    """_Class for interactive rotation of a volume."""

    def __init__(self, stack, slices=None, figsize=(10, 4)):
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
        self.xslice = stack.data[slices[0], :, :]
        self.zslice = stack.data[:, slices[1], :].T
        self.yslice = stack.data[:, :, slices[2]]
        self.x_bckg = self.xslice[0:5, 0:5].mean()
        self.z_bckg = self.zslice[0:5, 0:5].mean()
        self.y_bckg = self.yslice[0:5, 0:5].mean()
        self.figsize = figsize
        self.angle1 = 0
        self.angle2 = 0
        self.angle3 = 0
        self.output = widgets.Output()
        self.slider1 = widgets.FloatSlider(
            value=0,
            min=-30,
            max=30,
            step=0.5,
            description="Theta:",
        )
        self.slider2 = widgets.FloatSlider(
            value=0,
            min=-30,
            max=30,
            step=0.5,
            description="Psi:",
        )
        self.slider3 = widgets.FloatSlider(
            value=0,
            min=-30,
            max=30,
            step=0.5,
            description="Phi:",
        )
        self.button = widgets.Button(description="Get Slider Values")
        self.slider_values = np.zeros(3)
        self.create_plot()
        self.link_widgets()

    def create_plot(self):
        """Create display for interactive rotation."""
        with self.output:
            self.fig, axs = plt.subplots(1, 3, figsize=self.figsize)
            self.im1 = axs[0].imshow(
                ndimage.rotate(
                    self.xslice,
                    -self.angle1,
                    reshape=False,
                    cval=self.x_bckg,
                ),
                cmap="inferno",
            )
            axs[0].tick_params(axis="x", labelbottom=False)
            axs[0].tick_params(axis="y", labelleft=False)

            axs[0].set_xlabel("y")
            axs[0].set_ylabel("z")
            self.im2 = axs[1].imshow(
                ndimage.rotate(
                    self.zslice,
                    -self.angle2,
                    reshape=False,
                    cval=self.z_bckg,
                ),
                cmap="inferno",
            )

            axs[1].set_xlabel("x")
            axs[1].set_ylabel("y")
            axs[1].tick_params(axis="x", labelbottom=False)
            axs[1].tick_params(axis="y", labelleft=False)

            self.im3 = axs[2].imshow(
                ndimage.rotate(
                    self.yslice,
                    -self.angle3,
                    reshape=False,
                    cval=self.y_bckg,
                ),
                cmap="inferno",
            )
            axs[2].set_xlabel("z")
            axs[2].set_ylabel("x")
            axs[2].tick_params(axis="x", labelbottom=False)
            axs[2].tick_params(axis="y", labelleft=False)

            self.fig.tight_layout()
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
                        self.xslice,
                        -change["new"],
                        reshape=False,
                        cval=self.x_bckg,
                    ),
                )
            elif change["owner"] == self.slider2:
                self.im2.set_data(
                    ndimage.rotate(
                        self.zslice,
                        -change["new"],
                        reshape=False,
                        cval=self.z_bckg,
                    ),
                )
            elif change["owner"] == self.slider3:
                self.im3.set_data(
                    ndimage.rotate(
                        self.yslice,
                        -change["new"],
                        reshape=False,
                        cval=self.y_bckg,
                    ),
                )
            self.fig.canvas.draw_idle()

    def on_button_click(self, b):  # noqa: ARG002
        """Handle the button click event.

        Retrieves the current slider values, prints them, closes the plot, and clears
        the output widget. Also disables the slider widgets and the button to prevent
        further interactions.
        """
        self.slider_values[0] = self.slider1.value
        self.slider_values[1] = self.slider2.value
        self.slider_values[2] = self.slider3.value
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


def calculate_rotation(rotation_angles, volshape):
    """Summary."""
    theta, psi, phi = np.deg2rad(rotation_angles)

    rotation_theta = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
    )

    rotation_psi = np.array(
        [[np.cos(psi), 0, np.sin(psi)], [0, 1, 0], [-np.sin(psi), 0, np.cos(psi)]],
    )

    rotation_phi = np.array(
        [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]],
    )

    rotation_matrix = rotation_phi @ rotation_psi @ rotation_theta

    center = 0.5 * (np.array(volshape) - 1)

    offset = center - np.dot(rotation_matrix, center)
    return rotation_matrix, offset
