# General Information

The recommended way to use the ETSpy library is by importing the
`api` module:

```
import etspy.api as etspy
```

Running this command is approximately equivalent to running:

```
from etspy import align, io, utils  # (1)!
from etspy.base import TomoStack
from etspy.io import load
```

```{eval-rst}
.. code-annotations::
    1. .. note::

          If you wish to import things directly from ``etspy``, this
          approach is perfectly valid. The ``etspy.api`` is just a helper
          module to expose a more limited set of commonly used features.
```

The `etspy.api` module exposes the most commonly used features of ETSpy in a
convenient interface. In particular, it provides access tothe [`align`](alignment),
[`io`](io), and [`utils`](utilities) sub-modules. It also directly exposes
the [`TomoStack`](#etspy.base.TomoStack) constructor and the [`load`](#etspy.io.load) method,
which allow for creation of [`TomoStack`](#etspy.base.TomoStack) objects either directly
from NumPy arrays, or by loading data from `.mrc`, `.dm3`, or `.dm4` files.

:::{tip}
For more examples, please consult the example notebook on the 
[Example Usage](examples/etspy_demo) page.
:::

(signals)=

# Signals

ETSpy provides two dedicated signal types that are extensions of the regular
[HyperSpy `Signal2D`](inv:hyperspy#hyperspy.api.signals.Signal2D) class.
[`TomoStack`](#etspy.base.TomoStack) objects provide the majority of the
functionality of ETSpy, and represent a tomographic tilt series data set.
[`RecStack`](#etspy.base.RecStack) objects provide a (currently limited)
set of additional operations useful for visualizing and processing the
results of a tomography reconstruction. For more details on these classes,
please consult the following pages: 

```{eval-rst}
.. python-apigen-group:: signals
```

(io)=

# File I/O

Experimental data files can be easily loaded into ETSpy using the
[`load`](#etspy.io.load) function. If your data is already held
in memory as a NumPy array, a [`TomoStack`](#etspy.base.TomoStack)
signal can also be created directly using [`TomoStack()`](#etspy.base.TomoStack)
constructor. Additionally, the [`io`](io) module provides a number of
other helper functions:

```{eval-rst}
.. python-apigen-group:: io

```

(alignment)=

# Alignment Methods

The [`align`](alignment) module provides a range of tools to correct
for misalignments of tilt image series that are common during an experimental
data collection:

```{eval-rst}
.. python-apigen-group:: align
```

(reconstruction)=

# Reconstruction

The [`recon`](reconstruction) module contains a number of lower-level helper
methods that run directly on NumPy arrays. These functions make extensive use
of the [ASTRA Toolbox](https://astra-toolbox.com) and are primarily designed to
be used internally by other parts of the code, but can be used directly, if desired.

:::{tip}
For an easier way to reconstruct a {py:class}`~etspy.base.TomoStack` signal, use the
{py:meth}`etspy.base.TomoStack.reconstruct` function instead.
:::

```{eval-rst}
.. python-apigen-group:: recon
```

(simulation)=

# Data Simulation

The [`simulation`](simulation) module provides tools to create model data, which can
be used to test various reconstruction routines:

```{eval-rst}
.. python-apigen-group:: simulation
```

(datasets)=

# Example Datasets

The [`datasets`](datasets) module provides two experimental datasets distributed
along with the ETSpy code that can be used in examples or for testing reconstruction
and alignment routines:

```{eval-rst}
.. python-apigen-group:: datasets
```

(utilities)=

# Utilities

The [`utils`](utilities) module provides a few miscellaneous functions that perform
assorted tasks related to tomographic reconstruction:

```{eval-rst}
.. python-apigen-group:: utilities
```
