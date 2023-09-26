TomoTools package
===========

TomoTools is a Hyperspy-based software package for the aligment and reconstruction
of electron tomography data from TEM/STEM instruments. Tools are provided for basic 
tilt series data processing, stack alignment, and reconstruction using the astra-toolbox.


Installation
------------

  Anaconda (Preferred):
  ---------------------
  * The required packages can be installed into an existing Anaconda environmnent.
    Install major dependencies Astra Toolbox, Hyperspy, and Hyperspy widgets.
    Install OpenCV through pip.  Currently, the conda repo for OpenCV has conflicts
    with Hyperspy and/or Astra.
    NOTE: `astra-toolbox` must be installed first due to a dependency conflict.
    ```bash
    conda create -n tomo -c astra-toolbox astra-toolbox
    conda install -c conda-forge hyperspy hyperspy-gui-ipywidgets ipympl
    pip install opencv-python

    ```

  * Install the TomoTools package from GitHub:
    ```bash
    pip install git+https://github.com/usnistgov/tomotools.git
    ```

Removal
-------
The package can be removed with:

```bash
pip uninstall tomotools
```


Usage
-----
In python or ipython:

```python
import tomotools.api as tomotools
stack = tomotools.load('TiltSeries.mrc')
```

Documentation is very limited at this point


Documentation
-------------
Release: https://github.com/andrewherzing/tomotools

Further documentation, notebooks and examples will be made available over time.


Related projects
----------------
http://hyperspy.org/

https://opencv.org/

https://www.astra-toolbox.com/
