TomoTools package
===========

The tomotools package provides code for the aligment and reconstruction
of electron tomography data from TEM/STEM instruments. 

The package mostly wraps existing libraries to perform many of the lower level
operations.  Hyperspy is employed for data input/output, the astra-toolbox performs
the 3D reconstructions, and rigid transformation for alignments are calculated
and applied using the OpenCV library.


Installation
------------

  Anaconda (Preferred):
  ---------------------
  * The required packages can be installed into an existing Anaconda environmnent.
    Install major dependencies Astra Toolbox, Hyperspy, OpenCV, and TomoPy 
    through conda. NOTE: `astra-toolbox` must be installed first due to a
    dependency conflict.
    ```bash
    conda install -c astra-toolbox astra-toolbox
    conda install -c conda-forge hyperspy opencv tomopy

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

https://tomopy.readthedocs.io/en/latest/
