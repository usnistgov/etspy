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
    Install major dependencies Astra Toolbox and HyperSpy. Optional but highly recommended,
    ipympl should also be installed to enable interactive plotting in Jupyter Lab.
    
    # NOTE: `astra-toolbox` must be installed first due to a dependency conflict.
    ```bash
    conda create -n tomo
    conda activate tomo
    conda install -c astra-toolbox astra-toolbox 
    conda install -c conda-forge hyperspy 
    conda install -c conda-forge ipympl
    conda update --all
    ```

  * Install the TomoTools package from GitHub:
    ```bash
    pip install git+https://github.com/usnistgov/tomotools.git
    ```

  * OPTIONAL: Install GENFIRE and PyFFTW:
    # NOTE: Use of GENFIRE requires a headless implementation currently located on GitHub.
    ```bash
    pip install git+https://github.com/AndrewHerzing/GENFIRE-Python@Headless
    conda install -c conda-forge pyfftw
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

A demo notebook is available in the resources folder.  More documentation
will be made available over time.


Related projects
----------------
http://hyperspy.org/

https://www.astra-toolbox.com/
