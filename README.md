ETSpy package
===========

> ⚠️ ETSpy is still in a pre-release status, and the API may change with little warning!

ETSpy is a [HyperSpy](https://hyperspy.org) extension package package for the processing, aligment, and reconstruction
of electron etspygraphy data from TEM/STEM instruments. Tools are provided for basic 
tilt series data processing, stack alignment, and reconstruction using the
[ASTRA Toolbox](https://astra-toolbox.com/).


Installation
------------

  Anaconda (Preferred):
  ---------------------
  * Install major dependencies Astra Toolbox and HyperSpy.
    
    ```bash
    conda create -n etspy
    conda activate etspy
    conda install -c astra-toolbox astra-toolbox 
    conda install -c conda-forge hyperspy-base hyperspy-gui-ipywidgets 
    ```

  * Install the ETSpy package from GitHub:
    ```bash
    pip install etspy
    ```

  Optional Jupyter components (higly recommended):
  ------------------------------------------------
  * Install `ipympl` and `ipykernel` to use `etspy` with Jupyter.
    * `ipympl` enables interactive plotting in Jupyter Lab or Notebook.  
    * `ipykernel` allows use of the the etspy kernel with Jupyter installed in a different environment. 

    ```bash
    conda install ipympl ipykernel
    ```

  * To "register" the python kernel associated with the `etspy` conda environment, run
    the following after ensuring that environment is activated (with `conda activate etspy`):

    ```bash
    python -m ipykernel install --user --name ETSpy
    ```

    You will then be able to select the "ETSpy" kernel when running Jupyter and creating new
    notebooks

  Using pip (may be difficult, depending on your system)
  ------------------------------------------------------

  Assuming you have the prequisite packages on your system (including CUDA libraries), ETSpy
  should be able to be installed with a simple:

  * ```bash
    pip install etspy
    ```

  Some dependencies of ETSpy require compilation of C code, so often on Windows, the Anaconda
  approach will simplify things greatly.

  
Removal
-------
The package can be removed with:

```bash
pip uninstall etspy
```


Basic Usage
-----------
In python or ipython:

```python
import etspy.api as etspy
stack = etspy.load('TiltSeries.mrc')
```

For more details, see the [documentation](https://pages.nist.gov/etspy)


Documentation
-------------
Release: [https://github.com/usnistgov/etspy](https://github.com/usnistgov/etspy)

A demo notebook is available in the resources folder.  More documentation
will be made available over time.


Related projects
----------------

- [http://hyperspy.org/](http://hyperspy.org/)
- [https://www.astra-toolbox.com/](https://www.astra-toolbox.com/)
