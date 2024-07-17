ETSpy package
===========

ETSpy is a Hyperspy extension package package for the processing, aligment, and reconstruction
of electron tomography data from TEM/STEM instruments. Tools are provided for basic 
tilt series data processing, stack alignment, and reconstruction using the astra-toolbox.


Installation
------------

  Anaconda (Preferred):
  ---------------------
  * Install major dependencies Astra Toolbox and HyperSpy.
    
    ```bash
    conda create -n tomo
    conda activate tomo
    conda install -c astra-toolbox astra-toolbox 
    conda install -c conda-forge hyperspy-base hyperspy-gui-ipywidgets 
    ```

  * Install the ETSpy package from GitHub:
    ```bash
    pip install git+https://github.com/usnistgov/etspy.git
    ```

  Optional (higly recommended):
  ---------------------
  * Install `ipympl` and `ipykernel` to use `etspy` with Jupyter.
    * `ipympl` enables interactive plotting in Jupyter Lab or Notebook.  
    * `ipykernel` allows use of the the tomoools kernel with Jupyter installed in a different environment. 

    ```bash
    conda install ipympl ipykernel
    ```
  
Removal
-------
The package can be removed with:

```bash
pip uninstall etspy
```


Usage
-----
In python or ipython:

```python
import etspy.api as etspy
stack = etspy.load('TiltSeries.mrc')
```

Documentation is very limited at this point


Documentation
-------------
Release: https://github.com/usnistgov/etspy

A demo notebook is available in the resources folder.  More documentation
will be made available over time.


Related projects
----------------
http://hyperspy.org/

https://www.astra-toolbox.com/
