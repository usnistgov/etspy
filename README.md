
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
* Install with YML file

  Download the tomo.yml file located in the 'resources' directory and install via conda using:

  ```bash
  conda env create -f tomo.yml
  ```
* The package can also be installed into an existing environmnent without the YML file 
  Install major dependencies Hyperspy, OpenCV, and astra-toolbox either through conda using:
  ```bash
  conda install -c conda-forge hyperspy
  conda install -c conda-forge opencv
  conda install -c astra-toolbox astra-toolbox
  
  pip install git+https://github.com/andrewherzing/tomotools.git
  ```

pip:
* Install major dependencies Hyperspy, OpenCV, and astra-toolbox either through conda using:

* See installation instructions for astra-toolbox at link below.  Then install remaining packages:

  ```bash
  pip install hyperspy
  pip install opencv-python
  pip install git+https://github.com/andrewherzing/tomotools.git
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
import tomotools
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
