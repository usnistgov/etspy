# ETSpy

[![Documentation link](https://img.shields.io/badge/Documentation-blue?logo=readthedocs&logoColor=white&labelColor=gray&style=flat-square)](https://pages.nist.gov/etspy)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/etspy?label=pypi%2Fetspy&style=flat-square)](https://pypi.org/project/etspy/)
[![Conda versions](https://anaconda.org/conda-forge/etspy/badges/version.svg)](https://anaconda.org/conda-forge/etspy)

ETSpy is a [HyperSpy](https://hyperspy.org) extension package package for the processing, aligment, and reconstruction
of electron tomography data from TEM/STEM instruments. Tools are provided for basic 
tilt series data processing, stack alignment, and reconstruction using the
[ASTRA Toolbox](https://astra-toolbox.com/).

## Installation

Depending on your system, there are a few ways to install ETSpy. Due to 
dependencies that require compilation of binaries and the use of GPU-accelerated
libraries, [conda](https://anaconda.org/anaconda/conda) is the simplest way to
get started. It will auto-detect CUDA-capable GPUs and install the correct version
of whatever packages are required.

> ⚠️ ETSpy requires a Python version `>= 3.10` and `< 3.13` (3.13 and above are not supported due to dependencies).
> If installing manually using `pip`, please ensure you are using a supported version.

### Anaconda (Preferred)

  *Works on Windows, MacOS, and Linux*

  * First, ensure you have either [Anaconda](https://www.anaconda.com/download/success)
    or [Miniconda](https://docs.anaconda.com/miniconda/) installed on your system.

  * Run the following command to create a new environment then activate the newly created
    environment:
    
    ```shell
    # if you would like your environment to be stored in a specific place, use the "-p <path>" option
    $ conda create -n etspy
    $ conda activate etspy
    ```

  * With the `etspy` environment activated, install the ETSpy package from the `conda-forge` repo:

    ```shell
    (etspy) $ conda install -c conda-forge etspy
    ``` 

  * (Alternatively) if you have a GPU and wish to make full use of the GPU-accelerated code, install
    the `etspy-gpu` package, which will pull in a few added dependencies to enable these features:

    ```shell
    (etspy) $ conda install -c conda-forge etspy-gpu
    ```

####  Optional Jupyter components (higly recommended)

  * To use ETSpy from within a Jupyter Lab/Notebook environment, you will need to
    "register" the python kernel associated with the `etspy` conda environment. Run
    the following after ensuring that environment is activated (with `conda activate etspy`):

    ```shell
    (etspy) $ conda install -c conda-forge ipykernel
    (etspy) $ python -m ipykernel install --user --name ETSpy
    ```

    You will then be able to select the "ETSpy" kernel when running Jupyter and creating new
    notebooks

###  Using pip

  *Works on Linux only, with additional prerequisites*

  Assuming you have the prequisite packages on your system (including
  the [CUDA libraries](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)),
  ETSpy should be able to be installed with a simple `pip` command (it is recommended to install
  ETSpy in a dedicated virtual environment). Pick one of the following options depending on your needs:

  ```{tip}
  On Ubuntu-based systems, the NVIDIA/CUDA dependencies installed via the system-provided `nvidia-cuda-toolkit` apt package may be out of date and incompatible with the ASTRA toolkit. We recommend installing the version directly from NVIDIA.
  ```

  * ```shell
    $ pip install etspy
    ```

  * To use ETSpy in Jupyter interface from within a dedicated virtual environment, installing
    `ipykernel` is necessary (as with Anaconda). This can be done by specifying
    the `[jupyter]` group when installing ETSpy:

    ```shell
    $ pip install etspy[jupyter]
    ```

  * To use the `cupy` accelerated code in ETSpy, you will need to install `cupy`. 
    This can be done by specifying the `[gpu]` group when installing ETSpy:

    ```shell
    $ pip install etspy[gpu]
    ```

  * A shortcut for doing both of the above is to install the `[all]` target:

    ```shell
    $ pip install etspy[all]
    ```  

  * To register the ETSpy virtual environment as a Jupyter kernel, run the following with
    the virtual environment enabled:

    ```shell
    (etspy) $ python -m ipykernel install --user --name ETSpy
    ```

  Some dependencies of ETSpy require compilation of C code, meaning using the Anaconda approach
  above will simplify things greatly if you have trouble with "pure" `pip`.

### Removal

The package can be removed with:

```shell
$ pip uninstall etspy
```

## Basic Usage

The majority of the functionality of ETSpy can be accessed by importing the `etspy.api` module.
For example, to load a tilt series dataset into a `TomoStack`, you could do the following:

```python
import etspy.api as etspy
stack = etspy.load('TiltSeries.mrc')
```

For more details, see the dedicated [documentation](https://pages.nist.gov/etspy), including
the [example Jupyter notebook](https://pages.nist.gov/etspy/examples/etspy_demo.html) and the more detailed
[API Reference](https://pages.nist.gov/etspy/api.html).

## Developer documentation

See the [developer docs](https://pages.nist.gov/etspy/development) for more information.

## Related projects

- [https://hyperspy.org/](https://hyperspy.org/)
- [https://astra-toolbox.com/](https://astra-toolbox.com/)
