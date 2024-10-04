# ETSpy

> ⚠️ ETSpy is still in a pre-release status, and the API may change with little warning, and
> the documentation may not be 100% correct/complete!

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

### Anaconda (Preferred)

  *Works on Windows, MacOS, and Linux*

  * First, ensure you have either [Anaconda](https://www.anaconda.com/download/success)
    or [Miniconda](https://docs.anaconda.com/miniconda/) installed on your system.

  * Run the following command to create a new environment based on the contents
    of the ETSpy YAML specification file, which will install the required dependencies,
    and then activate the newly created environment:
    
    ```shell
    $ conda env create -f https://raw.githubusercontent.com/usnistgov/etspy/refs/heads/master/resources/etspy.yml
    # if you would like your environment to be stored in a specific place, use the "-p <path>" option
    
    $ conda activate etspy
    ```

  * Finally, (with the `etspy` environment activated), install the ETSpy package:

    ```shell
    (etspy) $ pip install etspy
    ``` 

####  Optional Jupyter components (higly recommended)

  * To use ETSpy from within a Jupyter Lab/Notebook environment, a few other optional 
    dependencies are required.
    * `ipympl` enables interactive plotting in Jupyter Lab or Notebook.  
    * `ipykernel` allows use of the the etspy kernel with Jupyter installed in a different environment. 

    ```shell
    (etspy) $ conda install ipympl ipykernel
    ```

  * To "register" the python kernel associated with the `etspy` conda environment, run
    the following after ensuring that environment is activated (with `conda activate etspy`):

    ```shell
    (etspy) $ python -m ipykernel install --user --name ETSpy
    ```

    You will then be able to select the "ETSpy" kernel when running Jupyter and creating new
    notebooks

###  Using pip

  *Works on Linux, with additional prerequisites*

  Assuming you have the prequisite packages on your system (including
  the [CUDA libraries](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)),
  ETSpy should be able to be installed with a simple `pip` command (it is recommended to install
  ETSpy in a dedicated virtual environment):

  * ```shell
    $ pip install etspy
    ```

  * To use ETSpy in Jupyter interface from within a dedicated virtual environment, installing
    `ipykernel` and `ipympl` are necessary (as with Anaconda). This can be done by specifying
    the `[jupyter]` group when installing ETSpy:

    ```shell
    $ pip install etspy[jupyter]
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
the [example Jupyter notebook](https://pages.nist.gov/etspy/examples) and the more detailed
[API Reference](https://pages.nist.gov/etspy/api).

## Related projects

- [https://hyperspy.org/](https://hyperspy.org/)
- [https://astra-toolbox.com/](https://astra-toolbox.com/)
