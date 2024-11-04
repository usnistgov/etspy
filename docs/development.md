# Development tips

```{eval-rst}
.. note::

      This page contains information useful for developers and those who may wish to
      contribute to ETSpy.
```

## Installing a development version of ETSpy

If you wish to contribute to ETSpy or otherwise install a development version,
this can be accomplished using either [Anaconda](https://www.anaconda.com/download/success)
or [Poetry](https://python-poetry.org).

If using conda, create and activate a new environment using one of the *development* specifications:

| Description | Link |
| ----------- | ---- |
| Development environment without `cupy` | [`etspy-dev.yml`](https://raw.githubusercontent.com/usnistgov/etspy/refs/heads/master/resources/etspy-dev.yml) |
| Development environment with `cupy` and CUDA packages for GPU operations | [`etspy-gpu-dev.yml`](https://raw.githubusercontent.com/usnistgov/etspy/refs/heads/master/resources/etspy-gpu-dev.yml) |


Then you will need to [fork the repository on Github](https://github.com/usnistgov/etspy/fork), clone the repo locally, and
install the package in "editable" mode using `pip`:

```shell
$ conda env create -f https://raw.githubusercontent.com/usnistgov/etspy/refs/heads/master/resources/etspy-dev.yml
# or the following for the GPU dependencies:
#    conda env create -f https://raw.githubusercontent.com/usnistgov/etspy/refs/heads/master/resources/etspy-gpu-dev.yml
$ conda activate etspy-dev
$ git clone https://github.com/<your_account_name>/etspy
$ cd etspy
$ pip install -e .
```

If using Poetry (currently only working on Linux due to some limitations of dependency packages),
make sure you have `poetry` and the CUDA libraries installed, clone the `etspy` repository, and
run the install command:

```shell
$ git clone https://github.com/<your_account_name>/etspy
$ cd etspy
$ poetry install   # (to get the cupy dependency add "--all-extras" to the install command)
```

```{note}
Sometimes, on headless Linux systems without a desktop environment installed, the `poetry install`
command will hang due to an outstanding issue with handling the system keyring
(see [this issue](https://github.com/python-poetry/poetry/issues/8623)). To workaround the issue,
run the command `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` prior to running
`poetry install`, and it should work. 
```

At this point, you should have an "editable" version of the latest development
version of ETSpy installed to hack on as you wish.

Contributions to the project are welcome, and can be submitted via the GitHub pull
request mechanism.

## Coverage

ETSpy strives to have high code coverage through the use of tests in the `etspy/tests/` directory.
When developing, you can run the tests from the main directory with the following command, which
will output the coverage results to the terminal, as well as to a `etspy/tests/coverage.xml` file that can
be interpreted by various editors to display the coverage stats interactively, and as html in the
`etspy/tests/htmlcov` directory that can be viewed in a web browser:

```shell
$ poetry run pytest etspy/tests/
```

By default, this will exclude CUDA-related code from the coverage
report (via the `[tool.coverage.report]` setting in `pyproject.toml`),
since most CI/CD systems will not have CUDA enabled. If you would
like to run the tests with coverage (including CUDA), you can use
the `run_tests.sh` helper script (on Linux), which will detect whether
or not CUDA is available, and choose whether or not to exclude those
lines from the report depending:

```shell
$ poetry run ./run_tests.sh
```

## Debugging when using coverage

ETSpy has the test suite configured to automatically run code coverage
analysis when using `pytest`. This interferes when using interactive
debuggers (such as PyCharm or VSCode), since they use the same mechanism
under the hood to inspect what code is being run. This will manifest as
your breakpoints never triggering when running a "Debug" configuration.
For more information, see the following links:
[one](https://github.com/microsoft/vscode-python/issues/693),
[two](https://stackoverflow.com/a/67185092),
[three](https://youtrack.jetbrains.com/issue/PY-20186). There are a
few workarounds discussed in those threads, but the gist is that
when debugging, you should disable the coverage plugin to ensure
that breakpoints will be hit. This varies depending on your IDE/setup,
but one option for VSCode is to put the following configuration
in your project's `launch.json`, which will ensure coverage is disabled
when running "Debug test" via the `PYTEST_ADDOPTS` environment variable:

```json
  {
      "name": "Debug Tests",
      "type": "debugpy",
      "request": "launch",
      "purpose": ["debug-test"],
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {"PYTEST_ADDOPTS": "--no-cov"}
  }
```

## Releasing a version

### Testing a pre-release

```bash
# bump version using poetry
$ poetry version prerelease  # this will append the version number and a pre-release indicator e.g ".a0"
$ poetry lock  # ensure you've updated the lockfile and any dependencies
$ poetry build  # buildssource and binary "wheel" distributions
$ poetry publish  # requires registering poetry with tokens for your PyPI account (see https://python-poetry.org/docs/repositories/#configuring-credentials )
```

You should then be able to install from PyPI with the new version (e.g. `pip install etspy==0.9.3a2`)

### Releasing a new version

- Basically the same as above, but run `poetry version patch` rather than `prerelease`.
- Should also create a git tag for the version, and create a release on GitHub. This may be done automatically in the future.

## Setting up conda-forge packages

- Packages on `conda-forge` were set up using the great instructions provided by the
  [PyOpenSci](https://www.pyopensci.org/python-package-guide/tutorials/publish-conda-forge.html)
  project
