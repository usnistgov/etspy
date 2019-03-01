from setuptools import setup, find_packages

setup(
    name = 'tomotools',
    version = '0.2.0',
    author = 'Andrew A. Herzing',
    description = 'Suite of data processing algorithms for processing electron tomography data',
    packages = [
        'tomotools',
        ],
    install_requires = [
        'hyperspy',
        'numpy',
        'astra-toolbox',
        'scipy',
        'tqdm',
        'matplotlib',
        'tomopy',
    ],
)
