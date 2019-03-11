from setuptools import setup

setup(
    name='tomotools',
    version='0.3.0',
    author='Andrew A. Herzing',
    description='Suite of data processing algorithms for processing electron '
            'tomography data',
    packages=[
        'tomotools',
        ],
    install_requires=[
        'hyperspy',
        'numpy',
        'astra-toolbox',
        'scipy',
        'tqdm',
        'matplotlib',
      #  'tomopy @ https://github.com/tomopy/tomopy/archive/1.4.0.zip',
    ],
)
