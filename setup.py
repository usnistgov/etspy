"""Setup file for TomoTools package."""

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
        # 'hyperspy',
        # 'numpy',
        # 'astra-toolbox',
        # 'opencv-python',
        # 'scipy',
        # 'tqdm',
        # 'matplotlib',
        'tomopy @ git+http://git@github.com/tomopy/tomopy.git#egg=tomopy-1.2.1-py3.6',
    ],
    dependency_links=[
        'git+https://git@github.com/tomopy/tomopy.git#egg=tomopy-1.2.1-py3.6'
    ]
)
