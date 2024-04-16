"""Setup file for TomoTools package."""

from setuptools import setup

setup(
    name='tomotools',
    version='0.6.0',
    author='Andrew A. Herzing',
    description='Suite of data processing algorithms for processing electron '
            'tomography data',
    packages=['tomotools'],
    install_requires=[
        'pystackreg',
    ]
)
