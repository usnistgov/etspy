"""Setup file for TomoTools package."""

from setuptools import setup

setup(
    name='tomotools',
    version='0.7.1',
    author='Andrew A. Herzing',
    description='Suite of tools for processing and reconstruction of electron '
            'tomography data',
    packages=['tomotools'],
    install_requires=[
        'pystackreg',
    ]
)
