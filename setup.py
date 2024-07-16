"""Setup file for ETSpy package."""

from setuptools import setup

setup(
    name='etspy',
    version='0.7.1',
    author='Andrew A. Herzing',
    description='Suite of tools for processing and reconstruction of electron '
            'tomography data',
    packages=['etspy'],
    install_requires=[
        'pystackreg',
    ]
)
