"""Setup file for ETSpy package."""

from setuptools import setup

setup(
    name="etspy",
    version="0.8",
    author="Andrew A. Herzing",
    description="Suite of tools for processing and reconstruction of electron "
    "tomography data",
    packages=["etspy"],
    install_requires=[
        "pystackreg",
    ],
    package_data={
        # Include all data files in the data directory
        "etspy": [
            "tests/test_data/*",
            "tests/test_data/DM_Series_Test/*",
            "tests/test_data/SerialEM_Multiframe_Test/*",
        ],
    },
)
