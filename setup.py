from distutils.core import setup

setup(
    name='TomoTools',
    version='0.1.0',
    author='Andrew A. Herzing',
    description='Suite of data processing algorithems for processing electron tomography data',
    install_requires=[
        "Hyperspy >= 1.0",
        "numpy",
        "scipy",
        "cv3",
        "matplotlib",
        "astra",
        "tqdm"
    ],
)
