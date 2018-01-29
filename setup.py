from distutils.core import setup

setup(
    name='tomotools',
    version='0.1.0',
    author='Andrew A. Herzing',
    description='Suite of data processing algorithems for processing electron tomography data',
    install_requires=[
        "hyperspy >= 1.0",
        "numpy",
        "scipy",
        "matplotlib",
        "astra-toolbox",
        "opencv",
        "tqdm"
    ],
)
