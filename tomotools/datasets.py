import tomotools.api as tomotools
from tomotools.simulation import misalign_stack, add_gaussian_noise


def get_needle_data(aligned=False):
    """
    Retrieve experimental tilt series of needle-shaped specimen.

    Returns
    ----------
    needle : TomoStack object
        TomoStack containing the simulated catalyst tilt series

    """
    test_data_path = os.path.dirname(tomotools.__file__)
    # test_data_path = imp.find_module("tomotools")[1] + '\\tests\\test_data\\'
    if aligned:
        needle =\
            tomotools.load(test_data_path + 'HAADF_Aligned.hdf5')
    else:
        needle =\
            tomotools.load(test_data_path + 'HAADF.mrc')
    return needle


def get_catalyst_tilt_series(misalign=False, minshift=-5, maxshift=5,
                             tiltshift=0, tiltrotate=0, xonly=False,
                             noise=False, noise_factor=0.2):
    """
    Retrieve model catalyst tilt series.

    Returns
    ----------
    catalyst : TomoStack object
        TomoStack containing the simulated catalyst tilt series

    """
    test_data_path = os.path.dirname(tomotools.__file__)
    # test_data_path = imp.find_module("tomotools")[1] + '\\tests\\test_data\\'
    catalyst =\
        tomotools.load(test_data_path + 'Catalyst3DModel_TiltSeries180.hdf5')
    if misalign:
        catalyst = misalign_stack(catalyst, min_shift=minshift,
                                  max_shift=maxshift, tilt_shift=tiltshift,
                                  tilt_rotate=tiltrotate, x_only=xonly)
    if noise:
        catalyst = add_gaussian_noise(catalyst, noise_factor)
    return catalyst
