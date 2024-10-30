"""Tests for the ETSpy package."""

from hyperspy.io import load as hs_load

from etspy.api import etspy_path, load


def hspy_mrc_reader_check():
    """Test loading of an MRC file using the HyperSpy reader."""
    dirname = etspy_path / "tests" / "test_data" / "SerialEM_Multiframe_Test"
    files = dirname.glob("*.mrc")
    file = next(files)
    s = hs_load(file)
    return s

def load_serialem_multiframe_data():
    """
    Load example dataset that yields a multiframe tilt stack.

    Signal dimensions are (2, 3|1024, 1024); underlying Numpy array shape
    is (3, 2, 1024, 1024); nframes = 2; ntilts = 3
    """
    dirname = etspy_path / "tests" / "test_data" / "SerialEM_Multiframe_Test"
    files = list(dirname.glob("*.mrc"))
    stack = load(files)
    return stack
