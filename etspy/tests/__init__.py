"""Tests for the ETSpy package."""

from hyperspy.io import load as hs_load

from etspy.api import etspy_path


def hspy_mrc_reader_check():
    """Test loading of an MRC file using the HyperSpy reader."""
    dirname = etspy_path / "tests" / "test_data" / "SerialEM_Multiframe_Test"
    files = dirname.glob("*.mrc")
    file = next(files)
    s = hs_load(file)
    return s
