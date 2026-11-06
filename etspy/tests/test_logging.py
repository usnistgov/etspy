"""Test the logging behaviour of ETSpy."""

import importlib
import logging

import numpy as np
import pytest

import etspy.api as etspy
from etspy.api import etspy_path

MODULES = [
    "etspy.align",
    "etspy.api",
    "etspy.base",
    "etspy.io",
    "etspy.projmatch.projmatch",
    "etspy.recon",
    "etspy.transform",
]


@pytest.fixture
def etspy_logger():
    """Restore the state of the ``etspy`` logger after a test."""
    logger = logging.getLogger("etspy")
    level, handlers = logger.level, list(logger.handlers)
    yield logger
    logger.setLevel(level)
    for handler in list(logger.handlers):
        if handler not in handlers:
            logger.removeHandler(handler)


class TestLogLevels:
    """Test control of the ETSpy log level."""

    @pytest.mark.parametrize("name", MODULES)
    def test_modules_do_not_set_log_level(self, name):
        # module loggers inherit their level from the "etspy" logger
        importlib.import_module(name)
        assert logging.getLogger(name).level == logging.NOTSET

    def test_etspy_logger_level_controls_modules(self, etspy_logger, caplog):
        filename = etspy_path / "tests" / "test_data" / "HAADF.mrc"
        with caplog.at_level(logging.INFO):
            etspy_logger.setLevel(logging.WARNING)
            etspy.load(filename)
            assert "Tilts found in MRC file header" not in caplog.text
            etspy_logger.setLevel(logging.INFO)
            etspy.load(filename)
            assert "Tilts found in MRC file header" in caplog.text

    @pytest.mark.usefixtures("etspy_logger")
    def test_set_log_level_displays_messages(self, capsys):
        etspy.set_log_level("INFO")
        etspy.load(etspy_path / "tests" / "test_data" / "HAADF.mrc")
        out = capsys.readouterr().out
        assert "INFO | ETSpy | Tilts found in MRC file header (etspy.io)" in out

    def test_set_log_level_single_handler(self, etspy_logger):
        before = len(etspy_logger.handlers)
        etspy.set_log_level("INFO")
        etspy.set_log_level("DEBUG")
        assert len(etspy_logger.handlers) == before + 1
        assert etspy_logger.level == logging.DEBUG

    def test_set_log_level_keeps_user_handlers(self, etspy_logger):
        user_handler = logging.NullHandler()
        etspy_logger.addHandler(user_handler)
        etspy.set_log_level("INFO")
        etspy.set_log_level("WARNING")
        assert user_handler in etspy_logger.handlers


class TestNoTiltsWarning:
    """Test the warning when loaded data has no tilt angles."""

    def test_warns_when_no_tilts(self, tmp_path, caplog):
        filename = tmp_path / "no_tilts.hspy"
        etspy.TomoStack(np.zeros((5, 8, 8), np.float32)).save(filename)
        with caplog.at_level(logging.WARNING, logger="etspy.io"):
            etspy.load(filename)
        assert "No tilt angles were found for the loaded data" in caplog.text

    def test_no_warning_with_user_tilts(self, tmp_path, caplog):
        filename = tmp_path / "no_tilts.hspy"
        etspy.TomoStack(np.zeros((5, 8, 8), np.float32)).save(filename)
        with caplog.at_level(logging.WARNING, logger="etspy.io"):
            etspy.load(filename, tilts=np.arange(5.0))
        assert "No tilt angles were found" not in caplog.text

    def test_no_warning_with_file_tilts(self, caplog):
        filename = etspy_path / "tests" / "test_data" / "HAADF.mrc"
        with caplog.at_level(logging.WARNING, logger="etspy.io"):
            etspy.load(filename)
        assert "No tilt angles were found" not in caplog.text


@pytest.fixture(scope="module")
def raw_needle():
    """Needle test data as stored (background of about -31900)."""
    return etspy.load(etspy_path / "tests" / "test_data" / "HAADF.mrc")


class TestBackgroundWarnings:
    """Test warnings about a background level that will cause artifacts."""

    def test_reconstruct_warns(self, raw_needle, caplog):
        with caplog.at_level(logging.WARNING, logger="etspy.base"):
            raw_needle.isig[120:121, :].reconstruct(
                "FBP",
                cuda=False,
                show_progressbar=False,
            )
        assert "background (vacuum) level of about -318" in caplog.text
        assert "a ring of incorrect intensity" in caplog.text

    def test_reconstruct_no_warning_after_removal(self, raw_needle, caplog):
        corrected = raw_needle.isig[120:121, :].remove_background()
        with caplog.at_level(logging.WARNING, logger="etspy.base"):
            corrected.reconstruct("FBP", cuda=False, show_progressbar=False)
        assert "background (vacuum) level" not in caplog.text

    def test_recon_error_warns(self, raw_needle, caplog):
        with caplog.at_level(logging.WARNING, logger="etspy.base"):
            raw_needle.recon_error(iterations=2, cuda=False)
        assert "a ring of incorrect intensity" in caplog.text

    @pytest.mark.parametrize(("method", "warns"), [("COM", True), ("PC", False)])
    def test_stack_register(self, raw_needle, caplog, method, warns):
        with caplog.at_level(logging.WARNING, logger="etspy.base"):
            raw_needle.inav[:10].stack_register(method)
        assert ("biased center of mass alignment" in caplog.text) == warns

    def test_tilt_align_com_warns(self, raw_needle, caplog):
        with caplog.at_level(logging.WARNING, logger="etspy.base"):
            raw_needle.tilt_align("CoM", iterations=1)
        assert "biased tilt axis alignment" in caplog.text

    @pytest.mark.parametrize(("background", "warns"), [(2.0, False), (20.0, True)])
    def test_warning_threshold(self, caplog, background, warns):
        # signal range of 1000, so 2 is 0.2% and 20 is 2% of the data range
        data = np.zeros((5, 32, 32), np.float32)
        data[:, 8:24, 8:24] = 1000
        stack = etspy.TomoStack(data + background, tilts=np.linspace(-60, 60, 5))
        with caplog.at_level(logging.WARNING, logger="etspy.base"):
            stack.reconstruct("FBP", cuda=False, show_progressbar=False)
        assert ("background (vacuum) level" in caplog.text) == warns


class TestInt16OffsetWarning:
    """Test the warning for unsigned data stored as int16 with an offset."""

    def test_offset_int16_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="etspy.io"):
            etspy.load(etspy_path / "tests" / "test_data" / "HAADF.mrc")
        assert "may be unsigned values stored with an offset of -32768" in caplog.text

    @pytest.mark.parametrize(
        "filename",
        ["HAADF.dm3", "SerialEM_MultiTilt_Test/test_000.mrc"],
    )
    def test_no_warning_for_other_data(self, caplog, filename):
        with caplog.at_level(logging.WARNING, logger="etspy.io"):
            etspy.load(etspy_path / "tests" / "test_data" / filename)
        assert "offset of -32768" not in caplog.text
