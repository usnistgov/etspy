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
