"""Test version of package."""

import importlib.metadata

import tomli

from etspy.api import etspy_path


def test_version():
    """Test version definition."""
    pyproject = etspy_path.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomli.load(f)
        toml_version = data["tool"]["poetry"]["version"]
    assert importlib.metadata.version("etspy") == toml_version
