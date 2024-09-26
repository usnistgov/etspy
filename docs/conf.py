# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
import sys

from pathlib import Path
from typing import cast

sys.path.insert(0, os.path.abspath(".."))

# PACKAGE_PATH used to exclude autodoc members not from ETSpy
PACKAGE_PATH = Path(__file__).parent.parent

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ETSpy'
# copyright = '2024, Andrew Herzing, Joshua Taillon'
author = 'Andrew Herzing, Joshua Taillon'
release = '0.8.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = [
#     "numpydoc",
#     "sphinx.ext.autodoc",
#     "sphinx.ext.autosummary",
#     "sphinx.ext.duration",
#     "sphinx.ext.doctest",
#     "sphinx.ext.githubpages",
#     "sphinx.ext.intersphinx",
#     "sphinx.ext.napoleon",
#     "sphinx_copybutton",
#     "sphinx_favicon",
#     "myst_parser",
# ]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_immaterial",
    "sphinx_immaterial.apidoc.python.apigen",
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
]

master_doc = "index"
templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
source_suffix = [".rst", ".md"]
intersphinx_mapping = {
    "exspy": ("https://hyperspy.org/exspy/", None),
    "hyperspy": ("https://hyperspy.org/hyperspy-doc/current/", None),
    "rsciio": ("https://hyperspy.org/rosettasciio/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Define a custom inline Python syntax highlighting literal
rst_prolog = """
.. role:: python(code)
   :language: python
   :class: highlight
"""

# Sets the default role of `content` to :python:`content`, which uses the custom Python syntax highlighting inline literal
default_role = "python"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_immaterial'
html_static_path = ['_static']
html_logo = "_static/ETSpy_logo_trans.png"
html_css_files = [
    "custom_css.css",
]
html_favicon = '_static/favicon-64x64.png'


html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    "site_url": "https://pages.nist.gov/etspy/",
    "repo_url": "https://github.com/usnistgov/etspy/",
    "repo_name": "ETSpy",
    "edit_uri": "blob/main/docs",
    "globaltoc_collapse": True,
    "features": [
        "navigation.expand",
        # "navigation.tabs",
        "toc.integrate",
        "navigation.sections",
        "navigation.instant",
        # "header.autohide",
        "navigation.top",
        # "navigation.tracking",
        # "search.highlight",
        "search.share",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "blue",
            "accent": "indigo",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "blue",
            "accent": "indigo",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            },
        },
    ],
    "social": [
        {
            "icon": "fontawesome/brands/github",
            "link": "https://github.com/usnistgov/etspy",
            "name": "Source on github.com",
        },
        {
            "icon": "fontawesome/brands/python",
            "link": "https://pypi.org/project/etspy/",
        },
    ],
}


favicons = [
    "favicon-16x16.png",
    "favicon-32x32.png",
    "favicon-64x64.png",
    "favicon-128x128.png",
    "favicon-256x256.png",
    "favicon-512x512.png",
    "icon.svg",
]

autodoc_default_options = {
    "imported-members": True,
    "members": True,
    # "special-members": True,
    # "inherited-members": "ndarray",
    # "member-order": "groupwise",
}
autodoc_typehints = "signature"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"

# -- Sphinx Immaterial configs -------------------------------------------------

# Python apigen configuration
python_apigen_modules = {
    "etspy": "api/etspy.",
    "etspy.align": "api/etspy.align.",
    "etspy.api": "api/etspy.api.",
    "etspy.base": "api/etspy.base.",
    "etspy.datasets": "api/etspy.datasets.",
    "etspy.io": "api/etspy.io.",
    "etspy.recon": "api/etspy.recon.",
    "etspy.simulation": "api/etspy.simulation.",
    "etspy.utils": "api/etspy.utils.",
}
python_apigen_show_base_classes = True

python_apigen_default_groups = [
    (".*etspy\\.api.*", "api"),
    (".*etspy\\.align.*", "align"),
    (".*etspy\\.utils.*", "utilities"),
    (".*etspy\\.io.*", "io"),
    (".*etspy\\.simulation.*", "simulation"),
    (".*etspy\\.recon.*", "recon"),
    (".*etspy\\.datasets.*", "datasets"),
    (".*etspy\\.base.*", "signals"),
    # ("class:.*", "Classes"),
    # ("data:.*", "Variables"),
    # ("function:.*", "Functions"),
    # ("classmethod:.*", "Class methods"),
    # ("method:.*", "Methods"),
    (r"method:.*\.[A-Z][A-Za-z,_]*", "Constructors"),
    (r"method:.*\.__[A-Za-z,_]*__", "Special methods"),
    (r"method:.*\.__(init|new)__", "Constructors"),
    (r"method:.*\.__(str|repr)__", "String representation"),
    ("property:.*", "Properties"),
    (r".*:.*\.is_[a-z,_]*", "Attributes"),
]
python_apigen_default_order = [
    # ("class:.*", 10),
    # ("data:.*", 11),
    # ("function:.*", 12),
    # ("classmethod:.*", 40),
    # ("method:.*", 50),
    (r"method:.*\.[A-Z][A-Za-z,_]*", 20),
    (r"method:.*\.__[A-Za-z,_]*__", 28),
    (r"method:.*\.__(init|new)__", 20),
    (r"method:.*\.__(str|repr)__", 30),
    ("property:.*", 60),
    (r".*:.*\.is_[a-z,_]*", 70),
]
python_apigen_order_tiebreaker = "alphabetical"
python_apigen_case_insensitive_filesystem = False

# autodoc_default_options = {
#     # other options
#     'inherited-members': False
# }

# def autodoc_process_bases(app, name, obj, options, bases):
#     """
#     Remove private classes or mixin classes from documented class bases.
#     """
#     # Determine the bases to be removed
#     remove_bases = []
#     for base in bases:
#         print(f"{app}, {name}, {obj}, {base}")
#         if "etspy" not in str(base):
#             remove_bases.append(base)

#     # Remove from the bases list in-place
#     for base in remove_bases:
#         bases.remove(base)

def autodoc_skip_member(app, what, name, obj, skip, options):
    """
    Instruct autodoc to skip members that not directly from ETSpy.
    """
    # print(f"{what}, {name}, {obj}, {skip}")

    if skip:
        # Continue skipping things Sphinx already wants to skip
        return skip

    if name == "__init__":
        return False
    
    try:
        # get source file for property via fget
        if isinstance(obj, property):
            p = Path(inspect.getfile(obj.fget))  # pyright: ignore[reportArgumentType]
        else:
            p = Path(inspect.getfile(obj))
        if PACKAGE_PATH not in p.parents:
            # skip if member does not come from ETSpy
            # print(f"skipping {p}")
            return True
    except TypeError:
        pass
        # print(f"could not get file for {what}, {name}, {obj},")


    # if name[0] == "_":
    #     # For some reason we need to tell Sphinx to hide private members
    #     return True

    return skip

def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
    # app.connect("autodoc-process-bases", autodoc_process_bases)
    # app.connect("autodoc-process-signature", autodoc_process_signature)