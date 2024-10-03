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

project = "ETSpy"
# copyright = '2024, Andrew Herzing, Joshua Taillon'
author = "Andrew Herzing, Joshua Taillon"
release = "0.8.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

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

today_fmt = '%B %-d, %Y at %I:%M %p'
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
source_suffix = [".rst", ".md"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "hyperspy": ("https://hyperspy.org/hyperspy-doc/current", None),
    "rsciio": ("https://hyperspy.org/rosettasciio", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "astra": ("https://astra-toolbox.com", None)
}

# Define a custom inline Python syntax highlighting literal
rst_prolog = """
.. role:: python(code)
   :language: python
   :class: highlight
"""

# Sets the default role of `content` to :python:`content`, which uses the custom Python syntax highlighting inline literal
# default_role = "python"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_logo = "_static/ETSpy_logo_trans.png"
html_css_files = [
    "custom_css.css",
]
html_favicon = "_static/favicon-64x64.png"


html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    "site_url": "https://pages.nist.gov/etspy/",
    "repo_url": "https://github.com/usnistgov/etspy/",
    "repo_name": "ETSpy",
    "edit_uri": "blob/master/docs",
    "globaltoc_collapse": True,
    "toc_title_is_page_title": True,
    "features": [
        "navigation.expand",
        # "navigation.tabs",
        # "toc.integrate",
        # "navigation.sections",
        "navigation.instant",
        # "header.autohide",
        "navigation.top",
        "navigation.tracking",
        "search.highlight",
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

# make sure "Examples", "Notes", etc. rubrics are included in right-side ToC 
object_description_options = [
    ("py:.*", dict(
        include_object_type_in_xref_tooltip=False,
        include_rubrics_in_toc=True,
    )),
]

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

# set up "automatic" groupings for members
python_apigen_default_groups = [
    ("class:.*", "Classes"),
    ("data:.*", "Variables"),
    ("function:.*", "Functions"),
    ("classmethod:.*", "Class methods"),
    ("method:.*", "Methods"),
    (r"method:.*\.[A-Z][A-Za-z,_]*", "Constructors"),
    (r"method:.*\.__[A-Za-z,_]*__", "Special methods"),
    (r"method:.*\.__(init|new)__", "Constructors"),
    (r"method:.*\.__(str|repr)__", "String representation"),
    ("property:.*", "Properties"),
    (r"etspy\.AlignmentMethod.*", "align"),
    # (r".*:.*\.is_[a-z,_]*", "Attributes"),
    (r"attribute:.*etspy\.AlignmentMethod.*", "Enum Values"),
    # (".*etspy\\.api.*", "api"),
    # (".*etspy\\.align.*", "align"),
    # (".*etspy\\.utils.*", "utilities"),
    # (".*etspy\\.io.*", "io"),
    # (".*etspy\\.simulation.*", "simulation"),
    # (".*etspy\\.recon.*", "recon"),
    # (".*etspy\\.datasets.*", "datasets"),
    # (".*etspy\\.base.*", "signals"),
]
python_apigen_default_order = [
    (".*etspy\\.api.*", 20),
    (".*etspy\\.align.*", 21),
    (".*etspy\\.utils.*", 22),
    (".*etspy\\.io.*", 23),
    (".*etspy\\.simulation.*", 24),
    (".*etspy\\.recon.*", 25),
    (".*etspy\\.datasets.*", 26),
    (".*etspy\\.base.*", 27),
    ("class:.*", 30),
    ("data:.*", 31),
    ("function:.*", 32),
    ("classmethod:.*", 40),
    ("method:.*", 50),
    (r"method:.*\.[A-Z][A-Za-z,_]*", 20),
    (r"method:.*\.__[A-Za-z,_]*__", 28),
    (r"method:.*\.__(init|new)__", 20),
    (r"method:.*\.__(str|repr)__", 30),
    ("property:.*", 60),
    (r".*:.*\.is_[a-z,_]*", 70),
]
python_apigen_order_tiebreaker = "definition_order"
python_apigen_case_insensitive_filesystem = False


def autodoc_skip_member(app, what, name, obj, skip, options):
    """
    Instruct autodoc to skip members that not directly from ETSpy.
    """
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
            return True
    except TypeError as e:
        if 'AlignmentMethod' in str(obj):
            return False
        return 'etspy' not in str(obj)

    return skip

def autodoc_process_signature(app, what, name, obj, options, signature, return_annotation):
    if return_annotation == "~typing.Self":
        print(f"SIGNATURE: {what}, {name}, {obj}, {options}, {signature}, {return_annotation}")
    
    # replace "Self" annotations with current class name
    if return_annotation == "~typing.Self" and name[:11] == "etspy.base.TomoStack":
        replaced_annotation = "~" + '.'.join(name.split('.')[:-1])
        return signature, replaced_annotation
    
def autodoc_process_docstring(app, what, name, obj, options, lines):
    if 'TomoStack' in name:
        print(f"DOCSTRING: {what}, {name}, {obj}, {options}, {lines}")
    pass

def autodoc_process_bases(app, name, obj, options, bases):
    print(f"BASES: {app}, {name}, {obj}, {options}, {bases}")
    pass

def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
    # app.connect("autodoc-process-bases", autodoc_process_bases)
    # app.connect("autodoc-process-docstring", autodoc_process_docstring)
    # app.connect("autodoc-process-signature", autodoc_process_signature)


# -- Link checking configs -------------------------------------------------

linkcheck_ignore = [
    "https://doi.org/10.1103/PhysRevB.72.052103"  # 403 Client Error: Forbidden for url: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.72.052103
]

linkcheck_exclude_documents = []

# Specify a standard user agent, as Sphinx default is blocked on some sites
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.54"
