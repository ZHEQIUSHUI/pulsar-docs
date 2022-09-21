# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))

import sphinx_rtd_theme
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify
import sphinxcontrib.mermaid
# -- Project information -----------------------------------------------------

project = 'SuperPulsar'
copyright = '2022, AXera 爱芯元智'
author = 'AXera-Tech'

# The full version, including alpha/beta/rc tags
release = 'V0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# myst_parser is incompatible with recommonmark, myst_parser support mermaid.
extensions = ['sphinxcontrib.mermaid', 'recommonmark', 'sphinx_markdown_tables', 'sphinx_copybutton']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:

# source_parsers = {
#   '.md': CommonMarkParser,
# }
source_suffix = ['.rst', '.md']

# mermaid
mermaid_output_format = 'raw'
mermaid_version = 'latest'
