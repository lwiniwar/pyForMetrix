# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

autodoc_mock_imports = ['xarray', 'pandas', 'numpy',
                        'scipy', 'laxpy', 'tqdm',
                        'matplotlib', 'shapely']
# -- Project information

project = 'pyForMetrix'
copyright = '2022, Lukas Winiwarter'
author = 'Lukas Winiwarter'
master_doc = 'index'
release = '0.0'
version = '0.0.1'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',

    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
}


intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'