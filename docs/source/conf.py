# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
# -- Project information

from sphinx_pyproject import SphinxConfig

config = SphinxConfig("../../pyproject.toml", globalns=globals())
#
project = 'pyForMetrix'
copyright = '2022, Lukas Winiwarter'
author = 'Lukas Winiwarter'
master_doc = 'index'
# release = '0.0'
# version = '0.0.1a'


autodoc_mock_imports = ['xarray', 'pandas', 'numpy',
                        'scipy', 'laxpy', 'tqdm', 'laspy',
                        'matplotlib', 'shapely', 'deprecated']
# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',

    'm2r2',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'pandas': ('https://pandasguide.readthedocs.io/en/latest/', None),
    'geopandas': ('https://geopandas.org/en/stable/', None),
}


intersphinx_disabled_domains = ['std']


templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

#
# import matplotlib
# matplotlib.use('agg')