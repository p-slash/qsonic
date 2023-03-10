# I ran `sphinx-apidoc -o docs/source/generated py/qsonic`
# and changed every file in generated directory to have qsonic.io etc

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# import os
# import sys

# sys.path.insert(0, os.path.abspath('../../py/'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'qsonic'
copyright = '2023, Naim Goksel Karacayli'
author = 'Naim Goksel Karacayli'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinxarg.ext'
    # 'nbsphinx',
    # 'autoapi.extension'
]

# autodoc_mock_imports = ["mpi4py"]
autodoc_member_order = "bysource"
# autodoc_class_signature = "separated"
autodoc_default_options = {
    # 'members': 'var1, var2',
    # 'member-order': 'bysource',
    # 'special-members': '__call__',
    # 'undoc-members': True,
    'exclude-members': '__init__',
    'private-members': True
}

exclude_patterns = []

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None)
}
intersphinx_disabled_domains = ['std']
# intersphinx_disabled_reftypes = ["*"]

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']
