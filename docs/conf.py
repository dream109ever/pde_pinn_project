# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PINN PDE Solver'
copyright = '2026, dream109ever'
author = 'dream109ever'
release = '1.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

project_root = os.path.abspath('..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

extensions = [
    'sphinx.ext.autodoc',  
    'sphinx.ext.viewcode', 
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
]

mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js'
mathjax_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

suppress_warnings = [
    'autodoc',      
    'misc',         
]

# -- Autodoc 配置 ------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_docstring_signature = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'show-inheritance': True,
    'special-members': '__init__',
}

autodoc_mock_imports = [
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtWidgets',
    'PyQt5.QtGui',
    'PyQt5.QtXml',
    'PyQt5.QtChart',
    'ipywidgets',
    'IPython',
    'IPython.display',
    'matplotlib',
    'matplotlib.backends',
    'matplotlib.backends.backend_qt5agg',
    'mpl_toolkits',
]

autodoc_import_modules = True
