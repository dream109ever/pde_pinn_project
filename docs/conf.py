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

sys.path.insert(0, os.path.abspath('../src'))

extensions = [
    'sphinx.ext.autodoc',      # 自动从 docstring 生成文档
    'sphinx.ext.napoleon',     # 支持 Google/NumPy 风格 docstring
    'sphinx.ext.viewcode',     # 显示源代码链接
    'sphinx.ext.mathjax',      # 支持数学公式
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
    'autodoc',           # 忽略 autodoc 的所有警告
    'misc',              # 忽略其他杂项警告
]

# 自动生成文档时的配置
autodoc_member_order = 'bysource'  # 按源码顺序排列
autodoc_typehints = 'description'  # 在描述中显示类型提示
autodoc_docstring_signature = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'show-inheritance': True,
    'special-members': '__init__',
    'exclude-members': '__weakref__',
}

# 排除一些不需要的文件
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
