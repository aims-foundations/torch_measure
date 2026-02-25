# Configuration file for the Sphinx documentation builder.

project = "torch_measure"
copyright = "2026, AIMS Foundation"
author = "AIMS Foundation"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Theme --
html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/aims-foundation/torch_measure",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# -- Autodoc --
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Intersphinx --
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

autosummary_generate = True

# -- Markdown support --
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
