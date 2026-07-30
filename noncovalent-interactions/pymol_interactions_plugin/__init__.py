"""PyMOL Plugin Manager package wrapper for interactions_plugin.

Loading this package (via Plugin Manager > Install from the .zip) imports the
implementation module, which registers the `detect_interactions` and
`show_interaction_legend` commands. `__init_plugin__` is the Plugin Manager
entry point.
"""

from . import interactions_plugin

__version__ = "0.4.1"


def __init_plugin__(app=None):
    interactions_plugin.__init_plugin__(app)
