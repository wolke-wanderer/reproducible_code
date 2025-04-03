# ruff: noqa: E402  # ignore imports not at top
"""
General plotting style defaults

.. module:: plot_defaults

:author:
    Falco Bentvelsen (falco.bentvelsen@knmi.nl)

:copyright:
    2024, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
__version__ = "0.2"
__author__ = "Falco Bentvelsen"
__all__ = ["plt", "mcolors", "mdates", "pe", "mticker", "cmap0", "cmap1", "cmaps"]

import os

import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 100
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

# Add Tex Gyre Heros to Matplotlib
# download & unzip from http://www.gust.org.pl/projects/e-foundry/tex-gyre/heros -> "basic stuff" link
from matplotlib import font_manager, rcParams

# %matplotlib widget


homedir = os.environ["HOME"]
for font_path in font_manager.findSystemFonts(
    f"{homedir}/lib/qhv2/fonts/opentype/public/tex-gyre/"
):
    font_manager.fontManager.addfont(font_path)
# rcParams['font.family'] = 'tex gyre heros'  # TODO: ask Jelle where to get tex gyre heros
rcParams["font.family"] = "DejaVu Sans"

cmap0 = mcolors.LinearSegmentedColormap.from_list(
    "", ["white", *plt.cm.get_cmap("magma_r").colors]
)
cmap1 = mcolors.LinearSegmentedColormap.from_list(
    "", [*plt.cm.get_cmap("magma").colors, "white"]
)

# TODO: expand plot_defaults with KNMI colors, fonts, etc.

# import colormaps as cmaps
import matplotlib.cm as cmaps
