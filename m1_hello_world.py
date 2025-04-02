# ruff: noqa: E402  # ignore imports not at top
"""
.. module:: hello_world

:author:
    Falco Bentvelsen (falco.bentvelsen@knmi.nl)

:copyright:
    2025, Falco Bentvelsen

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)


:summary:
    Preparation for the reproducible code tutorial
"""
# module level dunders
# __all__ = ["__version__", "__author__"]  # TODO: learn __all__ and use it
__version__ = "0.1"
__author__ = "Falco"

# %%
# %%
""" 1.-3. import packages """
# 1. import general packages
########################################
import os
import netCDF4 as nc
import numpy as np
import sys
# from typing import Tuple

# 2. import third party packages
########################################
import pandas as pd
import xarray as xr
# from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

# 3. import local packages
########################################
from logs import configure, jokes, logging
# from modules.obspy_extension import reset_tag_year_from_terminal
# from modules.plotting_parameters import marker_config, marker_config_ec, line_config_ec

from IPython import get_ipython

ipython = get_ipython()
if ipython:
    ipython.run_line_magic("config", "Completer.use_jedi = False")
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")


# %%
""" 4. define global constants """
################################################################################
TAG = "2020_sns_hist_2pvu"
################################################################################
TAG_bf_in = "2020_bb_full_p6"
################################################################################
configure.setup_logging(tag=TAG)  # set up logging
logging.info("  TAG = %s\n %s", TAG, jokes.get_random_joke())  # test
#######################################################################

# %% define functions
""" 5. define functions """
def print_hello_world():
    """Prints a hello world message."""
    print("Hello, World!")

def create_sine_wave_plot():
    """Creates a simple sine wave plot."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Sine Wave")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid()
    # plt.show()
    return fig, ax

# # example usage
# print_hello_world()

# %%
def reset_tag_year_from_terminal(tag):
    """Reset the year in the tag based on a command-line argument."""
    if len(sys.argv) > 1:
        try:
            year = int(sys.argv[1])
            parts = tag.split("_")
            if len(parts) > 0:
                parts[0] = str(year)
                new_tag = "_".join(parts)
                logging.info("Tag updated from %s to %s", tag, new_tag)
                return new_tag
        except ValueError:
            logging.error("Invalid year provided. Please provide a valid integer year.")
    else:
        logging.warning("No year provided. Using the default tag.")
    return tag

# %%
""" 6. main """
if __name__ == "__main__":
    # 6.0. reset the tag year from terminal
    TAG = reset_tag_year_from_terminal(tag=TAG)  # Update TAG with the new year if provided

    # 6.1. test the logging
    logging.info("  TAG = %s\n %s", TAG, jokes.get_random_joke())  # test

    # 6.2. run the main function
    fig, ax = create_sine_wave_plot()
    # 6.3. save the figure
    fig.savefig("figures/sine_wave_plot.png")

# %%
""" 7. clean up / finish """
# close the figures
plt.close("all") 

# end with a joke
logging.info("  %s", jokes.get_random_joke())

# %%
