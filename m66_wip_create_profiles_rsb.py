# ruff: noqa: E402  # ignore imports not at top
"""
.. module:: create_profiles_rsb

:author:
    Falco Bentvelsen (falco.bentvelsen@knmi.nl)

:copyright:
    2025, Falco Bentvelsen

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)


:summary:
    This module creates skew-T plots with hodographs and additional information for radiosonde data.
    # TODO: profiles of Richardson number, and brunt-vaisala frequency from radiosonde data.
"""
# module level dunders
# __all__ = ["__version__", "__author__"]  # TODO: learn __all__ and use it
__version__ = "0.1"
__author__ = "Falco Bentvelsen"

## %%
# 1.-3. import packages:
# 1. import general packages
########################################
import os
from IPython import get_ipython

# 2. import third party packages
########################################
import numpy as np
import xarray as xr

import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
from metpy.units import units
from pint import Quantity

import matplotlib.pyplot as plt
import seaborn as sns

# 3. import local packages
########################################
from logs import configure, jokes, logging


# further package setup
ipython = get_ipython()
if ipython:
    ipython.run_line_magic("config", "Completer.use_jedi = False")
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

sns.set_theme(style="whitegrid")

##############################################################################
# %%
# 4. globals and constants

# define TAG for saving and logging
#######################################################################
TAG = "2025_skewT_adv_rsb_t2"
#######################################################################
configure.setup_logging(tag=TAG)  # set up logging
logging.info(jokes.get_random_joke())  # test
#######################################################################
# %%
# # Uncomment the following parameters for usage testing (manual)
# stnm = 10548  # Meinigen (150 km West of WBCI)
# year = 2020
# month = 10
# day = 3
# hour = 00

# if stnm == 10868:
#     station_title = "München-Oberschlssheim"
# elif stnm == 10548:
#     station_title = "Meiningen"

##############################################################################
# %%
# 5. define functions
# %%
def load_radiosonde_data(stnm, year, month, day, hour) -> xr.Dataset:
    """open a specific radiosonde netcdf file and load it into an xarray dataset

    Args:
        stnm (_type_): _description_
        year (_type_): _description_
        month (_type_): _description_
        day (_type_): _description_
        hour (_type_): _description_

    Returns:
        xr.Dataset: _description_
    """
    # set the path to the netcdf directory
    ncdir = f"data/balloons_rs/stnm_{stnm}/{year}/netcdf"
    logging.info("opening nc file(s) in directory: %s", ncdir)

    # open the radiosonde data file
    fname = f"sid.{stnm}_tid.{str(year)[-2:]}{month:02d}{day:02d}.{hour:02d}00Z.nc"
    ds_sliced = extract_radiosonde_data(ncdir, fname)

    return ds_sliced

def extract_radiosonde_data(ncdir, fname):
    """extract the radiosonde data from the netcdf file (lower level function to load_radiosonde_data)"""
    ncfile = f"{ncdir}/{fname}"
    logging.info("opening file: %s", ncfile)
    ds = xr.open_dataset(ncfile)

    # remove all first and last rows in the dataset
    ds_sliced = ds.isel(index=slice(1, -1))
    # Why slice?: first entries only contain PRES & HGHT, last entries don't contain DRCT & SKNT (often)
    return ds_sliced


# # example usage
# ds = load_radiosonde_data(stnm, year, month, day, hour)
# ds

# %%
def extract_sounding_ds2mpy(ds: xr.Dataset) -> tuple:
    """get the units from the dataset Data variables units and extract the data to metpy format

    Args:
        ds (xr.Dataset): xarray dataset containing the radiosonde data with metpy units

    Returns:
        p (Quantity): pressure data
        z (Quantity): altimetric height data
        T (Quantity): temperature data
        Td (Quantity): dewpoint temperature data
        wind_speed (Quantity): total horizontal wind speed data
        u (Quantity): zonal wind component data
        v (Quantity): meridional wind component data
    """

    p_unit = ds["PRES"].units  # hPa
    z_unit = ds["HGHT"].units  # m
    T_unit = ds["TEMP"].units  # C
    Td_unit = ds["DWPT"].units  # C
    wind_d_unit = ds["DRCT"].units  # degrees
    wind_s_unit = ds["SKNT"].units  # knots

    # make sure Celsius are not interpreted as coulomb
    T_unit = T_unit.replace("C", "degC").replace("degdegC", "degC")
    Td_unit = Td_unit.replace("C", "degC").replace("degdegC", "degC")

    # extract the data to metpy format
    p = ds["PRES"].values * units[f"{p_unit}"]
    z = ds["HGHT"].values * units[f"{z_unit}"]
    T = ds["TEMP"].values * units[f"{T_unit}"]
    Td = ds["DWPT"].values * units[f"{Td_unit}"]
    # convert the wind components from knots to m/s (SI unit)
    wind_speed = (ds["SKNT"].values * units[f"{wind_s_unit}"]).to("m/s")
    wind_dir = ds["DRCT"].values * units[f"{wind_d_unit}"]
    # calculate wind components u and v
    u, v = mpcalc.wind_components(wind_speed, wind_dir)

    return p, z, T, Td, wind_speed, u, v

# # example usage
# p, z, T, Td, wind_speed, u, v = extract_sounding_ds2mpy(ds)


# %%
def metpy_skewT_simple(p, T, Td, u, v) -> plt.Figure:
    """Create the simple Skew-T plot as on the metpy github page
    /examples/Advanced_Sounding_With_Complex_Layout.html

    Args:
        p (_type_): _description_
        T (_type_): _description_
        Td (_type_): _description_
        u (_type_): _description_
        v (_type_): _description_

    Returns:
        plt.Figure: _description_
    """

    fig = plt.figure(figsize=(9, 9))
    # add_metpy_logo(fig, 90, 80, size="small")
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.55, 0.85))

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot(p, T, "r")
    skew.plot(p, Td, "g")
    skew.plot_barbs(p, u, v)

    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()

    # Create a hodograph
    ax = plt.axes((0.7, 0.75, 0.2, 0.2))
    h = Hodograph(ax, component_range=40.0)
    h.add_grid(increment=20)
    h.plot(u, v)

    # Change to adjust data limits and give it a semblance of what we want
    # skew.ax.set_adjustable("datalim")  # <- no lo sé!, but ruins the plot
    skew.ax.set_ylim(1020, 100)
    skew.ax.set_xlim(-30, 40)

    return fig


# example usage
# using the previously defined metpy data (p, T, Td, u, v)
# fig = metpy_skewT_simple(p, T, Td, u, v)


# %%
def metpy_skewT_advanced(
    stnm: str | int,
    year: str | int,
    month: str | int,
    day: str | int,
    hour: str | int,
    station_title: str,
    d: xr.Dataset,
    p: Quantity,
    z: Quantity,
    T: Quantity,
    Td: Quantity,
    wind_speed: Quantity,
    u: Quantity,
    v: Quantity,
) -> plt.Figure:
    """Make a Skew-T plot with hodograph inset, wind barbs, and some extra elements.
    Inspired by the Metpy example: Advanced_Sounding_With_Complex_Layout.html

    Args:
        stnm (str | int): station number identifier
        year (str | int): year of the sounding
        month (str | int): month of the sounding (01-12)
        day (str | int): day of the sounding (01-31)
        hour (str | int): hour of the sounding (00-23)
        station_title (str): human-readable station name
        ds (xr.Dataset): xarray dataset containing the radiosonde data
        p (Quantity): pressure data
        z (Quantity): altimetric height data
        T (Quantity): temperature data
        Td (Quantity): dewpoint temperature data
        wind_speed (Quantity): total horizontal wind speed data
        u (Quantity): zonal wind component data
        v (Quantity): meridional wind component data

    Returns:
        plt.Figure: a figure object with the vertical profile and hodograph
    """
    # STEP 1: CREATE THE SKEW-T OBJECT
    #######################################################################
    # Create a new figure. The dimensions here give a good aspect ratio
    fig = plt.figure(figsize=(18, 12))
    skew = SkewT(fig, rotation=45, rect=(0.05, 0.05, 0.50, 0.90))

    # Change to adjust data limits and give it a semblance of what we want
    # skew.ax.set_adjustable('datalim')
    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-29, 35)

    # Set some better labels than the default to increase readability
    skew.ax.set_xlabel(f"Temperature ({T.units:~P})", weight="bold")
    skew.ax.set_ylabel(f"Pressure ({p.units:~P})", weight="bold")

    # Set the facecolor of the skew-t object and the figure to white
    # fig.set_facecolor('#ffffff')
    # skew.ax.set_facecolor('#ffffff')

    # Here we can use some basic math and Python functionality to make a cool
    # shaded isotherm pattern.
    x1 = np.linspace(-100, 40, 8)
    x2 = np.linspace(-90, 50, 8)
    y = [1100, 50]
    for i in range(0, 8):
        skew.shade_area(y=y, x1=x1[i], x2=x2[i], color="gray", alpha=0.02, zorder=1)

    # STEP 2: PLOT DATA ON THE SKEW-T PROFILE
    #######################################################################
    skew.plot(p, T, "r", lw=4, label="TEMPERATURE")
    skew.plot(p, Td, "g", lw=4, label="DEWPOINT")

    # 'resample' the wind barbs for a cleaner output
    interval = np.logspace(2, 3, 40) * units.hPa
    idx = mpcalc.resample_nn_1d(p, interval)
    skew.plot_barbs(pressure=p[idx], u=u[idx], v=v[idx])

    # Add the relevant special lines native to the Skew-T Log-P diagram &
    # provide basic adjustments to linewidth and alpha to increase readability
    # first, we add a matplotlib axvline to highlight the 0-degree isotherm
    skew.ax.axvline(0 * units.degC, linestyle="--", color="blue", alpha=0.5)
    skew.plot_dry_adiabats(lw=1, alpha=0.3)
    skew.plot_moist_adiabats(lw=1, alpha=0.3)
    # skew.plot_mixing_lines(lw=1, alpha=0.3)

    # Calculate LCL height and plot as a black dot. The `0` index is selected for
    # `p`, `T`, and `Td` to lift the parcel from the surface
    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
    skew.plot(lcl_pressure, lcl_temperature, "ko", markerfacecolor="black")
    # Calculate full parcel profile and add to plot as black line
    prof = mpcalc.parcel_profile(p, T[0], Td[0]).to("degC")
    skew.plot(p, prof, "k", linewidth=2, label="SB PARCEL PATH")

    # Shade areas of CAPE and CIN
    skew.shade_cin(p, T, prof, Td, alpha=0.2, label="SBCIN")
    skew.shade_cape(p, T, prof, alpha=0.2, label="SBCAPE")

    # STEP 3: CREATE THE HODOGRAPH INSET
    #######################################################################
    # Create a hodograph object:
    hodo_ax = plt.axes((0.48, 0.45, 0.5, 0.5))
    h = Hodograph(hodo_ax, component_range=35.0)

    # Add two separate grid increments for aesthetics
    h.add_grid(increment=20, ls="-", lw=1.5, alpha=0.5)
    h.add_grid(increment=10, ls="--", lw=1, alpha=0.2)

    # For a clean hodograph inset, remove some elements
    h.ax.set_box_aspect(1)
    h.ax.set_yticklabels([])
    h.ax.set_xticklabels([])
    h.ax.set_xticks([])
    h.ax.set_yticks([])
    h.ax.set_xlabel(" ")
    h.ax.set_ylabel(" ")

    # Add tick marks to the inside of the hodograph plot
    plt.xticks(np.arange(0, 0, 1))
    plt.yticks(np.arange(0, 0, 1))
    for i in range(10, 90, 10):
        h.ax.annotate(
            str(i),
            (i, 0),
            xytext=(0, 2),
            textcoords="offset pixels",
            clip_on=True,
            fontsize=10,
            weight="bold",
            alpha=0.3,
            zorder=0,
        )
    for i in range(10, 90, 10):
        h.ax.annotate(
            str(i),
            (0, i),
            xytext=(0, 2),
            textcoords="offset pixels",
            clip_on=True,
            fontsize=10,
            weight="bold",
            alpha=0.3,
            zorder=0,
        )

    # for hodograph, crop to the tropopause at 100 hPa
    # determine the height index
    try:
        tropo_idx = np.where(p <= 100 * units.hectopascal)[0][0]
        logging.info("tropo_idx 100hPa: %s", tropo_idx)
    except IndexError:
        tropo_idx = -1
        logging.info("tropo_idx lower than 100 hPa: %s", tropo_idx)

    # determine the max height for the legend
    max_height = np.nanmax(z[:tropo_idx]).to("km")

    # plot the hodograph itself, colored by height
    h.plot_colormapped(
        u[:tropo_idx],
        v[:tropo_idx],
        c=z[:tropo_idx],
        linewidth=6,
        label=f"0-{max_height:.0f~P} WIND ({wind_speed.units})",
        cmap="viridis_r",
    )

    # STEP 4: ADD A FEW EXTRA ELEMENTS TO REALLY MAKE A NEAT PLOT
    #######################################################################
    # Add values of data to the plot for easy viewing
    #                                  xloc   yloc   xsize  ysize
    fig.patches.extend(
        [
            plt.Rectangle(
                (0.563, 0.05),
                0.334,
                0.37,
                edgecolor="black",
                facecolor="white",
                linewidth=1,
                alpha=1,
                transform=fig.transFigure,
                figure=fig,
            )
        ]
    )

    # Calculate some sounding parameters:
    #######################################################################
    # mixed layer parcel properties!
    ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=50 * units.hPa)
    ml_p, _, _ = mpcalc.mixed_parcel(p, T, Td, depth=50 * units.hPa)
    mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, prof, depth=50 * units.hPa)

    # most unstable parcel properties!
    mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=50 * units.hPa)

    # Compute Surface-based CAPE
    sbcape, sbcin = mpcalc.surface_based_cape_cin(p, T, Td)

    # Compute Bulk Shear components and then magnitude
    ubshr1, vbshr1 = mpcalc.bulk_shear(p, u, v, height=z, depth=1 * units.km)
    bshear1 = mpcalc.wind_speed(ubshr1, vbshr1)
    ubshr3, vbshr3 = mpcalc.bulk_shear(p, u, v, height=z, depth=3 * units.km)
    bshear3 = mpcalc.wind_speed(ubshr3, vbshr3)
    ubshr6, vbshr6 = mpcalc.bulk_shear(p, u, v, height=z, depth=6 * units.km)
    bshear6 = mpcalc.wind_speed(ubshr6, vbshr6)
    ubshr9, vbshr9 = mpcalc.bulk_shear(p, u, v, height=z, depth=9 * units.km)
    bshear9 = mpcalc.wind_speed(ubshr9, vbshr9)

    # add the parameters to the plot
    #######################################################################
    # Set default text properties using plt.rcParams
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = 15
    plt.rcParams["text.color"] = "black"
    plt.rcParams["figure.autolayout"] = True

    # Add some thermodynamic parameters
    #          xloc   yloc   text           alignment
    plt.figtext(0.58, 0.37, "SBCAPE:", ha="left")
    plt.figtext(0.71, 0.37, f"{sbcape:.0f~P}", color="orangered", ha="right")
    plt.figtext(0.58, 0.34, "SBCIN:", ha="left")
    plt.figtext(0.71, 0.34, f"{sbcin:.0f~P}", color="lightblue", ha="right")
    plt.figtext(0.58, 0.29, "MLCAPE:", ha="left")
    plt.figtext(0.71, 0.29, f"{mlcape:.0f~P}", color="orangered", ha="right")
    plt.figtext(0.58, 0.26, "MLCIN:", ha="left")
    plt.figtext(0.71, 0.26, f"{mlcin:.0f~P}", color="lightblue", ha="right")
    plt.figtext(0.58, 0.21, "MUCAPE:", ha="left")
    plt.figtext(0.71, 0.21, f"{mucape:.0f~P}", color="orangered", ha="right")
    plt.figtext(0.58, 0.18, "MUCIN:", ha="left")
    plt.figtext(0.71, 0.18, f"{mucin:.0f~P}", color="lightblue", ha="right")

    # add the launch location
    try:
        launch_text = f"{ds['lon'].values.all()} E / {ds['lat'].values.all()} N / {ds['elevation'].values.all()} m"
    except TypeError:
        launch_text = f"{ds['lon'].values} E / {ds['lat'].values} N / {ds['elevation'].values} m"
    plt.figtext((0.58 + 0.88) / 2, 0.13, "LAUNCH:", ha="center")
    plt.figtext((0.58 + 0.88) / 2, 0.10, f"{launch_text}", ha="center")

    # add the full station title in the bottom row
    plt.figtext((0.58 + 0.88) / 2, 0.07, f"{station_title}", ha="center")

    # Add some kinematic parameters
    #          xloc   yloc   text           alignment
    plt.figtext(0.73, 0.37, "0-1km B SHEAR: ", ha="left")
    plt.figtext(0.88, 0.37, f"{bshear1:.0f~P}", ha="right")
    plt.figtext(0.73, 0.34, "0-3km B SHEAR: ", ha="left")
    plt.figtext(0.88, 0.34, f"{bshear3:.0f~P}", ha="right")
    plt.figtext(0.73, 0.29, "0-6km B SHEAR: ", ha="left")
    plt.figtext(0.88, 0.29, f"{bshear6:.0f~P}", ha="right")
    plt.figtext(0.73, 0.26, "0-9km B SHEAR: ", ha="left")
    plt.figtext(0.88, 0.26, f"{bshear9:.0f~P}", ha="right")
    plt.figtext(0.73, 0.21, "BULK Ri: ", ha="left")
    plt.figtext(0.88, 0.21, f"{ds.attrs['Bulk Richardson Number']}", ha="right")
    plt.figtext(0.73, 0.18, "BULK Ri CAPV: ", ha="left")
    plt.figtext(
        0.88, 0.18, f"{ds.attrs['Bulk Richardson Number using CAPV']}", ha="right"
    )

    # Add legends to the skew and hodo
    #######################################################################
    skewleg = skew.ax.legend(loc="upper left", fontsize=12)
    for text in skewleg.get_texts():
        text.set_fontweight("normal")

    hodoleg = h.ax.legend(loc="upper left", fontsize=12)
    for text in hodoleg.get_texts():
        text.set_fontweight("normal")

    # add a title
    #######################################################################
    plt.figtext(
        0.45,
        0.97,
        f"{stnm} | {year}-{month:02d}-{day:02d} - {hour:02d}Z VERTICAL PROFILE",
        fontsize=20,
        ha="center",
    )

    return fig


# # # example usage
# fig = metpy_skewT_advanced(
#     stnm, year, month, day, hour, station_title, ds, p, z, T, Td, wind_speed, u, v
# )

# # save the figure
# #######################################################################
# dirname = f"figures/vertical_profile/skewT_metpy_adv/{TAG}/{stnm}"
# os.makedirs(dirname, exist_ok=True)
# figname = f"{dirname}/id{stnm}_{year}.{month:02d}.{day:02d}_{hour:02d}Z.png"

# plt.savefig(figname, dpi=100)

##############################################################################
# %%
# 6. main
if __name__ == "__main__":
    # # choose the balloon release station
    # stnm = 10868  # Munchen
    stnm = 10548  # Meinigen (150 km West of WBCI)

    if stnm == 10868:
        station_title = "München-Oberschlssheim"
    elif stnm == 10548:
        station_title = "Meiningen"

    # define lists of possible values for years, months, days, and hours
    years = [2025]
    for year in years:
        TAG = f"{year}{TAG[4:]}"
        logging.info("TAG: %s", TAG)

        ncdir = f"data/balloons_rs/stnm_{stnm}/{year}/netcdf"
        logging.info("opening nc file(s) in directory: %s \n%s", ncdir, "-" * 80)
        for fname in sorted(os.listdir(ncdir)):
            if fname.endswith(".nc"):
                # get month, day, and hour from the filename
                datestr = fname.split("tid.")[1].strip("Z.nc")
                month = int(datestr[2:4])
                day = int(datestr[4:6])
                hour = int(datestr[-4:-2])
                logging.info("opening file: %s \n  with month: %s, day: %s, hour: %s", fname, month, day, hour)

                # open the radiosonde data file
                ds = extract_radiosonde_data(ncdir, fname)

                # extract the metpy data
                p, z, T, Td, wind_speed, u, v = extract_sounding_ds2mpy(ds)

                # balloon sounding check (QC): if the data is not very complete, skip the plotting
                # QC check: if the max pressure is below 850 hPa or the min pressure is above 195 hPa
                # Ensure valid numeric data in p.magnitude
                valid_p = p.magnitude[np.isfinite(p.magnitude)]
                if valid_p.size == 0 or valid_p.max() < 850 or valid_p.min() > 195:
                    logging.warning(
                        "\n%s\n  skipping file: %s due to missing or invalid data \n%s\n",
                        "*" * 80,
                        fname,
                        "*" * 80,
                    )
                    continue

                # create the advanced Skew-T plot
                fig = metpy_skewT_advanced(
                    stnm, year, month, day, hour, station_title, ds, p, z, T, Td, wind_speed, u, v
                )

                # save the figure
                #######################################################################
                dirname = f"figures/vertical_profile/skewT_metpy_adv/{TAG}/{stnm}"
                os.makedirs(dirname, exist_ok=True)
                figname = f"{dirname}/id{stnm}_{year}.{month:02d}.{day:02d}_{hour:02d}Z.png"

                plt.savefig(figname, dpi=100)

        logging.info("finished processing all files in directory: %s \n%s", ncdir, "-" * 80)

# %%
# TODO: figure out what I want:
# AND/OR the pre-calculated Bulk Richardson numbers in their file,
# AND/OR calculate the Brunt-Vaisala frequency and plot a profile of that.
# AND/OR calculate the Richardson number and plot a profile of that.
# AND/OR to calculate the Richardson number and plot a profile of that.
# AND/OR to calculate the minimum Richardson number and check that it is the relevant one underneath the jet.


# %%
# finish with a joke
logging.info(
    "FEIERABEND! (for TAG = %s) -----> \n      %s", TAG, jokes.get_random_joke()
)
# %%
