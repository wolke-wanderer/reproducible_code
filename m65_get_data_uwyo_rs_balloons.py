# ruff: noqa: E402  # ignore imports not at top
"""
.. module:: get_data_uwyo_rs_balloons

:author:
    Falco Bentvelsen (falco.bentvelsen@knmi.nl)

:copyright:
    2025, Falco Bentvelsen

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)


:summary:
    This module downloads radiosonde profiles from the University of Wyoming and writes them to disk.
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
import calendar
import requests
from io import StringIO
from itertools import product

# 2. import third party packages
########################################
import pandas as pd
import xarray as xr
from lxml import etree

# 3. import local packages
########################################
from logs import configure, jokes, logging


# further package setup
ipython = get_ipython()
if ipython:
    ipython.run_line_magic("config", "Completer.use_jedi = False")
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

##############################################################################
# %%
# 4. globals and constants

# define TAG for saving and logging
#######################################################################
TAG = "request_wyoming_rsb_data"
#######################################################################
configure.setup_logging(tag=TAG)  # set up logging
logging.info(jokes.get_random_joke())  # test
#######################################################################
# %%
# select the balloon launch parameters
# stnm = 10868  # Munchen
stnm = 10548  # Meinigen (150 km West of WBCI)
year = 2020
month = int("03")
region = "europe"

##############################################################################
# %%
# 5. define functions
def get_balloon_data_from_uwyo(stnm: int, region: str, year: int, month: int) -> tuple[str, str]:
    """ Download radiosonde data 
    
    Downloads radiosonde data per month from the University of Wyoming and saves it to disk per day.
    The data is saved in two formats: raw text and netcdf. (TODO: consider implementing a switch to only netcdf)

    Args:
        stnm (int): 
            Station number (e.g. 10868)
        region (str): 
            Global region (e.g. 'europe')
        year (int): 
            Year of the data (e.g. 2020)
        month (int): 
            Month of the data (e.g. 3 for March)

    Returns:
        tuple[str, str]: Directory path where data is saved and the sounding ID
    """
    # make sure a valid TIME parameter is given
    # determine how many days are in the specified month year combination
    days = calendar.monthrange(year, month)[1]

    logging.info("days in month %s-%s = %s", year, month, days)
    daterange = f"YEAR={year}&MONTH={month:02d}&FROM=0100&TO={days}12"

    ## %%
    # construct the url where the data is located
    url = f"https://weather.uwyo.edu/cgi-bin/sounding?region={region}&TYPE=TEXT%3ALIST&{daterange}&STNM={stnm}"
    # logging.info("constructed url = \n    %s", url)

    # make sure the local directory exists
    dir_path = f"data/balloons_rs/stnm_{stnm}/{year}"
    os.makedirs(dir_path, exist_ok=True)
    logging.info("dir created or pre-existing: \n    %s", dir_path)
    # TODO: consider splitting the path into year and month folders

    ## %%
    # download the data

    got_response = False
    while not got_response:
        try:
            raw_response = requests.get(url, verify=False)
            raw_response.raise_for_status()
            got_response = True
            logging.info("got response :) for \n %s \n Station = %s", daterange, stnm)
        except requests.exceptions.HTTPError as e:
            logging.error("HTTPError: %s", e)
            logging.error("retrying ...")
            got_response = False

    ## %%

    response = raw_response.text
    responses = []
    # split the responses by the header <H2>
    for i, response in enumerate(response.split("<H2>")):
        if i == 0:
            continue
        responses.append(response)

    # from the last response, remove the footer
    responses[-1] = responses[-1].split("<P>Description of the")[0]

    i = 1  # TODO: write into separate function?
    for i, response in enumerate(responses):
        # preprint header_2 tag, previously used to split the responses
        html_text = f"<H2>{responses[i]}"

        # parse the html text
        parser = etree.HTMLParser()
        tree = etree.parse(StringIO(html_text), parser)

        # get the table with the data
        table = tree.xpath("//pre")[0].text

        # get the station information and sounding indices in the header_3 tags
        sounding_info = tree.xpath("//pre")[1].text
        # put the values in each line of the station_info into a dictionary
        meta_dict = {}
        for line in sounding_info.split("\n"):
            if line == "":
                continue
            key, value = line.split(":")
            meta_dict[key.strip()] = value.strip()

        logging.info("station_info_dict = \n%s", meta_dict)

        # make a sounding_id for the file name
        sounding_id = f"sid.{meta_dict['Station number']}_tid.{meta_dict['Observation time'].replace('/', '.')}"

        ## %%
        # first split the table into lines
        # lines = table.split("\n")
        # logging.info("number of lines = %s", len(lines))

        # get the header line
        [_, header_units, data_1str] = table.split(
            "-----------------------------------------------------------------------------"
            )
        [_, header, units, _] = header_units.split("\n")
        logging.info("header = \n%s", header)
        logging.info("units = \n%s", units)

        # get the data lines
        data_lines = data_1str.split("\n")
        logging.info("number of data lines = %s", len(data_lines))

        ## %%
        # save the raw responses to disk in text file 
        # (this is an easier to comprehend duplicate of netcdf)
        os.makedirs(f"{dir_path}/raw_text", exist_ok=True)
        logging.info("saving response %s/raw_text", dir_path)

        with open(f"{dir_path}/raw_text/{sounding_id}Z.txt", "w") as f:
            f.write(f"{table}\n{'*' * 77}\n{sounding_info}")

        ## %%
        # data wrangling to netcdf:
        # convert the text table with the variables seperated by spaces into a pandas dataframe
        df = pd.read_csv(
            StringIO(table), sep="\s+", skiprows=5, header=None, names=header.split()
        )
        # logging.info("df = \n%s", df)
        # use xarray to easily convert pandas df -> netcdf
        ds = df.to_xarray()  
        # add the units to the dataset
        header_list = header.split()
        units_list = units.split()
        for var in ds.data_vars:
            var_index = header_list.index(var)
            ds[var].attrs["units"] = units_list[var_index]

        # from the sounding_info, add the metadata to the dataset
        for key, value in meta_dict.items():
            ds.attrs[key] = value

        # replace the coordinates: index -> lon, lat, time, instrument
        ds.coords["lon"] = meta_dict["Station longitude"]
        ds.coords["lat"] = meta_dict["Station latitude"]
        ds.coords["elevation"] = meta_dict["Station elevation"]
        ds.coords["instrument"] = meta_dict["Station number"]
        ds.coords["reference_time"] = pd.to_datetime(f"20{meta_dict['Observation time']} Z").isoformat()

        # add a description
        ds.attrs["description"] = "Radiosonde data provided by the University of Wyoming"
        ds.attrs["source"] = url

        ## %%
        # make sure the directory exists
        os.makedirs(f"{dir_path}/netcdf", exist_ok=True)
        logging.info("saving dataset to disk at \n    %s/netcdf/%s.nc", dir_path, sounding_id)

        # save to .nc file (overwrites existing files)
        ds.to_netcdf(f"{dir_path}/netcdf/{sounding_id}Z.nc")

    return dir_path, sounding_id

##############################################################################
# %%
# 6. main code

if __name__ == "__main__":
    for year, month in product([2025], range(3, 4)):
        logging.info("year = %s, month = %s", year, month)

        dir_path, sounding_id = get_balloon_data_from_uwyo(stnm=stnm, region=region, year=year, month=month)

    # open the last netcdf file to check if it is saved correctly
    ds = xr.open_dataset(f"{dir_path}/netcdf/{sounding_id}Z.nc")
    logging.info("testing the last netcdf file saved for the xarray content: \n%s", ds)

    logging.info("feierabend   ------>  %s", jokes.get_random_joke())

    # %%
