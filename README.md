## A descriptive project title
What if we go up? Estimating atmospheric profiles from balloon data

## Motivation (why the project exists)
We want to know what the weather will be tomorrow, so we estimate atmospheric parameters from so-called radiosonde balloons that are released worldwide by meteorological institutes. We do a case study in Germany to compare with a [new wind estimation method being developed at KNMI](https://www.knmi.nl/research/seismology-acoustics/projects/tracking-jet-stream-winds-mso-project). 

## How to setup
- install the packages in a virtual environment from requirements.txt
- Run the [m65_get_data_uwyo_rs_balloons.py](./m65_get_data_uwyo_rs_balloons.py) file to scrape balloon data from the web for a Bavarian station for the first months of 2025
- Run the [m66_wip_create_profiles_rsb.py](./m66_wip_create_profiles_rsb.py) file to make vertical profiles

## Copy-pastable quick start code example
- an example balloon sounding can be found in folder [./example_data](./example_data/raw_text/sid.10548_tid.250403.0000Z.txt), both in .txt (for easy reading) and .netcdf (standard format used in computations)

## Link or instructions for contributing
- this is a demonstration project, publicly available at [github](https://github.com/wolke-wanderer/reproducible_code), pull requests are appreciated! :D
- no active maintenance is done

## Recommended citation
TBD
