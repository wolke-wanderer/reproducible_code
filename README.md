## A descriptive project title
What if we go up? Estimating atmospheric profiles from balloon data

## Motivation (why the project exists)
We want to know what the weather will be tomorrow, so we estimate atmospheric parameters from so-called radiosonde balloons that are released worldwide by meteorological institutes. We do a case study in Germany to compare with a [new wind estimation method being developed at KNMI](https://www.knmi.nl/research/seismology-acoustics/projects/tracking-jet-stream-winds-mso-project). 

## Setup
- install the packages in a virtual environment from requirements.txt
Installation

Create a clone, or copy of the xcorr repository in an empty directory

```bash
git clone https://github.com/wolke-wanderer/reproducible_code
```

Run git pull to update the local repository to this master repository.

Required are Python3.12 or higher and the modules to install from requirements.txt (below)

Install required packages in a virtual environment via pip in a terminal (BASH/ZSH):
```bash
# Create a new virtual environment
python3 -m venv reproducible_venv

# Activate the virtual environment
source reproducible_venv/bin/activate # On Windows use: reproducible_venv\Scripts\activate

# Install the required packages from requirements.txt
pip install -r requirements.txt
```
Run the files, this may be done from terminal or your IDE

- Run the [m65_get_data_uwyo_rs_balloons.py](./m65_get_data_uwyo_rs_balloons.py) file to scrape balloon data from the web for a Bavarian station for march 2025
    - There's also example data for 31st of march if this doesn't work, so this step may be skipped.
```bash
python3 m65_get_data_uwyo_rs_balloons.py
```
- Run the [m66_wip_create_profiles_rsb.py](./m66_wip_create_profiles_rsb.py) file to make vertical profiles of what is measured in the balloon
```bash
python3 m66_wip_create_profiles_rsb.py
```

Congratiolations! Your generated balloon profiles are in a subfolder of ./figures/ (keep in diving in)

## Copy-pastable quick start code example
- an example balloon sounding can be found in folder [./example_data](./example_data/raw_text/sid.10548_tid.250403.0000Z.txt), both in .txt (for easy reading) and .netcdf (standard format used in computations)

## Link or instructions for contributing
- this is a demonstration project, publicly available at [github](https://github.com/wolke-wanderer/reproducible_code), pull requests are appreciated! :D
- no active maintenance is done

## Recommended citation
TODO...
