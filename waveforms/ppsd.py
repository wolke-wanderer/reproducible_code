"""
Probablistic Power Spectral Density (PPSD) interface functions

.. module:: ppsd

:author:
    Jelle Assink (jelle.assink@knmi.nl)
    Edited by Falco Bentvelsen 2024

:copyright:
    2021, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import FuncFormatter
from obspy.signal import PPSD
from obspy.signal.spectral_estimation import NOISE_MODEL_FILE

from waveforms.waveform import stream_unique_seed_id

IDC2010 = {}
IDC2010["path"] = (
    "/Users/falco/surf/drive/tutorials/ppsd_gw_py311/waveforms/noisemodel/"
)
IDC2010["L"] = {}
IDC2010["H"] = {}
IDC2010["L"]["seismo"] = "IDC2010_LS.txt"
IDC2010["L"]["hydro"] = "IDC2010_LH.txt"
IDC2010["L"]["infra"] = "IDC2010_LI.txt"
IDC2010["H"]["seismo"] = "IDC2010_HS.txt"
IDC2010["H"]["hydro"] = "IDC2010_HH.txt"
IDC2010["H"]["infra"] = "IDC2010_HI.txt"

WN19 = {}
WN19["path"] = IDC2010["path"]
WN19["LNM"] = "wolinmcnamara_Perm_LNM.mod"
WN19["HNM"] = "wolinmcnamara_HNM.mod"

# Global dB params
db_params = dict(seismo=dict(), hydro=dict(), infra=dict())
db_params["seismo"] = dict(
    ref=1,
    label="1 m$^2$/s$^4$/Hz",
    min=-180,
    max=-50.0,
    bin=1,
    psd_min=-120,
    psd_max=-70,
)
db_params["hydro"] = dict(
    ref=1e-6,
    label="1 $\mu$Pa$^2$/Hz",
    min=40,
    max=190.0,
    bin=1,
    psd_min=90,
    psd_max=130,
)
db_params["infra"] = dict(
    ref=20e-6,
    label="20 $\mu$Pa$^2$/Hz",
    min=-10,
    max=135.0,
    bin=1,
    psd_min=25,
    psd_max=100,
)
db_params["baro"] = dict(
    ref=1, label="1 Pa$^2$/Hz", min=0, max=100, bin=1, psd_min=0, psd_max=100
)


def get_channel_technology(channel):
    """
    Helper function to return technology type based on SEED channel
    """
    (_, instr_code, orient_code) = channel
    if instr_code == "D":
        if orient_code == "F":
            technology = "infra"
        elif orient_code == "A":
            technology = "baro"
        elif orient_code == "H":
            technology = "hydro"
    else:
        technology = "seismo"
    return technology


def get_ppsd_mean(ppsd, starttime=None, endtime=None):
    print(" -> Computing mean PSD")
    ppsd.calculate_histogram(starttime=starttime, endtime=endtime)
    (_p, ppsd_mean) = ppsd.get_mean()
    ppsd_mean = np.array(ppsd_mean)
    ppsd_mean_freq = np.array(1.0 / _p)
    ppsd_mean[ppsd_mean == 134.5] = np.nan
    return (ppsd_mean_freq, ppsd_mean)


def get_ppsd_percentile(ppsd, percentile):
    print(f" -> Computing {percentile}% PSD")
    (_p, ppsd_percentile) = ppsd.get_percentile(percentile=percentile)
    ppsd_percentile = np.array(ppsd_percentile)
    ppsd_percentile_freq = np.array(1.0 / _p)
    return (ppsd_percentile_freq, ppsd_percentile)


def idc_nm(technology="infra", **kwargs):
    """
    CTBTO / IDC (2010) Noise model, David Brown (PAGEOPH)
    Low / High noise model curves

    Returns low and high noise model values in decibel (dB) relative
    to the community standard:

    - seismo :     1 m/s**2
    - hydro  :  1e-6 Pa
    - infra  : 20e-6 Pa
    """
    nm_path = IDC2010["path"]
    fid_lnm = "{path}/{file}".format(path=nm_path, file=IDC2010["L"][technology])
    fid_hnm = "{path}/{file}".format(path=nm_path, file=IDC2010["H"][technology])
    (_lnm_freq, _lnm) = np.loadtxt(fid_lnm, skiprows=1, unpack=True)
    (_hnm_freq, _hnm) = np.loadtxt(fid_hnm, skiprows=1, unpack=True)

    # Convert to Hz, convert to dB re 1 [arb]
    nm = dict(low=dict(), high=dict())
    nm["low"]["freq"] = 10**_lnm_freq
    nm["high"]["freq"] = 10**_hnm_freq

    # Convert PSD values that are reported in IDC noise model files
    if technology == "infra":
        # Convert to dB relative to 20e-6 Pa/Hz
        nm["low"]["values"] = 10 * _lnm - 20 * np.log10(20e-6)
        nm["high"]["values"] = 10 * _hnm - 20 * np.log10(20e-6)
    else:
        # For 'hydro' and 'seismo' the units are already in dB
        nm["low"]["values"] = _lnm
        nm["high"]["values"] = _hnm

    return nm


def nextpow2(i):
    n = 2
    while n < i:
        n = n * 2
    return n


def petersen93_nm(**kwargs):
    """
    Community standard model
    """
    petersen93 = np.load(NOISE_MODEL_FILE)
    nm = dict(low=dict(), high=dict())
    nm["low"]["freq"] = 1.0 / petersen93["model_periods"]
    nm["low"]["values"] = petersen93["low_noise"]
    nm["high"]["freq"] = 1.0 / petersen93["model_periods"]
    nm["high"]["values"] = petersen93["high_noise"]
    return nm


def plot_idc_nm(seismo=True, hydro=True, infra=True):
    n_shi = sum(locals().values())
    formatter = FuncFormatter(lambda y, _: "{:.16g}".format(y))

    # _freq = np.arange(0.01,10.0,0.001)

    fig, ax = plt.subplots(n_shi, sharex=True, figsize=(7, 5), squeeze=False)
    # add an axes, lower left corner in [0.83, 0.1]
    # measured in figure coordinate with axes width 0.02 and height 0.8
    fig.subplots_adjust(
        bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.02, hspace=0.2
    )

    i = 0
    if seismo:
        nm = idc_nm(technology="seismo")
        ax[i, 0].plot(nm["low"]["freq"], nm["low"]["values"], "C2", label="Seismology")
        ax[i, 0].plot(nm["high"]["freq"], nm["high"]["values"], "C2")
        ax[i, 0].set(ylabel="dB (m$^{2}$/s$^{-4}$/Hz)")
        i += 1

    if hydro:
        nm = idc_nm(technology="hydro")
        ax[i, 0].plot(
            nm["low"]["freq"], nm["low"]["values"], "xkcd:azure", label="Hydroacoustics"
        )
        ax[i, 0].plot(nm["low"]["freq"], nm["high"]["values"], "xkcd:azure")
        ax[i, 0].set(ylabel="dB ($\mu$Pa$^{2}$/Hz)")
        i += 1

    if infra:
        nm = idc_nm(technology="infra")
        ax[i, 0].plot(nm["low"]["freq"], nm["low"]["values"], "C1", label="Infrasound")
        ax[i, 0].plot(nm["high"]["freq"], nm["high"]["values"], "C1")
        ax[i, 0].set(xlabel="Frequency (Hz)", ylabel="dB (20 $\mu$Pa$^{2}$/Hz)")

    for i in range(0, n_shi):
        ax[i, 0].legend()
        ax[i, 0].set_xscale("log")
        ax[i, 0].grid(which="both", ls="-", color="0.65")
        ax[i, 0].xaxis.set_major_formatter(formatter)

    fig.align_ylabels(ax)
    plt.show()
    return fig


def fit_karman_spectrum(freq, U_av, C=4e5, lamda=5.0):
    """
    Fit to Von-Karman spectrum -- C and lambda are free parameters
    """
    C = 4e5
    lamda = 5.0
    k = 2 * np.pi * _freq / (0.7 * U_av)
    F = C / (1 + (k * lamda) ** 2) ** (5 / 6)
    return F


def plot_ppsd(
    ppsd,
    starttime=None,
    endtime=None,
    prob_min=None,
    prob_max=None,
    cmap="magma_r",
    freq_scale="log",
    show_mean=False,
    noise_model="idc",
    db_label=None,
    percentile=None,
    figsize=None,
):

    technology = get_channel_technology(ppsd.channel)

    if noise_model == "idc":
        # Get CTBT IDC Low-noise / High-noise models
        if technology == "baro":
            nm = idc_nm("infra")
            nm["low"]["values"] += 20 * np.log10(20e-6)
            nm["high"]["values"] += 20 * np.log10(20e-6)
        else:
            nm = idc_nm(technology)
    elif noise_model == "wn19":
        nm = wn19_nm()
    elif noise_model == "petersen93":
        nm = petersen93_nm()
    else:
        nm = None

    ppsd.calculate_histogram(starttime=starttime, endtime=endtime)
    # ppsd_plot_opts = dict(show_coverage=False,
    #                         show_percentiles=False,
    #                         percentiles=[5,95],
    #                         show_mean=False,
    #                         cmap=get_cmap('inferno_r'),
    #                         xaxis_frequency=True,
    #                         period_lim=(freq['min'], freq['max']),
    #                         show_noise_models=False,
    #                         show=False)
    # fig = ppsd.plot(**ppsd_plot_opts)
    # ax = fig.gca()

    # ax.plot(_freq, 20*np.log10(F), color='C0', lw=2,
    #         path_effects=[pe.Stroke(linewidth=4, foreground='white', alpha=0.8), pe.Normal()])

    # Prepare data
    data = ppsd.current_histogram * 100.0 / (ppsd.current_histogram_count or 1)
    data = xr.DataArray(data)
    q2 = float(data.quantile(0.02))
    q98 = float(data.quantile(0.98))

    if prob_min:
        data = data.where(data > prob_min)
        vmin = prob_min
    else:
        vmin = q2

    if prob_max:
        vmax = prob_max
    else:
        vmax = q98

    xedges = 1.0 / ppsd.period_xedges
    yedges = ppsd.db_bin_edges
    XV, YV = np.meshgrid(xedges, yedges)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ppsd_plot = ax.pcolormesh(
        XV, YV, data.T, cmap=cmap, vmin=vmin, vmax=vmax, zorder=-1, shading="auto"
    )

    cb_label = f"Probability (%)"
    cb = plt.colorbar(ppsd_plot, ax=ax, label=cb_label)

    if nm:
        ax.plot(nm["low"]["freq"], nm["low"]["values"], "0.4", linewidth=2, zorder=10)
        ax.plot(nm["high"]["freq"], nm["high"]["values"], "0.4", linewidth=2, zorder=10)

        if show_mean:
            (ppsd_mean_freq, ppsd_mean) = get_ppsd_mean(ppsd)
            ax.plot(
                ppsd_mean_freq,
                ppsd_mean,
                color="C0",
                lw=2,
                alpha=0.5,
                path_effects=[
                    pe.Stroke(linewidth=4, foreground="white", alpha=0.5),
                    pe.Normal(),
                ],
            )

    if percentile:
        (ppsd_percentile_freq, ppsd_percentile) = get_ppsd_percentile(ppsd, percentile)
        ax.plot(
            ppsd_percentile_freq,
            ppsd_percentile,
            color="black",
            lw=2,
            alpha=0.5,
            path_effects=[
                pe.Stroke(linewidth=4, foreground="white", alpha=0.5),
                pe.Normal(),
            ],
        )

    if db_label is None:
        db_label = db_params[technology]["label"]
    ax.set_ylabel(f"PSD (dB re {db_label})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xscale(freq_scale)
    return fig


def plot_ppsd_mean(
    ppsd,
    starttime=None,
    endtime=None,
    freq_scale="log",
    noise_model="idc",
    db_label=None,
):
    fig, ax = plt.subplots(1, 1)

    technology = get_channel_technology(ppsd.channel)
    if noise_model == "idc":
        # Get CTBT IDC Low-noise / High-noise models
        nm = idc_nm(technology)
    elif noise_model == "wn19":
        nm = wn19_nm()
    elif noise_model == "petersen93":
        nm = petersen93_nm()

    ax.plot(nm["low"]["freq"], nm["low"]["values"], "0.4", linewidth=2, zorder=10)
    ax.plot(nm["high"]["freq"], nm["high"]["values"], "0.4", linewidth=2, zorder=10)

    (ppsd_mean_freq, ppsd_mean) = get_ppsd_mean(ppsd)
    ax.plot(
        ppsd_mean_freq,
        ppsd_mean,
        color="C0",
        lw=2,
        alpha=0.5,
        path_effects=[
            pe.Stroke(linewidth=4, foreground="white", alpha=0.5),
            pe.Normal(),
        ],
    )

    if db_label is None:
        db_label = db_params[technology]["label"]
    ax.set_ylabel(f"PSD (dB re {db_label})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xscale(freq_scale)
    return fig


def plot_ppsd_spectrogram(
    ppsd, freq_axis_scale, ax, cmap="turbo", vmin=None, vmax=None, db_label=None, ylabel="Frequency (Hz)"
):
    """
    Plot the temporal evolution of the PSD in a spectrogram-like plot.

    .. note::
        For example plots see the :ref:`Obspy Gallery <gallery>`.

    :type cmap: :class:`matplotlib.colors.Colormap`
    :param cmap: Specify a custom colormap instance. If not specified, then
        the default ObsPy sequential colormap is used.
    """
    technology = get_channel_technology(ppsd.channel)

    quadmeshes = []
    yedges = ppsd.period_xedges

    for times, psds in ppsd._get_gapless_psd():
        xedges = [t.matplotlib_date for t in times] + [
            (times[-1] + ppsd.step).matplotlib_date
        ]
        meshgrid_x, meshgrid_y = np.meshgrid(xedges, yedges)
        data = np.array(psds).T

        quadmesh = ax.pcolormesh(
            meshgrid_x, 1.0 / meshgrid_y, data, cmap=cmap, zorder=-1
        )
        quadmeshes.append(quadmesh)

    if vmin is None:
        vmin = db_params[technology]["psd_min"]
    if vmax is None:
        vmax = db_params[technology]["psd_max"]
    clim = (vmin, vmax)

    for quadmesh in quadmeshes:
        quadmesh.set_clim(*clim)

    cb = plt.colorbar(quadmesh, ax=ax, extend="both")

    if db_label is None:
        db_label = db_params[technology]["label"]
    cb.ax.set_ylabel(f"PSD [dB re {db_label}]")
    ax.set_ylabel(ylabel)

    ax.set_yscale(freq_axis_scale)
    ax.set_xlim(
        ppsd.times_processed[0].matplotlib_date,
        (ppsd.times_processed[-1] + ppsd.step).matplotlib_date,
    )
    ax.set_ylim(yedges[0], yedges[-1])
    try:
        ax.set_facecolor("0.8")
    # mpl <2 has different API for setting Axes background color
    except AttributeError:
        ax.set_axis_bgcolor("0.8")

    return


def return_utcdays(ts, te):
    """
    Returns the UTCDateTime days between two timestamps
    """
    ts = ts.replace(hour=0, minute=0, second=0)
    te = te.replace(hour=0, minute=0, second=0) + 24 * 3600
    ndays = int(np.ceil((te - ts) / 86400))
    utc_days = []
    for i in range(0, ndays):
        utc_days.append(ts)
        ts += 24 * 3600
    return utc_days


def stream2ppsd(
    st,
    inventory,
    starttime,
    endtime,
    special_handling=None,
    overlap=0.5,
    wlen=3600,
    octave_smoothing=1,
    octave_step=0.125,
    db_params_local=None,
    freq_params_local=None,
):
    """
    Routine to convert trace to PPSD element
    """
    n_seed_id = len(stream_unique_seed_id(st))
    if n_seed_id > 1:
        raise TypeError("More than one unique Seed ID present in stream.")

    trace = st[0]
    (_, instr_code, orient_code) = trace.stats.channel

    if freq_params_local:
        period_max = 1.0 / freq_params_local["min"]
        period_min = 1.0 / freq_params_local["max"]
    else:
        period_max = 10 / wlen
        period_min = 2.0 / trace.stats.sampling_rate

    # In case of pressure sensor, process PPSD using 'hydrophone' tag
    # to make sure that spectra are not differentiated
    if instr_code == "D":
        special_handling = "hydrophone"

    # dB binning parameters
    # Can be defined, otherwise default values are determiend by SEED channel
    if db_params_local:
        db = db_params_local
    else:
        technology = get_channel_technology(trace.stats.channel)
        db = db_params[technology]
        print(db)

    ppsd = PPSD(
        trace.stats,
        metadata=inventory,
        ppsd_length=wlen,
        overlap=overlap,
        period_smoothing_width_octaves=octave_smoothing,
        period_step_octaves=octave_step,
        special_handling=special_handling,
        db_bins=(db["min"], db["max"], db["bin"]),
        period_limits=(period_min, period_max),
    )

    # Add all traces in stream
    for ms in st:
        print(" -> Processing [ {:15s} ]".format(ms.id))
        try:
            ms.data = ms.data.astype("float64") / db["ref"]
            ppsd.add(ms)
        except Exception as e:
            pass

    return ppsd


def wn19_nm(**kwargs):
    """
    Returns low and high noise seismic model values in decibel (dB) relative
    to community standard of 1 m/s**2
    """
    fid_lnm = f"{WN19['path']}/{WN19['LNM']}"
    fid_hnm = f"{WN19['path']}/{WN19['HNM']}"
    (_lnm_period, _lnm) = np.loadtxt(fid_lnm, skiprows=0, unpack=True)
    (_hnm_period, _hnm) = np.loadtxt(fid_hnm, skiprows=0, unpack=True)

    nm = dict(low=dict(), high=dict())
    nm["low"]["freq"] = 1.0 / _lnm_period
    nm["low"]["values"] = _lnm
    nm["high"]["freq"] = 1.0 / _hnm_period
    nm["high"]["values"] = _hnm
    return nm
