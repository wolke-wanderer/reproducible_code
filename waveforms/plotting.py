"""
Waveform plotting functions

.. module:: plotting

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2021, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
# from obspy import UTCDateTime, Stream
# from obspy.clients.filesystem.sds import Client as sds_client
# from obspy.clients.fdsn import Client as fdsn_client
# from obspy.clients.fdsn import RoutingClient
# import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import timedelta

import numpy as np
import xarray as xr
import re

from waveforms.waveform import compute_celerities
from waveforms.spectral import compute_spectrogram, convert_to_db#, compute_stockwell

MAX_SAMPLES_STOCKWELL = 60000

PLOTPARAMS = dict()
PLOTPARAMS['ACC'] = dict(DB_REF=1, UNIT='mm$^2$/s$^4$')
PLOTPARAMS['VEL'] = dict(DB_REF=1, UNIT='mm$^2$/s$^2$')
PLOTPARAMS['DF'] = dict(DB_REF=20e-6, UNIT='Pa$^2$')
PLOTPARAMS['DH'] = dict(DB_REF=1e-6, UNIT='Pa$^2$')

class param_object(object):
    pass

def plot_spectrogram(st, element, f_min, f_max, window_length=None,
                     window_type='hann', overlap=50, specgram='stft',
                     mode='psd', scaling='density',
                     oversampling=1, detrend='constant',
                     db_params=None, cmap='turbo', freqscale='linear',
                     celerities=None, t0=None, distance=None,
                     celerity_values=None, interpolation=None,
                     mask_threshold=None, unit=None, figsize=(8,4)):
    """
    """
    n_frames = 2
    fig, ax = plt.subplots(n_frames, 1, sharex=True, figsize=figsize,
                           gridspec_kw={'width_ratios': [1],
                                        'height_ratios': [1, 2]},
                           constrained_layout=True)

    ###########################################################################
    # Waveform
    i = 0

    # Collect waveform data
    (net, sta, loc, chan) = element.split('.')
    tr = st.select(network=net, station=sta, location=loc, channel=chan).copy()[0]
    tr.filter('bandpass', freqmin=f_min, freqmax=f_max, corners=4, zerophase=True)
    t_mpl = tr.times(type='matplotlib')
    w  = tr.data
    Fs = tr.stats.sampling_rate
    
    #ax[i].axhline(y=0, color='black', linewidth=1, linestyle=':')
    ax[i].plot(t_mpl, w, 'k', linewidth=0.7)
    ax[i].set(ylabel='Pressure (Pa)')

    ###########################################################################
    ## Spectrogram
    i += 1

    # Collect spectrogram data

    # S-transform (only if number of samples is not large enough)
    if specgram == 'stockwell' and len(w) <= MAX_SAMPLES_STOCKWELL:
        Pxx = compute_stockwell(w, Fs, f_min, f_max)
        f = [f_min, f_max]
        tf_mpl = t_mpl

    # Standard STFT
    else:
        specgram_dict = dict(window_length=window_length,
                            window_type=window_type,
                            overlap=overlap,
                            scaling=scaling,
                            oversampling=1,
                            mode=mode)
        Pxx = compute_spectrogram(tr, f_min, f_max, **specgram_dict)

    if mode == 'psd':
        if scaling == 'density':
            unit = f'{unit}$^2$/Hz'
        else:
            unit = f'{unit}$^2$'
    elif mode == 'magnitude':
        if scaling == 'density':
            unit = f'{unit}/sqrt(Hz)'
        else:
            unit = f'{unit}'        

    if db_params is not None:
        # Convert to dB re 20e-6 Pa/sqrt(Hz)
        #Pxx = 10*np.log10(Pxx/np.power(db_ref,2))
        Pxx = convert_to_db(Pxx, db_params['ref'])
        vmin = db_params['min']
        vmax = db_params['max']
        cbar_label = f'PSD [dB re {db_params["ref"]} {unit}]'
    else:
        vmin = Pxx.min()
        vmax = Pxx.max()
        cbar_label = f'PSD [{unit}]'

    if mask_threshold is not None:
        Pxx = xr.DataArray(Pxx)
        Pxx = Pxx.where(Pxx > mask_threshold*0.9)

    tf_mpl = mdates.date2num(Pxx.time)
    f_min = Pxx.freq.data[0]
    f_max = Pxx.freq.data[-1]
    extent = (tf_mpl[0], tf_mpl[-1], f_min, f_max)
    im = ax[i].imshow(Pxx, origin='lower', extent=extent, cmap=cmap,
                      interpolation=interpolation, vmin=vmin, vmax=vmax)
    ax[i].set_yscale(freqscale)
    ax[i].set(ylabel='Frequency (Hz)')
    ax[i].axis('tight')
    ax[i].set_aspect('auto')
    ax[i].set_ylim([f_min, f_max])
    #ax[i].grid()
    
    # Add colorbar
    plt.colorbar(im, ax=ax[i], label=cbar_label, extend='both')

    if celerities is not None:
        # Plot celerity axis
        try:
            celerities = compute_celerities(t0, distance, celerity_values)
            ax_celerity = ax[i].twiny()
            ax_celerity.set_xlim(t_mpl[0], t_mpl[-1])
            ax_celerity.set_xticks(mdates.epoch2num(list(celerities.values())))
            ax_celerity.set_xticklabels(list(celerities.keys()))
            ax_celerity.set_xlabel('Celerity (km/s)')
        except ValueError as e:
            print(e)
            pass

    ###########################################################################
    ## Finalizing plot

    ## Title plot
    station_id = '{network}.{station}.{location}.{channel}'.format(
        network=tr.stats.network,
        station=tr.stats.station,
        location=tr.stats.location,
        channel=tr.stats.channel
        )
    ax[0].set_title(station_id)

    return (fig, ax)


def plot_spectrogram_duo(st, elements, f_min, f_max, window_length=None,
                         window_type='hanning', overlap=50, specgram='stft',
                         mode='psd', scaling='density', 
                         oversampling=1, detrend='constant',
                         db_params=None, cmap='turbo', freqscale='linear',
                         celerities=None, t0=None, distance=None,
                         celerity_values=None, interpolation=None,
                         mask_threshold=None, unit=None, figsize=(12,4)):
    """
    """
    n_frames = 2
    fig, ax = plt.subplots(n_frames, 2, sharex=True, figsize=figsize,
                           gridspec_kw={'width_ratios': [1, 1],
                                        'height_ratios': [1, 2]},
                           constrained_layout=True)

    ###########################################################################
    # Waveform
    for idx, element in enumerate(elements):
        # Collect waveform data
        (net, sta, loc, chan) = element.split('.')
        tr = st.select(network=net, station=sta, location=loc, channel=chan).copy()[0]
        tr.filter('bandpass', freqmin=f_min, freqmax=f_max, corners=4, zerophase=True)
        t_mpl = tr.times(type='matplotlib')
        w  = tr.data
        Fs = tr.stats.sampling_rate
        
        #ax[0,idx].axhline(y=0, color='black', linewidth=0.5, linestyle=':')
        ax[0,idx].plot(t_mpl, w, 'k', linewidth=0.25)

        ###########################################################################
        ## Spectrogram

        # S-transform (only if number of samples is not large enough)
        if specgram == 'stockwell' and len(w) <= MAX_SAMPLES_STOCKWELL:
            Pxx = compute_stockwell(w, Fs, f_min, f_max)
            f = [f_min, f_max]
            tf_mpl = t_mpl

        # Standard STFT
        else:
            (tf, f, Pxx) = compute_spectrogram(w, Fs, f_min, f_max,
                                               window_length=window_length,
                                               window_type=window_type,
                                               overlap=overlap,
                                               scaling=scaling,
                                               oversampling=oversampling,
                                               detrend=detrend)
            tf_timestamp = [ tr.stats.starttime.timestamp + t for t in tf ] 
            tf_mpl = mdates.epoch2num(tf_timestamp)

        if scaling == 'density':
            unit = f'{unit}$^2$/Hz'
        else:
            unit = f'{unit}$^2$'

        if db_params is not None:
            # Convert to dB re 20e-6 Pa/sqrt(Hz)
            #Pxx = 10*np.log10(Pxx/np.power(db_ref,2))
            Pxx = 10*np.log10(Pxx/np.power(db_params['ref'],2))
            vmin = db_params['min']
            vmax = db_params['max']
            cbar_label = f'PSD [dB re {db_params["ref"]} {unit}]'
        else:
            vmin = Pxx.min()
            vmax = Pxx.max()
            cbar_label = f'PSD [{unit}]'

        if mask_threshold is not None:
            Pxx = xr.DataArray(Pxx)
            Pxx = Pxx.where(Pxx > mask_threshold*0.9)

        extent = (tf_mpl[0], tf_mpl[-1], f[0], f[-1])
        im = ax[1,idx].imshow(Pxx, origin='lower', extent=extent,
                              cmap=cmap, interpolation=interpolation,
                              vmin=vmin, vmax=vmax)
        ax[1,idx].set_yscale(freqscale)
        ax[1,idx].axis('tight')
        ax[1,idx].set_aspect('auto')
        ax[1,idx].set_ylim([f_min, f_max])
        #ax[i].grid()
        
        if celerities is not None:
            # Plot celerity axis
            try:
                celerities = compute_celerities(t0, distance, celerity_values)
                ax_celerity = ax[1,idx].twiny()
                ax_celerity.set_xlim(t_mpl[0], t_mpl[-1])
                ax_celerity.set_xticks(mdates.epoch2num(list(celerities.values())))
                ax_celerity.set_xticklabels(list(celerities.keys()))
                ax_celerity.set_xlabel('Celerity (km/s)')
            except ValueError as e:
                print(e)
                pass

        ###########################################################################
        ## Finalizing plot

        ## Title plot
        ax[0,idx].set_title(f'{element}')
    
    # Add colorbar
    plt.colorbar(im, ax=ax[1,1], label=cbar_label, extend='both')

    ax[0,0].set(ylabel='Pressure (Pa)')
    ax[1,0].set(ylabel='Frequency (Hz)')
    return (fig, ax)


def plot_spectrogram_seismoacoustic(st, elements, f_min, f_max, window_length=None,
                         window_type='hanning', overlap=50, specgram='stft',
                         scaling='density', oversampling=1, detrend='constant',
                         cmap='turbo', freqscale='linear', figsize=(12,4),
                         celerities=None, t0=None, distance=None,
                         celerity_values=None, interpolation=None,
                         mask_threshold=None):
    """
    """
    n_frames = 2
    fig, ax = plt.subplots(n_frames, 2, sharex=True, figsize=figsize,
                           gridspec_kw={'width_ratios': [1, 1],
                                        'height_ratios': [1, 2]},
                           constrained_layout=True)

    ###########################################################################
    # Waveform
    for idx, element in enumerate(elements):
        # Collect waveform data
        (net, sta, loc, chan) = element.split('.')

        scaling_factor = 1
        if re.match(r'.DF\b', chan):
            db_ref = PLOTPARAMS['DF']['DB_REF']
            unit = PLOTPARAMS['DF']['UNIT']
        elif re.match(r'.DH\b', chan):
            db_ref = PLOTPARAMS['DH']['DB_REF']
            unit = PLOTPARAMS['DH']['UNIT']
        elif re.match(r'.GZ\b', chan):
            db_ref = PLOTPARAMS['ACC']['DB_REF']
            unit = PLOTPARAMS['ACC']['UNIT']
            scaling_factor = 1e3
        elif re.match(r'.GZ\b', chan):
            db_ref = PLOTPARAMS['ACC']['DB_REF']
            unit = PLOTPARAMS['ACC']['UNIT']
            scaling_factor = 1e3
        else:
            db_ref = 1
            unit = 'arb.'

        if scaling == 'density':
            unit = f'{unit}/Hz'
        else:
            unit = f'{unit}'

        tr = st.select(network=net, station=sta, location=loc, channel=chan).copy()[0]
        tr.filter('bandpass', freqmin=f_min, freqmax=f_max, corners=4, zerophase=True)
        t_mpl = tr.times(type='matplotlib')
        w  = tr.data * scaling_factor
        Fs = tr.stats.sampling_rate
        
        ax[0,idx].axhline(y=0, color='black', linewidth=1, linestyle=':')
        ax[0,idx].plot(t_mpl, w, 'k', linewidth=2)

        ###########################################################################
        ## Spectrogram

        # S-transform (only if number of samples is not large enough)
        if specgram == 'stockwell' and len(w) <= MAX_SAMPLES_STOCKWELL:
            Pxx = compute_stockwell(w, Fs, f_min, f_max)
            f = [f_min, f_max]
            tf_mpl = t_mpl

        # Standard STFT
        else:
            (tf, f, Pxx) = compute_spectrogram(w, Fs, f_min, f_max,
                                               window_length=window_length,
                                               window_type=window_type,
                                               overlap=overlap,
                                               scaling=scaling,
                                               oversampling=oversampling,
                                               detrend=detrend)
            tf_timestamp = [ tr.stats.starttime.timestamp + t for t in tf ] 
            tf_mpl = mdates.epoch2num(tf_timestamp)

        # Convert to dB
        Pxx = 10*np.log10(Pxx/np.power(db_ref,2))
        vmin = Pxx.min()
        vmax = Pxx.max()

        if mask_threshold is not None:
            Pxx = xr.DataArray(Pxx)
            Pxx = Pxx.where(Pxx > mask_threshold*0.9)

        extent = (tf_mpl[0], tf_mpl[-1], f[0], f[-1])
        im = ax[1,idx].imshow(Pxx, origin='lower', extent=extent,
                              cmap=cmap, interpolation=interpolation,
                              vmin=vmin, vmax=vmax)
        ax[1,idx].set_yscale(freqscale)
        ax[1,idx].axis('tight')
        ax[1,idx].set_aspect('auto')
        ax[1,idx].set_ylim([f_min, f_max])

        # Add colorbar
        axins = inset_axes(ax[1,idx],
                           width="100%",  
                           height="5%",
                           loc='lower center',
                           borderpad=-7)
        plt.colorbar(im, cax=axins, extend='both',
                     orientation='horizontal')

        if celerities is not None:
            # Plot celerity axis
            try:
                celerities = compute_celerities(t0, distance, celerity_values)
                ax_celerity = ax[1,idx].twiny()
                ax_celerity.set_xlim(t_mpl[0], t_mpl[-1])
                ax_celerity.set_xticks(mdates.epoch2num(list(celerities.values())))
                ax_celerity.set_xticklabels(list(celerities.keys()))
                ax_celerity.set_xlabel('Celerity (km/s)')
            except ValueError as e:
                print(e)
                pass

        ## Title plot
        ax[0,idx].set_title(f'{tr.id}')

    return (fig, ax)

def custom_scatter(x, y, ax=None, **kwargs):
    """
    Wrapper to pyplot's :meth:`~matplotlib.axes.Axes.scatter`.

    Plot symbol edges behind symbols and sort symbols by the ``c``
    argument so that higher values plot above lower values.

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data

    ax : :class:`~matplotlib.axes.Axes`
        Axes to plot to. If ``None``, plotting is done to the active axes.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Other Parameters
    ----------------
    **kwargs : `~matplotlib.axes.Axes.scatter` and
    `~matplotlib.collections.Collection` properties
        Other keyword arguments are passed to
        `~matplotlib.axes.Axes.scatter` and then
        `~matplotlib.collections.Collection`

    """
    ax = ax or plt.gca()

    # make sure x and y are flattened numpy arrays, helps with sorting
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    try:
        ec = kwargs.pop('edgecolor')
    except KeyError:
        ec = 'k'

    try:  # grab the color-coding data
        c = kwargs.pop('c')
        if isinstance(c, str):  # c in not an array
            raise KeyError

        # convert to numpy array
        c = np.asarray(c).ravel()

        # sort
        _sort_ = c.argsort()
        x, y, c = x.ravel()[_sort_], y.ravel()[_sort_], c.ravel()[_sort_]

    except KeyError:
        c = 'k'

    # scatter values lower than `vmin` with a gray edgecolor
    try:
        idx = c < kwargs.get('vmin')
        ax.scatter(x[idx], y[idx], c='none', edgecolor='0.5', **kwargs)
        ax.scatter(x[idx], y[idx], c='w', edgecolor='none', **kwargs)

        ax.scatter(x[~idx], y[~idx], c='none', edgecolor=ec, **kwargs)
        sp = ax.scatter(x[~idx], y[~idx], c=c[~idx], edgecolor='none',
                        **kwargs)
    except ValueError:
        # c is not a sequence or vmin is not defined
        ax.scatter(x, y, c='none', edgecolor=ec, **kwargs)
        sp = ax.scatter(x, y, c=c, edgecolor='none',
                        **kwargs)
    return sp
