"""
Spectral analysis functions

.. module:: spectral_analysis

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2021, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

from datetime import timedelta

import numpy as np

# from genlib import nextpow2  # replaced with obspy:
from obspy.signal.util import next_pow_2 as nextpow2
from pandas import date_range
from scipy.linalg import toeplitz
from scipy.signal import csd, spectrogram, welch
from stockwell import st
from xarray import DataArray


def compute_coherence(trace1, trace2, window, noverlap=0, oversampling=1):
    signal1 = trace1.data
    signal2 = trace2.data
    fs = trace1.stats.sampling_rate
    window_samples = int(window * fs)
    noverlap = noverlap * window_samples

    NFFT = 2 ** (oversampling - 1) * nextpow2(window_samples)
    param = dict(nperseg=window_samples, noverlap=noverlap, nfft=NFFT)
    f, Pxx = welch(signal1, fs, **param)
    _, Pyy = welch(signal2, fs, **param)
    _, Pxy = csd(signal1, signal2, fs, **param)
    Cxy = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    return (f, Cxy)


def compute_psd(
    trace,
    nperseg,
    noverlap=0,
    oversampling=1,
    scaling="density",
    window="hann",
    method="welch",
    num_freqs=512,
):
    signal = trace.data

    if method == "welch":
        fs = trace.stats.sampling_rate
        nperseg = int(nperseg * fs)
        noverlap = noverlap * nperseg

        NFFT = 2 ** (oversampling - 1) * nextpow2(nperseg)
        param = dict(
            fs=fs,
            nfft=NFFT,
            noverlap=noverlap,
            nperseg=nperseg,
            return_onesided=True,
            scaling=scaling,
            window=window,
        )
        f, Pxx = welch(signal, **param)

    elif method == "capon":
        N = len(signal)

        # Autocorrelation matrix estimation
        R = np.correlate(signal, signal, mode="full")
        R = R[N - 1 :] / N  # Take one half and normalize
        R_matrix = toeplitz(R)

        # Inverse of the autocorrelation matrix
        R_inv = np.linalg.inv(R_matrix)

        # Frequency bins
        f = np.linspace(0, 0.5, num_freqs)
        Pxx = np.zeros_like(freqs)

        for i, f in enumerate(freqs):
            # Steering vectors e^{-iwt}
            a = np.exp(-2j * np.pi * f * np.arange(N))
            Pxx[i] = 1 / np.real(np.dot(a.conj().T, np.dot(R_inv, a)))
    return (f, Pxx)


def compute_spectrogram(trace, f_min, f_max, **kwargs):
    """ """
    w = trace.data
    sample_rate = trace.stats.sampling_rate
    trace_start = trace.stats.starttime.datetime
    window_length = kwargs["window_length"]

    # Set default window length based on minimum frequency
    if window_length is None:
        window_length = 2.0 / f_min
    segment_size = int(window_length * sample_rate)

    specgram_dict = dict(
        window=kwargs["window_type"],
        nperseg=segment_size,
        nfft=kwargs["oversampling"] * segment_size,
        noverlap=float(kwargs["overlap"] / 100) * segment_size,
    )

    kwargs.pop("overlap", None)
    kwargs.pop("oversampling", None)
    kwargs.pop("window_type", None)
    kwargs.pop("window_length", None)

    (f, tf, Sxx) = spectrogram(w, sample_rate, **specgram_dict, **kwargs)

    tf_pandas = date_range(
        trace_start + timedelta(seconds=tf[0]), periods=len(tf), freq=f"{tf[1]-tf[0]}s"
    )

    Sxx = DataArray(
        data=Sxx,
        dims=["freq", "time"],
        coords={"time": (["time"], tf_pandas), "freq": (["freq"], f)},
    )
    Sxx.freq.attrs = dict(
        units="Hz", plot_units="Hz", long_name="Frequency", standard_name="freq"
    )
    Sxx.time.attrs = dict(long_name="time", standard_name="time", reftime=trace_start)
    return Sxx.sel(freq=slice(f_min, f_max))


def compute_stockwell(trace, f_min, f_max, gamma=1.0, win_type="gauss"):
    """ """
    w = trace.data
    sample_rate = trace.stats.sampling_rate
    trace_start = trace.stats.starttime.datetime

    T = (len(w) - 1) / sample_rate
    df = 1.0 / T  # sampling step in frequency domain (Hz)
    fmin_samples = int(f_min / df)
    fmax_samples = int(f_max / df)

    stock = st.st(w, fmin_samples, fmax_samples, gamma=1.0, win_type="gauss")
    f = np.arange(f_min, f_max + df, df)

    tf_pandas = date_range(
        trace_start, periods=trace.stats.npts, freq=f"{trace.stats.delta}s"
    )

    Sxx = DataArray(
        data=stock,
        dims=["freq", "time"],
        coords={"time": (["time"], tf_pandas), "freq": (["freq"], f)},
    )
    Sxx.freq.attrs = dict(
        units="Hz", plot_units="Hz", long_name="Frequency", standard_name="freq"
    )
    Sxx.time.attrs = dict(long_name="time", standard_name="time", reftime=trace_start)
    return Sxx.sel(freq=slice(f_min, f_max))


def convert_to_db(Pxx, db_ref):
    return 10 * np.log10(Pxx) - 10 * np.log10(np.power(db_ref, 2))


def get_spectrogram_maximum(Pxx):
    # Find the index of the maximum value in the flattened array
    max_index_flat = Pxx.argmax()
    # Convert this index into a 2D index
    (fmax_idx, tmax_idx) = np.unravel_index(max_index_flat, Pxx.shape)

    specgram_fmax = Pxx.freq.data[fmax_idx]
    specgram_tmax = Pxx.time.data[tmax_idx + 1]
    return (specgram_tmax, specgram_fmax)
