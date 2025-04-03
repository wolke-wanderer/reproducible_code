"""
Fireball analysis helper functions.

References:
 * Edwards, W. N., Brown, P. G., & ReVelle, D. O. (2006). 
    Estimates of meteoroid kinetic energies from observations of infrasonic airwaves. 
    Journal of Atmospheric and Solar-Terrestrial Physics, 68(10), 1136-1160. ISO 690	
    https://doi.org/10.1016/j.jastp.2006.02.010

* Ens, T. A., Brown, P. G., Edwards, W. N., & Silber, E. A. (2012).
   Infrasound production by bolides: A global statistical study.
   Journal of Atmospheric and Solar-Terrestrial Physics, 80, 208-229.
   https://doi.org/10.1016/j.jastp.2012.01.018

.. module:: fireball

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2024, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
import numpy as np
from obspy.signal.filter import envelope

def compute_bandpass(freq, SNR_dB,
                     dB_criterion=1, min_bandwidth=1):
    # Step 1: Identify values above noise level
    above_noise = SNR_dB > dB_criterion

    # Step 2: Find changes in the consecutive condition
    change_points = np.diff(above_noise.astype(int))

    # Start indices: Where it changes from 0 to 1
    start_indices = np.where(change_points == 1)[0] + 1

    # End indices: Where it changes from 1 to 0
    end_indices = np.where(change_points == -1)[0]

    # If the array starts with values above 1, prepend a start index
    if above_noise[0]:
        start_indices = np.insert(start_indices, 0, 0)

    # If the array ends with values above 1, append an end index
    if above_noise[-1]:
        end_indices = np.append(end_indices, len(above_noise) - 1)

    bandwidth = freq[end_indices] - freq[start_indices]
    idx = np.where(bandwidth > min_bandwidth)[0][0]
    f_min = freq[start_indices][idx]
    f_max = freq[end_indices][idx]
    return (f_min, f_max)

def find_zero_crossings(arr):
    # Calculate the difference in signs between consecutive elements
    sign_changes = np.diff(np.sign(arr))
    # Find indices where the sign changes
    crossings = np.where(sign_changes != 0)[0]
    return crossings

def find_nearest_crossings(arr, reference_index):
    number_crossings = 4
    # Find all zero crossings
    crossings = find_zero_crossings(arr)
    # Calculate distances from the middle
    distances = np.abs(crossings - reference_index)
    # Find indices of four nearest crossings
    nearest_indices = np.argsort(distances)[:number_crossings]
    # Get the actual indices of the nearest crossings
    nearest_crossings = crossings[nearest_indices]
    return np.sort(nearest_crossings)

def get_period_timedomain(trace, nearest_crossings):
    time_samples = trace.times()[nearest_crossings]
    #[ print(item) for item in time_samples ]

    dt1 = (time_samples[2] - time_samples[0])
    dt2 = (time_samples[3] - time_samples[1])
    return 0.5*(dt1 + dt2)

def get_amplitude_timedomain(trace, nearest_crossings):
    w = trace.data
    w_env = envelope(w)
    indices = np.arange(nearest_crossings[0],
                        nearest_crossings[-1])
    w_main_signal = w[indices]

    amp_env = np.max(w_env)
    amp_p2p = np.max(w_main_signal) - np.min(w_main_signal)
    return (amp_env, amp_p2p)
