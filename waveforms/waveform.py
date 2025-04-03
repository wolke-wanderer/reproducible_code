"""
Waveform interface functions

.. module:: waveform

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2021, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
import os

from obspy import UTCDateTime, Stream
from obspy.clients.filesystem.sds import Client as sds_client
from obspy.clients.fdsn import Client as fdsn_client
from obspy.clients.fdsn import RoutingClient

from waveforms.definitions import default_celerities

def compute_celerities(t0, distance_km, celerities=None):
    """
    Compute timestamp for celerities

    t0 - Obspy UTCDateTime object
    distance_km - Distance to epicenter (km)
    celerities - tuple or list of celerities (km/s) to be considered

    returns: dictionary with celerity labels and timestamps
    """
    if celerities is None:
        celerity_keys = default_celerities
    else:
        celerity_keys = celerities

    celerities = dict()
    for celerity in celerity_keys:
        celerity_time = (t0 + float(distance_km / celerity)).datetime
        celerities[str(round(celerity, 2))] = celerity_time
    return celerities

def get_data(data_source, inventory, starttime, endtime, local_storage=False,
             sds_root='data'):
    """
    source type options:

    local - sds wavestructure
    fdsn - typical dataselect
    routing - 'iris-federator' or 'eida-routing'
    """
    [(key, value)] = data_source.items()

    stream = Stream()

    if key == 'local':
        cl = sds_client(value)
    elif key == 'fdsn':
        cl = fdsn_client(value)
    elif key == 'routing':
        cl = RoutingClient(value)

    for net in inventory:
        for sta in net:
            for cha in sta:
                try:
                    st = cl.get_waveforms(network=net.code,
                                          station=sta.code,
                                          location=cha.location_code,
                                          channel=cha.code,
                                          starttime=starttime,
                                          endtime=endtime)
                    
                    stream += st
                except Exception as e:
                    seed_id = (f'{net.code}.{sta.code}.'
                               f'{cha.location_code}.{cha.code}')
                    print(e)
                    print(f'No data available for {seed_id}')
                    pass

    for ms in stream:
        samp_rate = ms.stats.delta
        ms.trim(starttime, endtime-samp_rate)

    if local_storage:
        stream2sds(stream, sds_root)

    return stream

def get_stream_juldays(st):
    """
    """
    st.sort()
    t0 = st[0].stats.starttime
    t1 = st[-1].stats.endtime
    julday_fmt = '%Y%j'
    yj0 = int(t0.strftime(julday_fmt))
    yj1 = int(t1.strftime(julday_fmt))
    return range(yj0, yj1+1)

def get_stream_elements(st):
    """
    """
    elements = {}
    for ms in st:
        elements[ms.id] = ms.stats
    return elements

def remove_response(st, inv, f_min=1e-2, f_max='nyquist', output='VEL', method='response'):
    for ms in st:
        (net, sta, loc, cha) = ms.id.split('.')
        invs = inv.select(network=net, station=sta,
                          location=loc, channel=cha)
        if method == 'response':
            if f_max == 'nyquist':
                fn = ms.stats.sampling_rate / 2
                f_max = 0.9*fn
            pre_filt = [0.9*f_min, f_min, f_max, 1.1*f_max]
            ms.remove_response(inventory=invs, 
                               pre_filt=pre_filt,
                               output=output)
        elif method == 'sensitivity':
            sens = invs[0][0][0].response.instrument_sensitivity.value
            ms.data = ms.data.astype('float64') / sens

    return st

def stream2sds(stream, sds_root, reclen=512, wave_format='mseed', **kwargs):
    """
    Write ObsPy Stream out to SDS data structure
    """
    juldays = get_stream_juldays(stream)
    elements = get_stream_elements(stream)

    for day in juldays:
        t0 = UTCDateTime(str(day))
        t1 = t0 + 24*3600
        st = stream.copy().trim(t0, t1)

        for (ele, stats) in elements.items():
            st_ele = st.select(id=ele)
            st_ele.trim(t0, t1-stats['delta'])

            if len(st_ele) > 0:
                out_file = f'{ele}.D.{t0.year:4d}.{t0.julday:03d}'
                out_path = (f'{sds_root}/{t0.year:4d}/'
                            f'{stats["network"]}/'
                            f'{stats["station"]}/'
                            f'{stats["channel"]}.D')
                output = f'{out_path}/{out_file}'

                if os.path.isdir(out_path) is False:
                    try:
                        os.makedirs(out_path)
                    except OSError as e:
                        print(e.args)

                msg = f'-> Writing {wave_format} file [ {output} ] ... '
                print(msg, end='')
                try:
                    st_ele.write(output, flush=True,
                                 reclen=reclen, format=wave_format)
                    print('-> OK!')

                except Exception as e:
                    print(f'-> no data, not written. [ {e} ]')

            else:
                print(f'-> No data for station {ele} on [{t0} - {t1}]')
    return

def stream_count_samples(stream, **kwargs):
    """
    Helper function to count total number of samples in ObsPy stream
    """
    n_samples = 0
    for trace in stream:
        n_samples += trace.stats.npts
    return n_samples

def stream_unique_seed_id(stream, **kwargs):
    """
    Helper function to count occurrence of unique SEED ID's
    """
    unique_entries = []

    for ms in stream:
        unique_entries.append(ms.id)
    return set(unique_entries)
