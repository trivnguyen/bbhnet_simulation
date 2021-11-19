
import numpy as np

import bilby
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters

from gwpy.timeseries import TimeSeries

def get_snr(data, noise_psd, fs, flow=20):
    ''' Calculate the waveform SNR given the background noise PSD'''

    data_fd = np.fft.rfft(data) / fs
    data_freq = np.fft.rfftfreq(len(data)) * fs
    dfreq = data_freq[1] - data_freq[0]

    noise_psd_interp = noise_psd.interpolate(dfreq)
    noise_psd_interp[noise_psd_interp == 0] = 1.

    snr = 4 * np.abs(data_fd)**2 / noise_psd_interp.value * dfreq
    snr = np.sum(snr[fmin <= data_freq])
    snr = np.sqrt(snr)

    return snr

def as_stride(x, input_size, step, shift=0):
    ''' Divide input time series into overlapping chunk '''

    if shift != 0:
        x = np.roll(x, shift)

    noverlap = input_size  - step
    N_sample = (x.shape[-1] - noverlap) // step

    shape = (N_sample, input_size)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    return result

def pearson_shift(x, y, shift):
    ''' Calculate the Pearson correlation coefficient between x and y
    for each array shift value from [-shift, shift].
    '''

    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denom = ((x**2).sum(1) * (y**2).sum(1))**0.5

    corr = np.zeros((x.shape[0], shift * 2))
    for i, s in enumerate(range(-shift, shift)):
        xr = np.roll(x, s, axis=1)
        corr[:, i] = (xr * y).sum(1) / denom
    return corr

def simulate_whitened_bbh_signals(
    sample_params, sample_rate, sample_duration, triggers, H1_psd, L1_psd):
    ''' generate BBH signals
    Arguments:
    - sample_params: dictionary of GW parameters
    - sample_duration: time duration of each sample
    - triggers: trigger time (relative to `sample_duration`) of each sample
    - H1_psd, L1_psd: Hanford and Livingston PSD
    '''

    # define some signal properties
    # get total number of samples
    waveform_duration = 8
    num_samples = len(sample_params['geocent_time'])
    sample_size = int(sample_rate * sample_duration)
    waveform_size = int(sample_rate * waveform_duration)

    # define a Bilby waveform generator
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=waveform_duration, sampling_frequency=sample_rate,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            'waveform_approximant': 'IMRPhenomPv2',
            'reference_frequency': 50,
            'minimum_frequency': 20})

    # loop over all BBH signals
    signals = np.zeros((num_samples, 2, sample_size))
    snr = np.zeros((num_samples, 2))
    for i in range(num_samples):
        # get parameters of the current BBH signal
        p = dict()
        for k, v in sample_params.items():
            p[k] = v[i]
        ra, dec, geceont_time, psi = p['ra'], p['dec'], p['geocent_time'], p['psi']
        polarizations = waveform_generator.time_domain_strain(p)

        # simulate signals for Hanford and Livingston
        for j, ifo_name in enumerate(('H1', 'L1')):
            # get detector and PSD
            if ifo_name == 'H1':
                noise_psd = H1_psd
            else:
                noise_psd = L1_psd
            ifo = bilby.gw.detector.get_empty_interferometer(ifo_name)

            # add polarizations with detector geometry
            signal = np.zeros(waveform_size)
            for mode in polarizations.keys():
                # Get H1 response
                response = ifo.antenna_response(ra, dec, geocent_time, psi, mode)
                signal += polarizations[mode] * response

            # shift signal back to the trigger time of the simulated strain
            # also add shift due to time travel between detectors
            # total shift = shift to trigger time + geometric shift
            dt = (waveform_duration - sample_duration) / 2. + triggers[i]
            dt += ifo.time_delay_from_geocenter(ra, dec, geocent_time)
            signal = np.roll(signal, int(np.round(dt*sample_rate)))

            # bandpass filter and whitenting using gwpy
            signal = TimeSeries(signal, dt=1./sample_rate)
            if (flow is not None) and (fhigh is None):
                signal = signal.highpass(flow)
            elif (flow is None) and (fhigh is not None):
                signal = signal.lowpass(fhigh)
            else:
                signal = signal.bandpass(flow, fhigh)
            signal = signal.whiten(asd=np.sqrt(noise_psd[j]))

            # convert back to numpy array
            signal = signal.value

            # calculate snr
            snr[i][j] = get_snr(signal, noise_psd[j], sample_rate)

            # truncate signal
            istart = (waveform_size - sample_size) // 2
            tstop = idx_start + sample_size
            signal = signal[istart:istop]

            signals[i, j] += signal

    return signals, snr
