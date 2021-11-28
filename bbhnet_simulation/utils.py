
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
    snr = np.sum(snr[flow <= data_freq])
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

def time_domain_sine_gaussian(params, duration, sample_rate, clip=1.):
    ''' Function to generate Sine Gaussian'''
    # Get sine gauss parameters
    A = params.get('A', 1)
    f0 = params.get('f0', 100)
    tau = params.get('tau', 0.05)
    t0 = duration / 2.

    # Generate sine guassian
    t = np.arange(0., duration, 1./sample_rate)
    x = A * np.exp(-(t - t0)**2 / tau**2) * np.sin(2 * np.pi * f0 * (t - t0))

    # clip waveform
    max_A = A * clip
    x[x > max_A] = max_A
    x[x < -max_A] = -max_A

    return x

def time_domain_blip(duration, sample_rate, f0_min=20, f0_width=1000, N=20):
    ''' Generate blip glitches from adding sine gaussian waveform '''
    f0_arr = 10**np.linspace(np.log10(f0_min), np.log10(f0_min + f0_width), N)
    tau_arr = 10**np.linspace(-1, -4, N)

    # Generate blip glitch by adding SG
    x = np.zeros(int(duration * sample_rate))
    for i in range(N):
        params = dict(f0=f0_arr[i], tau=tau_arr[i])
        x += time_domain_sine_gaussian(params, duration, sample_rate)
    return x

def simulate_whitened_bbh_signals(
    sample_params, sample_rate, sample_duration, triggers, H1_psd, L1_psd,
    flow=None, fhigh=None):
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
        ra, dec, geocent_time, psi = p['ra'], p['dec'], p['geocent_time'], p['psi']
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
            signal = signal.whiten(asd=np.sqrt(noise_psd))

            # convert back to numpy array
            signal = signal.value

            # calculate snr
            snr[i][j] = get_snr(signal, noise_psd, sample_rate)

            # truncate signal
            istart = (waveform_size - sample_size) // 2
            istop = istart + sample_size
            signal = signal[istart:istop]

            signals[i, j] += signal

    return signals, snr

def simulate_whitened_blip_glitches(
    sample_params, sample_rate, sample_duration, triggers, H1_psd, L1_psd,
    flow=None, fhigh=None):
    ''' generate blip glitches
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

    # loop over all blip glitches signal
    signals = np.zeros((num_samples, 2, sample_size))
    snr = np.zeros((num_samples, 2))
    for i in range(num_samples):
        # randomly chosen whether the glitch is H1 or L1
        if np.random.rand() > 0.5:
            ifo = 'H1'
            noise_psd = H1_psd
        else:
            ifo = 'L1'
            noise_psd = L1_psd

        # simulate blip glitch signal
        f0_min = sample_params['f0_min'][i]
        f0_width = sample_params['f0_width'][i]
        target_snr = sample_params['snr'][i]
        signal = time_domain_blip(waveform_duration, sample_rate, f0_min, f0_width)

        # shift to trigger time
        dt = triggers[i] - sample_duration / 2.
        signal = np.roll(signal, int(dt * sample_rate))

        # bandpass filter
        signal = TimeSeries(signal, dt=1./sample_rate)
        if (flow is not None) and (fhigh is None):
            signal = signal.highpass(flow)
        elif (flow is None) and (fhigh is not None):
            signal = signal.lowpass(fhigh)
        else:
            signal = signal.bandpass(flow, fhigh)
        # calculate snr of signal and scale to the chosen snr
        signal_snr = get_snr(signal, noise_psd, sample_rate)
        signal = signal * target_snr / signal_snr

        # whitening signal
        signal = signal.whiten(asd=np.sqrt(noise_psd))

        # convert back to numpy array
        signal = signal.value


        # truncate signal
        istart = (waveform_size - sample_size) // 2
        istop = istart + sample_size
        signal = signal[istart:istop]

        # add to the appropriate detector
        if ifo == 'H1':
            snr[i][0] = target_snr
            signals[i, 0] += signal
        elif ifo == 'L1':
            snr[i][1] = target_snr
            signals[i, 1] += signal

    return signals, snr

