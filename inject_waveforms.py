#!/usr/bin/env python
# coding: utf-8

import os
import sys
import h5py
import copy
import argparse
import logging

import numpy as np
import scipy.signal as sig
import bilby
from gwpy.timeseries import TimeSeries

from bbhnet_simulation import utils

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO, stream=sys.stdout)

# Global constant
FLAGS = None

# Parse command-line arguments
def parse_cmd():
    parser = argparse.ArgumentParser()

    # input/output and gps time args
    parser.add_argument('-iH', '--H1-infile', required=True,
                        help='path to Hanford input file in GWF format')
    parser.add_argument('-iL', '--L1-infile', required=True,
                        help='path to Livingston input file in GWF format')
    parser.add_argument('-oH', '--H1-outfile', required=True,
                        help='path to write Hanford output file in GWF format')
    parser.add_argument('-oL', '--L1-outfile', required=True,
                        help='path to write Livingston output file in GWF format')
    parser.add_argument('-oP', '--parameters-outfile', required=True,
                        help='path to write output file containing parameters')

    # signal simulation args
    parser.add_argument(
        '-s', '--signal-type', required=True, type=str.lower, choices=('bbh', 'glitch'),
        help='type of signal, either bbh or glitch')
    parser.add_argument(
        '-p', '--prior-file', required=True,
        help='path to prior config file. Required for signal simulation')
    parser.add_argument('--min-dt', type=float, default=20,
                        help='Minimum time between signals')
    parser.add_argument('--max-dt', type=float, default=100,
                        help='Maximum time between signals')
    parser.add_argument('--min-snr', type=float, default=25,
                        help='mininum signal-to-noise ratio')
    parser.add_argument('--max-snr', type=float, default=25,
                        help='maximum signal-to-noise ratio')

    return parser.parse_args()


if __name__ == '__main__':
    ''' Start simulation '''

    # parse command-line arguments
    FLAGS = parse_cmd()
    waveform_duration = 8

    # read input frame file and get attributes
    logging.info('Reading Hanford    data from : {}'.format(FLAGS.H1_infile))
    logging.info('Reading Livingston data from : {}'.format(FLAGS.L1_infile))

    H1_strain = TimeSeries.read(FLAGS.H1_infile, 'H1:GWOSC-4KHZ_R1_STRAIN')
    L1_strain = TimeSeries.read(FLAGS.L1_infile, 'L1:GWOSC-4KHZ_R1_STRAIN')
    sample_rate = H1_strain.sample_rate.value
    duration = H1_strain.duration.value
    fftlength = int(max(2, np.ceil(2048 / sample_rate)))

    # calculate strain psd
    H1_psd = H1_strain.psd(fftlength)
    L1_psd = L1_strain.psd(fftlength)

    # Simulate waveform
    # first, we sample the time between subsequent signals
    max_num_samples = int(duration // FLAGS.min_dt)
    times = np.random.uniform(FLAGS.min_dt, FLAGS.max_dt, max_num_samples)
    times = np.cumsum(times)
    times = times[times < duration - FLAGS.min_dt]
    times += H1_strain.t0.value
    num_samples = len(times)

    logging.info('Number of samples: {}'.format(num_samples))
    logging.info('Minimum time between signals: {} s'.format(FLAGS.min_dt))
    logging.info('Maximum time between signals: {} s'.format(FLAGS.max_dt))
    logging.info('Minimum SNR: {}'.format(FLAGS.min_snr))
    logging.info('Maximum SNR: {}'.format(FLAGS.max_snr))

    if FLAGS.signal_type == 'bbh':
        # sample GW parameters from a prior distribution
        logging.info('Simulating BBH signals from prior file {}'.format(FLAGS.prior_file))
        priors = bilby.gw.prior.BBHPriorDict(FLAGS.prior_file)
        sample_params = priors.sample(num_samples)

        # Bilby does not sample GPS time, so we have to manually do it
        triggers = np.zeros(num_samples) + waveform_duration // 2
        sample_params['geocent_time'] = times

        # simulate whitened GW waveforms
        signals, old_snr = utils.simulate_bbh_signals(
            sample_params, sample_rate, waveform_duration, triggers,
            H1_psd=H1_psd, L1_psd=L1_psd)

        # sample snr
        new_snr = np.random.uniform(FLAGS.min_snr, FLAGS.max_snr, num_samples)

        # scale signal to new snr
        scale = new_snr / np.sqrt(np.sum(old_snr**2, axis=1))
        signals = signals * scale.reshape(-1, 1, 1)
        snr = old_snr * scale.reshape(-1, 1)
        sample_params['luminosity_distance'] /= scale

    # write to a different frame
    logging.info('Writing new Hanford    data to: {}'.format(FLAGS.H1_outfile))
    logging.info('Writing new Livingston data to: {}'.format(FLAGS.L1_outfile))

    # create a copy of current strain
    # add signals into new strain
    H1_new_strain = copy.deepcopy(H1_strain)
    L1_new_strain = copy.deepcopy(L1_strain)
    for t, signal in zip(times, signals):
        H1_signal = signal[0]
        L1_signal = signal[1]

        tstart_offset = t - H1_strain.t0.value - waveform_duration // 2
        istart = int(sample_rate * tstart_offset)
        istop = istart + int(sample_rate * waveform_duration)
        H1_new_strain[istart: istop] += H1_signal * H1_new_strain.unit
        L1_new_strain[istart: istop] += L1_signal * L1_new_strain.unit

    H1_new_strain.write(FLAGS.H1_outfile)
    L1_new_strain.write(FLAGS.L1_outfile)

    # write output parameters
    logging.info('Write output parameters to {}'.format(
        FLAGS.parameters_outfile))
    with h5py.File(FLAGS.parameters_outfile, 'w') as f:
        f.create_dataset('GPS-start', data=times)
        f.create_group('H1')
        f['H1'].create_dataset('SNR', data=snr[:, 0])
        f['H1'].create_dataset('signal', data=signals[:, 0])
        f.create_group('L1')
        f['L1'].create_dataset('SNR', data=snr[:, 1])
        f['L1'].create_dataset('signal', data=signals[:, 1])

        f.create_group('signal_params')
        for k, v in sample_params.items():
            f['signal_params'].create_dataset(k, data=v)
