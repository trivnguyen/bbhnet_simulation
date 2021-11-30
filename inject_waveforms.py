#!/usr/bin/env python
# coding: utf-8

import os
import sys
import h5py
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

    # signal simulation args
    parser.add_argument(
        '-s', '--signal-type', required=True, type=str.lower, choices=('bbh', 'glitch'),
        help='type of signal, either bbh or glitch')
    parser.add_argument(
        '-p', '--prior-file', required=True,
        help='path to prior config file. Required for signal simulation')
    parser.add_argument(
        '--min-trigger', type=float, default=0.05,
        help='mininum trigger time w.r.t to sample. must be within [0, sample_duration]')
    parser.add_argument(
        '--max-trigger', type=float, default=0.95,
        help='maximum trigger time w.r.t to sample. must be within [0, sample_duration]')

    return parser.parse_args()


if __name__ == '__main__':
    ''' Start simulation '''

    # parse command-line arguments
    FLAGS = parse_cmd()

    # compute some sim parameters from cmd input
    sample_size = int(FLAGS.sample_rate * FLAGS.sample_duration)
    fftlength = int(max(2, np.ceil(2048 / FLAGS.sample_rate)))

    # read input frame file and get attributes
    H1_strain = TimeSeries.read(FLAGS.H1_infile, 'H1:GWOSC-4KHZ_R1_STRAIN')
    L1_strain = TimeSeries.read(FLAGS.L1_infile, 'L1:GWOSC-4KHZ_R1_STRAIN')
    sample_rate = H1_strain.sample_rate.value
    duration = H1_strain.duration.value
    fftlength = int(max(2, np.ceil(2048 / sample_rate))

    print(H1_strain, L1_strain)
    print(sample_rate, duration)

    # calculate strain psd
    H1_psd = H1_strain.psd(fftlength)
    L1_psd = L1_strain.psd(fftlength)

    # simulate waveform
    times = np.arange() + H1.t0.value + waveform_duration // 2

    if FLAGS.signal_type == 'bbh':
        # sample GW parameters from a prior distribution
        logging.info('Simulating BBH signals from prior file {}'.format(FLAGS.prior_file))
        priors = bilby.gw.prior.BBHPriorDict(FLAGS.prior_file)
        sample_params = priors.sample(num_samples)

        # Bilby does not sample GPS time, so we have to manually do it
        triggers = np.zeros(FLAGS.n_sample) * triggers
        sample_params['geocent_time'] = triggers + times

        # simulate whitened GW waveforms
        signals, snr = utils.simulate_whitened_bbh_signals(
            sample_params, FLAGS.sample_rate, FLAGS.sample_duration, triggers)
