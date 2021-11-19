#!/usr/bin/env python
# coding: utf-8

import os
import sys
import h5py
import argparse
import logging

import numpy as np
import scipy.signal as sig
from gwpy.timeseries import TimeSeries

from bbhnet_simulation import utils

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO, stream=sys.stdout)

# Global constant
FLAGS = None

# Function to read and preprocess data
def read_strain(ifo, t0, t1, fs, flow=None, fhigh=None):
    '''
    Convienience function to read and preprocess strain data
    '''
    # download strain from GWOSC
    strain = TimeSeries.fetch_open_data(ifo, t0, t1)

    # resample strain
    strain = strain.resample(fs)

    # apply bandpass filter if given
    if (flow is not None) and (fhigh is not None):
        strain = strain.bandpass(flow, fhigh)
    elif (flow is None) and (fhigh is not None):
        strain = strain.lowpass(fhigh)
    elif (flow is not None) and (fhigh is None):
        strain = strain.highpass(flow)

    return  strain

# Parse command-line arguments
def parse_cmd():
    parser = argparse.ArgumentParser()

    # input/output and gps time args
    parser.add_argument('-t0', '--frame-start', type=float, required=True,
                        help='starting GPS time of strain')
    parser.add_argument('-t1', '--frame-stop', type=float, required=True,
                        help='stopping GPS time of strain')
    parser.add_argument('-t0-psd', '--frame-start-psd', type=float, required=True,
                        help='starting GPS time of strain for PSD estimation')
    parser.add_argument('-t1-psd', '--frame-stop-psd', type=float, required=True,
                        help='stopping GPS time of strain for PSD estimation')
    parser.add_argument('-o', '--outfile', required=True,
                        help='path to write output file in HDF5 format')

    # simulation args
    parser.add_argument('-fs', '--sample-rate', type=float, required=False, default=1024,
                        help='sampling rate of strain')
    parser.add_argument('-fl', '--flow', type=float, required=False,
                        help='minimum frequency of bandpass filter')
    parser.add_argument('-fh', '--fhigh', type=float, required=False,
                        help='maximum frequency of bandpass filter')
    parser.add_argument('-T', '--sample-duration', type=float, required=False, default=1,
                        help='duration in seconds of each sample')
    parser.add_argument('-dt', '--time-step', type=float, required=False, default=0.25,
                        help='time step size in seconds between consecutive samples')
    parser.add_argument('--correlation-shift', type=int, required=False,
                        help='if given, also compute the correlation with given shift value')

    return parser.parse_args()


if __name__ == '__main__':
    ''' Start background simulation '''

    # parse command-line arguments
    FLAGS = parse_cmd()

    # compute some simulation parameters
    sample_size = int(FLAGS.sample_rate * FLAGS.sample_duration)
    step_size = int(FLAGS.sample_rate * FLAGS.time_step)
    fftlength = int(max(2, np.ceil(2048 / FLAGS.sample_rate)))

    # download data strain and PSD strain from GWOSC
    logging.info('Download data strain from {} .. {}'.format(
        FLAGS.frame_start, FLAGS.frame_stop))
    logging.info('Download PSD  strain from {} .. {}'.format(
        FLAGS.frame_start_psd, FLAGS.frame_stop_psd))

    H1_strain = read_strain(
        'H1', FLAGS.frame_start, FLAGS.frame_stop, FLAGS.sample_rate,
        FLAGS.flow, FLAGS.fhigh)
    L1_strain = read_strain(
        'L1', FLAGS.frame_start, FLAGS.frame_stop, FLAGS.sample_rate,
        FLAGS.flow, FLAGS.fhigh)
    H1_psd_strain = read_strain(
        'H1', FLAGS.frame_start_psd, FLAGS.frame_stop_psd, FLAGS.sample_rate,
        FLAGS.flow, FLAGS.fhigh)
    L1_psd_strain = read_strain(
        'L1', FLAGS.frame_start_psd, FLAGS.frame_stop_psd, FLAGS.sample_rate,
        FLAGS.flow, FLAGS.fhigh)

    # calculate the PSD and whiten noise
    H1_psd = H1_psd_strain.psd(fftlength)
    L1_psd = L1_psd_strain.psd(fftlength)
    H1_strain = H1_strain.whiten(asd=np.sqrt(H1_psd))
    L1_strain = L1_strain.whiten(asd=np.sqrt(L1_psd))

    # crop out parts of the data that are corrupted due to whitening process
    H1_strain = H1_strain.crop(FLAGS.frame_start + 4, FLAGS.frame_stop - 4)
    L1_strain = L1_strain.crop(FLAGS.frame_start + 4, FLAGS.frame_stop - 4)

    # change data into format that the neural network can read
    # divide into small, overlapping segments
    H1_data = utils.as_stride(H1_strain.value, sample_size, step_size)
    L1_data = utils.as_stride(L1_strain.value, sample_size, step_size)
    data = np.stack([H1_data, L1_data], axis=1)

    # also compute the Pearson correlation array if shift is given
    if FLAGS.correlation_shift is not None:
        correlation = utils.pearson_shift(
            H1_data, L1_data, shift=FLAGS.correlation_shift)
    else:
        correlation = None

    # also get the starting GPS time of each sample
    times = H1_strain.t0.value + np.arange(0., len(H1_data)) * FLAGS.time_step

    # print out info
    logging.info('Number of samples: {}'.format(len(data)))

    # write output file
    logging.info('Write to file {}'.format(FLAGS.outfile))
    with h5py.File(FLAGS.outfile, 'w') as f:
        f.create_dataset('data', data=data)
        if correlation is not None:
            f.create_dataset('corr', data=correlation)
        f.create_dataset('times', data=times)

        f.create_dataset('h1_psd', data=H1_psd.value)
        f.create_dataset('l1_psd', data=L1_psd.value)
        f.create_dataset('freq', data=H1_psd.frequencies.value)

        # write noise attributes
        f.attrs.update({
            'size': len(data),
            'frame_start': FLAGS.frame_start,
            'frame_stop': FLAGS.frame_stop,
            'psd_frame_start': FLAGS.frame_start_psd,
            'psd_frame_stop': FLAGS.frame_stop_psd,
            'sample_rate': FLAGS.sample_rate,
            'sample_duration': FLAGS.sample_duration,
            'psd_fftlength': fftlength,
        })

