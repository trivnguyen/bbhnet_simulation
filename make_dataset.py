#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import argparse
import shutil

import numpy as np

FLAGS = None

# Parse cmd arguments
def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-files', required=True, nargs='+',
                        help='Path to input files')
    parser.add_argument('-o', '--out-dir', required=True,
                        help='Path to output directory')
    parser.add_argument('-n', '--num-splits', required=False, type=int, default=int,
                        help='Number of output files to split into')
    parser.add_argument('--overwrite', required=False, action='store_true',
                        help='Enable to overwrite existing dataset')
    parser.add_argument('--seed', required=False, type=int,
                        help='Random seed for reproducibility')
    return parser.parse_args()

if __name__  == '__main__':

    # parse cmd argument
    FLAGS = parse_cmd()

    if FLAGS.overwrite and os.path.exists(FLAGS.out_dir):
        shutil.rmtree(FLAGS.out_dir)
    os.makedirs(FLAGS.out_dir)

    # get the total size of the dataset
    num_samples = []
    for input_file in FLAGS.input_files:
        with h5py.File(input_file, 'r') as f:
            num_samples.append(f.attrs['size'])

    # shuffle indices
    np.random.seed(FLAGS.seed)
    shuffle = np.random.permutation(np.sum(num_samples))

    # Write dataset files
    # loop over all attributes
    for prop in ('data', 'label', 'corr', 'snr'):
        full_data = []
        # loop over all input files
        for input_file in FLAGS.input_files:
            with h5py.File(input_file, 'r') as f:
                full_data.append(f[prop][:])
        full_data = np.concatenate(full_data)
        full_data = full_data[shuffle]

        # split data into multiple files
        split_data = np.array_split(full_data, FLAGS.num_splits)
        for i, data in enumerate(split_data):
            output_file = os.path.join(FLAGS.out_dir, 'n{:02d}.h5'.format(i))
            with h5py.File(output_file, 'a') as f:
                f.create_dataset(prop, data=data)

    # also write the shuffle index to keep track of original position
    split_data = np.array_split(shuffle, FLAGS.num_splits)
    for i, data in enumerate(split_data):
        output_file = os.path.join(FLAGS.out_dir, 'n{:02d}.h5'.format(i))
        with h5py.File(output_file, 'a') as f:
            f.create_dataset('indices', data=data)

    # Write parameter files
    for i, input_file in enumerate(FLAGS.input_files):
        with h5py.File(input_file, 'r') as f:
            if f.get('signal_params') is not None:
                params = {}
                for k, v in f['signal_params'].items():
                    params[k] = v
                # get shuffle index
                istart = 0 if i==0 else int(np.cumsum(num_samples)[i-1])
                istop = int(np.cumsum(num_samples)[i])
                params['indices'] = shuffle[istart:istop]

                # write params file
                params_file = os.path.join(
                    FLAGS.out_dir, 'params-n{:02d}.h5'.format(i))
                with h5py.File(params_file, 'w') as out_f:
                    for k, v in params.items():
                        out_f.create_dataset(k, data=v)














