#!/usr/bin/env python
# coding: utf-8

import os
import glob
import argparse

from gwpy.timeseries import TimeSeries


def parse_cmd():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', required=True,
                        help='path to input directory')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='path to output directory')
    return parser.parse_args()


if __name__  == '__main__':
    ''' Split 4096-sec frame into 4 1024-sec frames '''

    FLAGS = parse_cmd()

    for ifo in ('H1', 'L1'):
        ifo_input_dir = os.path.join(FLAGS.input_dir, ifo)
        ifo_output_dir = os.path.join(FLAGS.output_dir, ifo)
        os.makedirs(ifo_output_dir, exist_ok=True)

        input_files = sorted(glob.glob(os.path.join(ifo_input_dir, '*.gwf')))

        channel = '{}:GWOSC-4KHZ_R1_STRAIN'.format(ifo)
        if ifo == 'H1':
            basename = 'H-H1_GWOSC_O2_4KHZ_R1'
        elif ifo == 'L1':
            basename = 'L-L1_GWOSC_O2_4KHZ_R1'

        for input_file in input_files:
            t0, T  = os.path.splitext(os.path.basename(input_file))[0].split('-')[2:]
            t0 = int(t0)
            T = int(T)

            # read frame
            strain = TimeSeries.read(input_file, channel)
            T_new = T//4

            for i in range(4):
                t0_new = int(t0 + i * T_new)
                strain_new = strain.crop(t0_new, t0_new + T_new)
                strain_new.write(os.path.join(ifo_output_dir, '{}-{}-{}.gwf'.format(
                                 basename, t0_new, T_new)))

