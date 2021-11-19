# Simulation Dataset for BBHnet
# NOTE: OLD README, UPDATE IN PROGRESS


We generate simulation dataset to train BBHnet, our deep learning framework for detection of
compact binary coalescene (CBC) gravitational-wave (GW) signals .

### Example

To generate a noise dataset, simply run `generateRealNoise.py`:
```
python generateRealNoise.py -t0 1186729980 -t1 1186734086 -t0-psd 1186729980 -t1-psd 1186734086
    -fs 1024 -fmin 20 -o test_noise.h5
```

To also add CBC signals, enable the flag `-S` and add the prior distribution file in Bilby format with `-p`
```
python generateRealNoise.py -t0 1186729980 -t1 1186734086 -t0-psd 1186729980 -t1-psd 1186734086
    -fs 1024 -fmin 20 -S -p config/priors/nonspin_BBH.prior -o test_signal.h5
```

A full list of `generateRealNoise.py` arguments can be found below:
```
usage: generateRealNoise.py [-h] -t0 FRAME_START -t1 FRAME_STOP -t0-psd FRAME_START_PSD -t1-psd FRAME_STOP_PSD -o OUTFILE [-S]
                            [-fs SAMPLE_RATE] [-fmin HIGH_PASS] [-T SAMPLE_DURATION] [-dt TIME_STEP] [-p PRIOR_FILE]
                            [--correlation-shift CORRELATION_SHIFT] [--min-trigger MIN_TRIGGER] [--max-trigger MAX_TRIGGER]
                            [-s SEED]

optional arguments:
  -h, --help            show this help message and exit
  -t0 FRAME_START, --frame-start FRAME_START
                        starting GPS time of strain
  -t1 FRAME_STOP, --frame-stop FRAME_STOP
                        stopping GPS time of strain
  -t0-psd FRAME_START_PSD, --frame-start-psd FRAME_START_PSD
                        starting GPS time of strain for PSD estimation
  -t1-psd FRAME_STOP_PSD, --frame-stop-psd FRAME_STOP_PSD
                        stopping GPS time of strain for PSD estimation
  -o OUTFILE, --outfile OUTFILE
                        path to write output file in HDF5 format
  -S, --signal          Enable to add GW signal on top of background noise
  -fs SAMPLE_RATE, --sample-rate SAMPLE_RATE
                        sampling rate of strain
  -fmin HIGH_PASS, --high-pass HIGH_PASS
                        frequency of highpass filter
  -T SAMPLE_DURATION, --sample-duration SAMPLE_DURATION
                        duration in seconds of each sample
  -dt TIME_STEP, --time-step TIME_STEP
                        time step size in seconds between consecutive samples
  -p PRIOR_FILE, --prior-file PRIOR_FILE
                        path to prior config file. Required for signal simulation
  --correlation-shift CORRELATION_SHIFT
                        if given, also compute the correlation with given shift value
  --min-trigger MIN_TRIGGER
                        mininum trigger time w.r.t to sample. must be within [0, sample_duration]
  --max-trigger MAX_TRIGGER
                        maximum trigger time w.r.t to sample. must be within [0, sample_duration]
  -s SEED, --seed SEED  random seed for reproducibility

```
