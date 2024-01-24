#!/usr/bin/python3 -u
# convert one or more wav files into a Digital RF dataset
#
# Copyright 2023 Franco Venturi K4VZ
#
# Version: 1.0 - Tue Dec 26 05:47:14 PM EST 2023

from datetime import datetime
import digital_rf as drf
import getopt
import numpy as np
import os
import soundfile as sf
import sys
import uuid



# Digital RF settings
drf_channel_name = 'ch0'
drf_subdir_cadence_secs = 21600
drf_file_cadence_millisecs = 3600000
drf_compression_level = 9


def main():
    drfdir = None
    frequencies = []
    start_time = 0
    # location (default: KFS)
    latitude = 37.444722
    longitude = -122.1125
    uuid_str = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:f:s:l:u:')
    except getopt.GetoptError as ex:
        print(ex, file=sys.stderr)
        sys.exit(1)
    for o, a in opts:
        if o == '-d':
            drfdir = a
        elif o == '-f':
            frequencies.extend([float(x) for x in a.split(',')])
        elif o == '-s':
            start_time = datetime.fromisoformat(a).timestamp()
        elif o == '-l':
            latitude, longitude = [float(x) for x in a.split(',')]
        elif o == '-u':
            uuid_str = a

    if not args:
        print('missing input files', file=sys.stderr)
        sys.exit(1)

    if drfdir is None:
        print('missing Digital RF output dir (-d) option', file=sys.stderr)
        sys.exit(1)

    if len(frequencies) != len(args):
        print(f'number of frequencies {len(frequencies)} does not match with number of input files {len(args)}', file=sys.stderr)
        sys.exit(1)

    sample_rate = None
    num_samples = None
    num_channels = None
    data_type = None
    samples = [None] * len(args)
    for idx, wav_file in enumerate(args):
        samples[idx], wav_sample_rate = sf.read(wav_file, dtype='float32')

        # sanity checks
        if sample_rate == None:
            sample_rate = wav_sample_rate
        elif wav_sample_rate != sample_rate:
            print('sample rates do not match - file', wav_file, 'has', wav_sample_rate, '- expecting:', sample_rate, file=sys.stderr)
            sys.exit(1)
        wav_num_samples = samples[idx].shape[0]
        if num_samples == None:
            num_samples = wav_num_samples
        elif wav_num_samples != num_samples:
            print('number of samples does not match - file', wav_file, 'has', wav_num_samples, '- expecting:', num_samples, file=sys.stderr)
            sys.exit(1)
        wav_num_channels = samples[idx].shape[1]
        if num_channels == None:
            num_channels = wav_num_channels
        elif wav_num_channels != num_channels:
            print('number of channels does not match - file', wav_file, 'has', wav_num_channels, '- expecting:', num_channels, file=sys.stderr)
            sys.exit(1)
        wav_data_type = samples[idx].dtype
        if data_type == None:
            data_type = wav_data_type
        elif wav_data_type != data_type:
            print('data type does not match - file', wav_file, 'has', wav_data_type, '- expecting:', data_type, file=sys.stderr)
            sys.exit(1)

    start_global_index = int(start_time * sample_rate)
    if uuid_str is None:
        uuid_str = uuid.uuid4().hex

    # the output directory must already exist
    chdir = os.path.join(drfdir, drf_channel_name)
    os.makedirs(chdir)
    metadatadir = os.path.join(chdir, 'metadata')
    os.mkdir(metadatadir)

    # write Digital RF metadata first
    do = drf.DigitalMetadataWriter(metadatadir,
                                   drf_subdir_cadence_secs,
                                   drf_subdir_cadence_secs,  # file_cadence_secs
                                   sample_rate,      # sample_rate_numerator 
                                   1,                # sample_rate_denominator
                                   'metadata'        # file_name
                                  )
    data_dict = {
        'uuid_str': uuid_str,
        'lat': np.single(latitude),
        'long': np.single(longitude),
        'center_frequencies': np.ascontiguousarray(frequencies)
    }
    do.write(start_global_index, data_dict)

    # write Digital RF data
    with drf.DigitalRFWriter(chdir,
                             data_type,
                             drf_subdir_cadence_secs,
                             drf_file_cadence_millisecs,
                             start_global_index,
                             sample_rate,            # sample_rate_numerator
                             1,                      # sample_rate_denominator
                             uuid_str,
                             drf_compression_level,
                             False,                  # checksum
                             num_channels == 2,      # is_complex
                             len(args),              # num_subchannels
                             True,                   # is_continuous
                             False                   # marching_periods
                            ) as do:
        do.rf_write(np.hstack(samples, casting='no'))
    print('done', file=sys.stderr)


if __name__ == '__main__':
    main()
