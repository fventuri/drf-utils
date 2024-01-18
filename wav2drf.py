#!/usr/bin/python3 -u
# convert a directory tree containing wav files into a Digital RF dataset
# directory structure:
#   - topdir (input dir: -i)
#     - subdir1 (subchannel 1) - frequency is now derived from channel name
#       - <filename>.wav
#     - subdir2 (subchannel 2) - frequency is now derived from channel name
#       - <filename>.wav
#     ...
#
# Copyright 2024 Franco Venturi K4VZ
#
# Version: 1.0 - Tue Jan 16 09:46:22 PM EST 2024

from collections import defaultdict
from configparser import ConfigParser
from datetime import datetime, timezone
import digital_rf as drf
import getopt
import numpy as np
import os
import re
import soundfile as sf
import sys
import uuid

# global variables
verbose = 0


def get_subchannels(inputdir, subdir2freq):
    # create list of subchannels and make sure that each has one wav file in it
    subchannels = []
    for subdir in os.listdir(inputdir):
        if subdir not in subdir2freq:
            print('Subdir', subdir, 'not found in subchannels list. Skipping it.', file=sys.stderr)
            continue
        subdir_content = os.listdir(os.path.join(inputdir, subdir))
        if not (len(subdir_content) == 1 and subdir_content[0].endswith('.wav')):
            print('Subdir', subdir, 'does not contain a single wav file. Skipping it.', f'(content: {subdir_content})', file=sys.stderr)
            continue
        subchannels.append((subdir, subdir_content[0], float(subdir2freq[subdir])))
    # return subchannels in ascending frequency order
    subchannels.sort(key=lambda x: x[2])
    return subchannels


def create_drf_dataset(inputdir, outputdir, subchannels, config_global, start_time, uuid_str=None):
    channel_name = config_global['channel name']
    subdir_cadence_secs = int(config_global['subdir cadence secs'])
    file_cadence_millisecs = int(config_global['file cadence millisecs'])
    compression_level = int(config_global['compression level'])
    dtype = 'i2'  # 16bit shorts
    #dtype = 'float32'

    if uuid_str is None:
        uuid_str = uuid.uuid4().hex

    print('writing Digital RF dataset. This will take a while', file=sys.stderr)

    # build np.array with samples and validate wav files to make sure they
    # are all consistent (same num samples, channels, data type, etc)
    sample_rate = None
    num_samples = None
    num_channels = None
    data_type = None
    samples = [None] * len(subchannels)
    for idx, subchannel in enumerate(subchannels):
        wav_file = os.path.join(inputdir, subchannel[0], subchannel[1])
        samples[idx], wav_sample_rate = sf.read(wav_file, dtype=dtype)

        # sanity checks
        if sample_rate == None:
            sample_rate = wav_sample_rate
        elif wav_sample_rate != sample_rate:
            printf('sample rates do not match - file', wav_file, 'has', wav_sample_rate, '- expecting:', sample_rate, file=sys.stderr)
            return False, None, None, None, None
        wav_num_samples = samples[idx].shape[0]
        if num_samples == None:
            num_samples = wav_num_samples
        elif wav_num_samples != num_samples:
            printf('number of samples does not match - file', wav_file, 'has', wav_num_samples, '- expecting:', num_samples, file=sys.stderr)
            return False, None, None, None, None
        wav_num_channels = samples[idx].shape[1]
        if num_channels == None:
            num_channels = wav_num_channels
        elif wav_num_channels != num_channels:
            printf('number of channels does not match - file', wav_file, 'has', wav_num_channels, '- expecting:', num_channels, file=sys.stderr)
            return False, None, None, None, None
        wav_data_type = samples[idx].dtype
        if data_type == None:
            data_type = wav_data_type
        elif wav_data_type != data_type:
            printf('data type does not match - file', wav_file, 'has', wav_data_type, '- expecting:', data_type, file=sys.stderr)
            return False, None, None, None, None

    if verbose >= 1:
        print('sample_rate:', sample_rate)
        print('num_samples:', num_samples)
        print('num_channels:', num_channels)
        print('data_type:', data_type)
        print('len(samples):', len(samples))

    start_global_index = int(start_time * sample_rate)
    if uuid_str is None:
        uuid_str = uuid.uuid4().hex

    # the output directory must already exist
    channel_dir = os.path.join(outputdir, channel_name)
    os.makedirs(channel_dir)

    with drf.DigitalRFWriter(channel_dir,
                             dtype,
                             subdir_cadence_secs,
                             file_cadence_millisecs,
                             start_global_index,
                             sample_rate,            # sample_rate_numerator
                             1,                      # sample_rate_denominator
                             uuid_str,
                             compression_level,
                             False,                  # checksum
                             num_channels == 2,      # is_complex
                             len(subchannels),       # num_subchannels
                             True,                   # is_continuous
                             False                   # marching_periods
                            ) as do:

        do.rf_write(np.hstack(samples, casting='no'))

    # hopefully deleting samples will free all the memory
    del samples

    return True, channel_dir, sample_rate, start_global_index, uuid_str


def create_drf_metadata(channel_dir, frequencies, config_global, config_metadata, sample_rate, start_global_index, uuid_str):
    subdir_cadence_secs = int(config_global['subdir cadence secs'])
    metadatadir = os.path.join(channel_dir, 'metadata')
    os.makedirs(metadatadir)
    do = drf.DigitalMetadataWriter(metadatadir,
                                   subdir_cadence_secs,
                                   subdir_cadence_secs,  # file_cadence_secs
                                   sample_rate,      # sample_rate_numerator 
                                   1,                # sample_rate_denominator
                                   'metadata'        # file_name
                                  )
    sample = start_global_index
    data_dict = {
        'uuid_str': uuid_str,
        'lat': np.single(float(config_metadata['latitude'])),
        'long': np.single(float(config_metadata['longitude'])),
        'center_frequencies': np.ascontiguousarray(frequencies)
    }
    # all all the remaining arguments as strings
    for k, v in config_metadata.items():
        if k in ['latitude', 'longitude']:
            continue
        data_dict[k] = v
    do.write(sample, data_dict)
    return True


def main():
    configfile = sys.argv[0].replace('.py', '.conf')
    inputdir = None
    outputdir = None
    start_time = 0
    uuid_str = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'c:i:o:s:u:v')
    except getopt.GetoptError as ex:
        print(ex, file=sys.stderr)
        sys.exit(1)
    for o, a in opts:
        if o == '-c':
            configfile = a
        elif o == '-i':
            inputdir = a
        elif o == '-o':
            outputdir = a
        elif o == '-s':
            # allow for time zone (default is local TZ)
            #start_time = datetime.fromisoformat(a).timestamp()
            # always UTC
            start_datetime = datetime.fromisoformat(a).replace(tzinfo=timezone.utc)
            start_time = start_datetime.timestamp()
        elif o == '-u':
            uuid_str = a
        elif o == '-v':
            global verbose
            verbose += 1

    if inputdir is None:
        print('missing input dir (-i) option', file=sys.stderr)
        sys.exit(1)

    if outputdir is None:
        print('missing output dir (-o) option', file=sys.stderr)
        sys.exit(1)

    if start_time == 0:
        for path_element in inputdir.split(os.path.sep):
            try:
                start_datetime = datetime.strptime(path_element, '%Y%m%d').replace(tzinfo=timezone.utc)
                start_time = start_datetime.timestamp()
                break
            except ValueError:
                pass

    config = ConfigParser(interpolation=None)
    config.optionxform = str
    config.read(configfile)

    subchannels = get_subchannels(inputdir, config['subchannels'])
    if len(subchannels) == 0:
        print("No subchannels (i.e. no wav files) found. Nothing to do.", file=sys.stderr)
        sys.exit(0)
    print('N subchannels:', len(subchannels), file=sys.stderr)
    if verbose >= 1:
        print('subchannels:', subchannels)

    ok, channel_dir, sample_rate, start_global_index, uuid_str = create_drf_dataset(inputdir, outputdir, subchannels, config['global'], start_time, uuid_str)
    print('create_drf_dataset returned', ok, file=sys.stderr)
    if not ok:
        sys.exit(1)

    frequencies = [float(x[2]) for x in subchannels]
    ok = create_drf_metadata(channel_dir, frequencies, config['global'], config['metadata'], sample_rate, start_global_index, uuid_str)
    print('create_drf_metadata returned', ok, file=sys.stderr)
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
