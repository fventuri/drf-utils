#!/usr/bin/python3 -u
# convert a bunch of flac files into a Digital RF dataset
# directory structure:
#   - topdir (input dir: -i)
#     - subdir1 (subchannel 1)
#       - <iso8601 timestamp>-<frequency1 Hz>-iq.flac
#     - subdir2 (subchannel 2)
#       - <iso8601 timestamp>-<frequency2 Hz>-iq.flac
#
# Copyright 2023 Franco Venturi K4VZ
#
# Version: 1.0 - Sun Dec 10 11:08:29 AM PST 2023

from collections import defaultdict
from configparser import ConfigParser
from datetime import datetime, timedelta
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


# https://stackoverflow.com/a/4628148
timedelta_regex = re.compile(r'((?P<days>\d+?)d)?((?P<hours>\d+?)h)?((?P<minutes>\d+?)m)?((?P<seconds>\d+?)s)?')
def parse_timedelta(delta):
    fields = timedelta_regex.fullmatch(delta)
    if not fields:
        print("invalid 'to' time delta:", delta, file=sys.stderr)
        sys.exit(1)
    fields = fields.groupdict()
    timedelta_params = {}
    for name, param in fields.items():
        if param:
            timedelta_params[name] = int(param)
    return timedelta(**timedelta_params)


timestamp_regex = re.compile(r'(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z')

def to_ts_from_delta(from_ts_str, to_ts_delta):
    ts = timestamp_regex.fullmatch(from_ts_str)
    if ts is None:
        print('invalid timestamp:', from_ts_str, file=sys.stderr)
        sys.exit(1)
    isots = f'{ts.group(1)}-{ts.group(2)}-{ts.group(3)}t{ts.group(4)}:{ts.group(5)}:{ts.group(6)}+00:00'
    from_ts = datetime.fromisoformat(isots)
    to_ts = from_ts + to_ts_delta
    return to_ts.strftime('%Y%m%dT%H%M%SZ')


def get_subchannels(inputdir, flac_file_pattern, from_ts=None, to_ts=None):
    flac_file_regex = re.compile(flac_file_pattern.replace('%TIMESTAMP%', r'(\d{8}T\d{6}Z)').replace('%FREQUENCY%', '(\d{5,9})'))

    subchannels = []
    timestamps = defaultdict(set)
    for subdir in os.listdir(inputdir):
        num_subchannel = len(subchannels)
        #print('subdir:', subdir)
        subdir_full = os.path.join(inputdir, subdir)
        frequency = None
        sample_rate = None
        num_channels = None
        for flac_file in sorted(os.listdir(subdir_full)):
            #print('flac_file:', flac_file)
            m = flac_file_regex.fullmatch(flac_file)
            if not m:
                continue
            ts = m.group(1)
            freq = m.group(2)
            if not ((from_ts is None or ts >= from_ts) and (to_ts is None or ts < to_ts)):
                continue
            if frequency is None:
                frequency = freq
            elif freq != frequency:
                print('multiple frequencies in the same subdirectory:', frequency, freq, file=sys.stderr)
                sys.exit(1)
            timestamps[ts].add(num_subchannel)
            # open the first file to get sample rate and number of channels
            if sample_rate is None:
                samples, sample_rate = sf.read(os.path.join(subdir_full, flac_file), dtype='int16')
                #print(flac_file, samples.shape, len(samples), sample_rate)
                num_channels = samples.shape[1]
        # if there are no valid flac files, skip the directory altogether
        if frequency is None:
            continue
        subchannels.append((subdir, frequency, sample_rate, num_channels))
    return subchannels, timestamps



# make sure all the subchannels have the same sample_rate and
# number of channels (1 or 2)
# make sure all the timestamps have all the channels
def check_subchannels_and_timestamps(subchannels, timestamps):
    sample_rate = subchannels[0][2]
    num_channels = subchannels[0][3]
    if not all(x[2] == sample_rate for x in subchannels):
        print('not all the subchannels have the expected sample rate of', sample_rate, file=sys.stderr)
        sys.exit(1)
    if not all(x[3] == num_channels for x in subchannels):
        print('not all the subchannels have the expected number of channels of', num_channels, file=sys.stderr)
        sys.exit(1)
    num_subchannels = len(subchannels)
    for timestamp, subchannels_in_ts in sorted(timestamps.items()):
        if len(subchannels_in_ts) != num_subchannels:
            for missing_subchannel_idx in sorted(set(range(num_subchannels)) - subchannels_in_ts):
                missing_subchannel = subchannels[missing_subchannel_idx]
                print('warning - no samples found for timestamp', timestamp, 'for subchannel', missing_subchannel[0], 'frequency', missing_subchannel[1], file=sys.stderr)
    return sample_rate, num_channels


def enrich_timestamps(timestamps, sample_rate):
    enriched_timestamps = []
    start_global_index = None
    for timestamp_str in sorted(timestamps):
        ts = timestamp_regex.fullmatch(timestamp_str)
        if ts is None:
            print('invalid timestamp:', timestamp_str, file=sys.stderr)
            sys.exit(1)
        isots = f'{ts.group(1)}-{ts.group(2)}-{ts.group(3)}T{ts.group(4)}:{ts.group(5)}:{ts.group(6)}+00:00'
        timestamp_num = datetime.fromisoformat(isots).timestamp()
        if start_global_index is None:
            start_global_index = int(timestamp_num * sample_rate)
        enriched_timestamps.append((timestamp_str, timestamp_num, int(timestamp_num * sample_rate) - start_global_index))
    # add number of expected samples
    for timestamp_idx in range(len(enriched_timestamps) - 1):
        timestamp_str, timestamp_num, next_sample = enriched_timestamps[timestamp_idx]
        expected_samples = enriched_timestamps[timestamp_idx + 1][2] - next_sample
        enriched_timestamps[timestamp_idx] = (timestamp_str, timestamp_num, next_sample, expected_samples)
    # assume last interval has the same length of the (last - 1)th
    timestamp_idx = len(enriched_timestamps) - 1
    timestamp_str, timestamp_num, next_sample = enriched_timestamps[timestamp_idx]
    enriched_timestamps[timestamp_idx] = (timestamp_str, timestamp_num, next_sample, enriched_timestamps[timestamp_idx - 1][3])
    return enriched_timestamps, start_global_index


# make sure that all the flac files have the expected number of samples
# based on their duration (computed as the difference between the current
# file timestamp and the next file timestamp) and the sample rate
def check_flac_files_sample_count(subchannels, timestamps, inputdir, flac_file_pattern):
    ok = True
    print('checking number of samples per file across all the subchannels. This will take a while', file=sys.stderr)
    for timestamp in timestamps:
        expected_samples = timestamp[3]
        for subchannel in subchannels:
            flac_file = flac_file_pattern.replace('%TIMESTAMP%', timestamp[0]).replace('%FREQUENCY%', subchannel[1])
            samples, sample_rate = sf.read(os.path.join(inputdir, subchannel[0], flac_file), dtype='int16')
            if len(samples) != expected_samples:
                print('Warning - flac file', flac_file, 'in subchannel', subchannel[0], 'has', len(samples), 'samples - expecting', expected_samples, 'samples', file=sys.stderr)
                ok = False
    return ok


def create_drf_metadata(subchannels, config, chdir, subdir_cadence_secs, sample_rate, start_global_index, uuid_str):
    metadatadir = os.path.join(chdir, 'metadata')
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
        'lat': np.single(float(config['metadata']['latitude'])),
        'long': np.single(float(config['metadata']['longitude'])),
        'center_frequencies': np.ascontiguousarray([float(x[1]) for x in subchannels])
    }
    # all all the remaining arguments as strings
    for k, v in config['metadata'].items():
        if k in ['latitude', 'longitude']:
            continue
        data_dict[k] = v
    do.write(sample, data_dict)


def create_drf_dataset(subchannels, timestamps, inputdir, flac_file_pattern, outputdir, config, sample_rate, num_channels, start_global_index):
    channel_name = config['global']['channel name']
    subdir_cadence_secs = int(config['global']['subdir cadence secs'])
    file_cadence_millisecs = int(config['global']['file cadence millisecs'])
    compression_level = int(config['global']['compression level'])
    dtype = 'i2'  # 16bit shorts

    # the output directory must already exist
    chdir = os.path.join(outputdir, channel_name)
    os.makedirs(chdir)

    uuid_str = uuid.uuid4().hex
    create_drf_metadata(subchannels, config, chdir, subdir_cadence_secs, sample_rate, start_global_index, uuid_str)

    print('writing Digital RF dataset. This will take a while', file=sys.stderr)

    with drf.DigitalRFWriter(chdir,
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

        for timestamp in timestamps:
            expected_samples = timestamp[3]
            samples = [None] * len(subchannels)
            for idx, subchannel in enumerate(subchannels):
                flac_file = flac_file_pattern.replace('%TIMESTAMP%', timestamp[0]).replace('%FREQUENCY%', subchannel[1])
                samples[idx], flac_file_sample_rate = sf.read(os.path.join(inputdir, subchannel[0], flac_file), dtype='int16')
                if flac_file_sample_rate != sample_rate:
                    print('sample rate for flac file', flac_file, 'in subchannel', subchannel[0], 'is', flac_file_sample_rate, '- expecting', sample_rate, file=sys.stderr)
                    return False
                flac_file_num_channels = samples[idx].shape[1]
                if flac_file_num_channels != num_channels:
                    print('num channels for flac file', flac_file, 'in subchannel', subchannel[0], 'is', flac_file_num_channels, '- expecting', num_channels, file=sys.stderr)
                    return False
                # FIXME
                flac_file_num_samples = samples[idx].shape[0]
                if flac_file_num_samples != expected_samples:
                    print('num samples for flac file', flac_file, 'in subchannel', subchannel[0], 'is', flac_file_num_samples, '- expecting', expected_samples, file=sys.stderr)
                    return False
            do.rf_write(np.hstack(samples, casting='no'))
            # hopefully deleting samples will free all the memory
            del samples

    return True


def main():
    configfile = sys.argv[0].replace('.py', '.conf')
    inputdir = None
    from_ts = None
    to_ts = None
    outputdir = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'c:i:f:t:o:v')
    except getopt.GetoptError as ex:
        print(ex, file=sys.stderr)
        sys.exit(1)
    for o, a in opts:
        if o == '-c':
            configfile = a
        elif o == '-i':
            inputdir = a
        elif o == '-f':
            from_ts = a
        elif o == '-t':
            if a.startswith('+'):
                to_ts = parse_timedelta(a[1:])
            else:
                to_ts = a
        elif o == '-o':
            outputdir = a
        elif o == '-v':
            global verbose
            verbose += 1

    if inputdir is None:
        print('missing input dir (-i) option', file=sys.stderr)
        sys.exit(1)

    if outputdir is None:
        print('missing output dir (-o) option', file=sys.stderr)
        sys.exit(1)

    if isinstance(to_ts, timedelta):
        to_ts = to_ts_from_delta(from_ts, to_ts)

    config = ConfigParser(interpolation=None)
    config.optionxform = str
    config.read(configfile)

    flac_file_pattern = config['global']['flac file pattern']

    subchannels, timestamps = get_subchannels(inputdir, flac_file_pattern, from_ts, to_ts)
    if len(subchannels) == 0:
        print("no subchannels (i.e. no valid flac files between 'from' timestamp and 'to' timestamp) found. Nothing to do.", file=sys.stderr)
        sys.exit(0)
    print('n subchannels:', len(subchannels), file=sys.stderr)
    if verbose >= 1:
        print(subchannels)
    sample_rate, num_channels = check_subchannels_and_timestamps(subchannels, timestamps)
    print('sample rate:', sample_rate, '- num channels:', num_channels, file=sys.stderr)
    timestamps, start_global_index = enrich_timestamps(timestamps, sample_rate)
    print('n timestamps:', len(timestamps), file=sys.stderr)
    if verbose >= 1:
        print(timestamps)
    print('start global index:', start_global_index, file=sys.stderr)
    # sort subchannels by frequency
    subchannels.sort(key=lambda x: float(x[1]))

    #ok = check_flac_files_sample_count(subchannels, timestamps, inputdir, flac_file_pattern)
    #print('check_flac_files_sample_count returned', ok, file=sys.stderr)

    ok = create_drf_dataset(subchannels, timestamps, inputdir, flac_file_pattern, outputdir, config, sample_rate, num_channels, start_global_index)
    print('create_drf_dataset returned', ok, file=sys.stderr)


if __name__ == '__main__':
    main()
