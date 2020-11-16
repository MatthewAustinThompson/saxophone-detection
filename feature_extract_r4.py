"""
Script to extract features from audio sample files
REVISION 3
"""

import os
import pickle

import librosa
import scipy.io.wavfile

from scipy.signal import wiener

# SAMPLE_DIRS = ['./data/5s/wav_sample/Batch_01',
#                './data/5s/wav_sample/Batch_02',
#                './data/5s/wav_sample/Batch_03',
#                './data/5s/wav_sample/Batch_04',
#                './data/5s/wav_sample/Batch_05',
#                './data/5s/wav_sample/Batch_06',
#                './data/5s/wav_sample/Batch_07',
#                './data/5s/wav_sample/Batch_08',
#                './data/5s/wav_sample/Batch_09',
#                './data/5s/wav_sample/Batch_10',
#                './data/5s/wav_sample/Batch_11',
#                './data/5s/wav_sample/Batch_14',
#                './data/5s/wav_sample/Batch_15',
#                './data/5s/wav_sample/Batch_TBP_01',
#                './data/5s/wav_sample/Batch_TBP_02',
#                './data/5s/wav_sample/Batch_TBP_03',
#                './data/5s/wav_sample/Batch_TBP_04',
#                './data/5s/wav_sample/Iowa_Alto']

SAMPLE_DIRS = ['./data/5s/wav_sample/Batch_05',
               './data/5s/wav_sample/Batch_06']
OUT_DIR = './data/5s/labeled/features_r04'

TEMP_FILENAME = 'temp.wav'


def extract_zero_crossings(data_):
    """Extract number of zero crossings """
    zero_crossings = librosa.zero_crossings(data_, pad=False)
    return sum(zero_crossings)


def extract_mfcc(data_, rate_):
    """Extract MFCC (mel frequency cepstral coefficients).
    Note that Librosa will default to extracting 20 series
    (n_mfcc=20). See docs for librosa.feature.mfcc """
    return librosa.feature.mfcc(data_, sr=rate_)


def extract_spectral_centroids(data_, rate_):
    """Extract spectral centroids.
    Here we're accepting defaults n_fft=2048, hop_length=512
    See docs for librosa.feature.spectral_centroid """
    return librosa.feature.spectral_centroid(data_, sr=rate_)


def extract_spectral_rolloff(data_, rate_):
    """Extract spectral rolloff """
    return librosa.feature.spectral_rolloff(data_, sr=rate_)


def extract_spectral_bandwidth(data_, rate_):
    """Extract spectral bandwidth """
    return librosa.feature.spectral_bandwidth(data_, sr=rate_)


def extract_spectral_contrast(data_, rate_):
    """Extract spectral contrast """
    return librosa.feature.spectral_contrast(data_, sr=rate_)


def extract_spectral_flatness(data_):
    """Extract spectral flatness """
    return librosa.feature.spectral_flatness(data_)


for sample_dir in SAMPLE_DIRS:
    for fn in os.listdir(sample_dir):
        if fn != TEMP_FILENAME:
            name, ext = os.path.splitext(fn)
            if ext == '.wav':
                print(fn)
                in_file = os.path.join(sample_dir, fn)
                out_features_file = os.path.join(OUT_DIR, name + '_PICKLE.dat')
                features = {'zero_crossings': None,
                            'mfcc': None,
                            'spectral_centroids': None,
                            'spectral_rolloff': None,
                            'spectral_bandwidth': None,
                            'spectral_contrast': None,
                            'spectral_flatness': None}

                # First apply wiener filter and save to temp file
                rate, data = scipy.io.wavfile.read(in_file)
                assert rate == 44100
                data = data.astype('float64')
                filtered = wiener(data)
                filtered = filtered.astype('int16')
                scipy.io.wavfile.write(TEMP_FILENAME, rate, filtered)

                # Librosa uses 22050 as default sample rate. Use `sr` keyword
                # param to set sample rate if you don't want the default of 22050.
                data, rate = librosa.load(TEMP_FILENAME, sr=44100)
                # It also appears that Librosa scales values to floats.

                # Features
                features['zero_crossings'] = extract_zero_crossings(data)
                features['mfcc'] = extract_mfcc(data, rate)
                features['spectral_centroids'] = extract_spectral_centroids(data, rate)
                features['spectral_rolloff'] = extract_spectral_rolloff(data, rate)
                features['spectral_bandwidth'] = extract_spectral_bandwidth(data, rate)
                features['spectral_contrast'] = extract_spectral_contrast(data, rate)
                features['spectral_flatness'] = extract_spectral_flatness(data)
                # Write features to file
                with open(out_features_file, 'wb') as fh:
                    pickle.dump(features, fh)
