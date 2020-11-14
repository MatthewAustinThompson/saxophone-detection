"""
Script to extract features from audio sample files, with a little plotting

C. Cafiero
"""

import os
import pickle

import matplotlib.pyplot as plt
import librosa
import librosa.display


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
#                './data/5s/wav_sample/Batch_14',
#                './data/5s/wav_sample/Batch_15',
#                './data/5s/wav_sample/Batch_TBP_01',
#                './data/5s/wav_sample/Batch_TBP_02',
#                './data/5s/wav_sample/Batch_TBP_03',
#                './data/5s/wav_sample/Batch_TBP_04',
#                './data/5s/wav_sample/Iowa_Alto']
SAMPLE_DIRS = ['./data/5s/wav_sample/Batch_11']

OUT_DIR = './data/5s/labeled/features_r02'


def extract_log_mel_spectogram(data_, rate_):
    """Log Mel spectrogram """
    lms = librosa.feature.melspectrogram(data_, rate_, n_mels=96)
    # Default is 128. Google says 96. ^
    return librosa.power_to_db(lms)


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
        name, ext = os.path.splitext(fn)
        if ext == '.wav':
            print(fn)
            in_file = os.path.join(sample_dir, fn)
            # out_waveform_file = os.path.join(SAMPLE_DIR, name + '_WF.png')
            # out_spectrogram_file = os.path.join(SAMPLE_DIR, name + '_SPEC.png')
            out_lms_file = os.path.join(OUT_DIR, name + '_PICKLE.lms')
            out_features_file = os.path.join(OUT_DIR, name + '_PICKLE.dat')
            features = {'zero_crossings': None,
                        'mfcc': None,
                        'spectral_centroids': None,
                        'spectral_rolloff': None,
                        'spectral_bandwidth': None,
                        'spectral_contrast': None,
                        'spectral_flatness': None}

            # Librosa uses 22050 as default sample rate. Use `sr` keyword
            # param to set sample rate if you don't want the default of 22050.
            data, rate = librosa.load(in_file, sr=44100)
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

            # log Mel spectrogram
            lms = extract_log_mel_spectogram(data, rate)
            # Write log mel spectrogram data to file
            with open(out_lms_file, 'wb') as fh:
                pickle.dump(lms, fh)

            '''
            # Plots
            try:
                librosa.display.specshow(features['mfcc'], sr=rate, x_axis='time')
                plt.savefig(out_mfcc_file)  # Must do this before calling show()
                # plt.show()
            except OverflowError:
                # E.g., matplotlib: In draw_markers: Exceeded cell block limit
                print("Unable to save MFCC plot: {}".format(out_mfcc_file))
            finally:
                plt.clf()

            try:
                librosa.display.waveplot(data, sr=rate)
                plt.savefig(out_waveform_file)  # Must do this before calling show()
                # plt.show()
            except OverflowError:
                # E.g., matplotlib: In draw_markers: Exceeded cell block limit
                print("Unable to save waveform plot: {}".format(out_waveform_file))
            finally:
                plt.clf()

            # spectrogram = librosa.amplitude_to_db(abs(librosa.stft(data)))
            # plt.figure(figsize=(10, 5))
            # librosa.display.specshow(spectrogram, sr=rate,
            #                          x_axis='time', y_axis='hz')
            # plt.colorbar()
            # plt.savefig(out_spectrogram_file)  # Must do this before calling show()
            # plt.show()
            '''
