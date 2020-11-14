"""
Script to extract features from audio sample files, with a little plotting

C. Cafiero
"""

import os
import pickle

import matplotlib.pyplot as plt
import librosa
import librosa.display


SAMPLE_DIRS = ['./data/5s/wav_sample/Batch_01',
               './data/5s/wav_sample/Batch_02',
               './data/5s/wav_sample/Batch_03',
               './data/5s/wav_sample/Batch_04',
               './data/5s/wav_sample/Batch_05',
               './data/5s/wav_sample/Batch_06',
               './data/5s/wav_sample/Batch_07',
               './data/5s/wav_sample/Batch_08',
               './data/5s/wav_sample/Batch_09',
               './data/5s/wav_sample/Batch_10',
               './data/5s/wav_sample/Batch_11',
               './data/5s/wav_sample/Batch_14',
               './data/5s/wav_sample/Batch_15',
               './data/5s/wav_sample/Batch_TBP_01',
               './data/5s/wav_sample/Batch_TBP_02',
               './data/5s/wav_sample/Batch_TBP_03',
               './data/5s/wav_sample/Batch_TBP_04',
               './data/5s/wav_sample/Iowa_Alto']
# SAMPLE_DIRS = ['./data/5s/wav_sample/test']

OUT_DIR = './data/5s/labeled/features_r02'
# OUT_DIR = './data/5s/wav_sample/test'


def extract_log_mel_spectogram_compressed(data_, rate_):
    """Log Mel spectrogram
    Force shape to 96 x 96 by setting appropriate hop length
    We have SR = 44100, t = 5s --> 220,500 samples.
    """
    lms = librosa.feature.melspectrogram(data_, rate_,
                                         hop_length=2297, n_mels=96)
    # Default is 128. Google says 96. ^
    lms = librosa.power_to_db(lms)
    return lms


if __name__ == '__main__':
    for sample_dir in SAMPLE_DIRS:
        for fn in os.listdir(sample_dir):
            name, ext = os.path.splitext(fn)
            if ext == '.wav':
                print(fn)
                in_file = os.path.join(sample_dir, fn)
                out_lms_compressed_file = os.path.join(OUT_DIR, name + '_PICKLE.lmsc')
                # Librosa uses 22050 as default sample rate. Use `sr` keyword
                # param to set sample rate if you don't want the default of 22050.
                data, rate = librosa.load(in_file, sr=44100)
                # It also appears that Librosa scales values to floats.

                # Features
                # log Mel spectrogram
                lms = extract_log_mel_spectogram_compressed(data, rate)
                # Write log mel spectrogram data to file
                with open(out_lms_compressed_file, 'wb') as fh:
                    pickle.dump(lms, fh)
