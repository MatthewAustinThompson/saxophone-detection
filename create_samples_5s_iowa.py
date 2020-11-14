"""
Create samples from library

"""
import os
import re
import logging

import scipy.io.wavfile
import ffmpeg

from models.audio import Audio, SourceTooShortException

logger = logging.getLogger()
hdlr = logging.FileHandler('create_samples.log')
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


IN_DIR = './data/iowa_padded_mono_alto'
OUT_DIR = './data/5s/wav_sample/Iowa_ALTO'

TEMP_FILENAME = 'tempfile.wav'

SAMPLE_DURATION = 5  # seconds
SAMPLES_START_AT = 0
SAMPLE_RATE = 44100


def extract_fragment(data_, trackname_):
    """Takes data read by Scipy from WAV file and extracts smaller fragments """
    start_sample = SAMPLES_START_AT * SAMPLE_RATE
    end_sample = (SAMPLES_START_AT + SAMPLE_DURATION) * SAMPLE_RATE
    try:
        snip_data = Audio.snip(data_, start_sample, end_sample)
        # snip_data = Audio.s2m(snip_data)  # Mixes stereo down to mono
        # ALREADY MONO
        snip_name = os.path.join(OUT_DIR, trackname_)
        scipy.io.wavfile.write(snip_name, SAMPLE_RATE, snip_data)
    except SourceTooShortException:
        # e.g., Audio source too short for snip!
        return


if __name__ == '__main__':

    logger.info("======== STARTING NEW RUN ========")
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)
    tmp = os.path.join(OUT_DIR, TEMP_FILENAME)

    for fn in os.listdir(IN_DIR):
        name, ext = os.path.splitext(fn)
        if ext == '.wav':
            print(fn)
            src = os.path.join(IN_DIR, fn) # this is the source file
            try:
                # Scipy reads audio data from WAV files
                rate, data = scipy.io.wavfile.read(src)
                # rate is the sample rate, data is the data
                assert rate == SAMPLE_RATE
            except ValueError as err:
                # E.g., Unsupported bit depth: the wav file has
                # 24-bit data or File format b''... not understood.
                print("Non-conforming file")
                print(fn)
                logger.error("Non-conforming file {}".format(fn))
                continue
            except AssertionError:
                print("Non-conforming sample rate {}".format(rate))
                print(fn)
                logger.error("Non-conforming sample rate in {}"
                             .format(fn))
                continue
            extract_fragment(data, fn)

    if os.path.isfile(tmp):
        os.remove(tmp)