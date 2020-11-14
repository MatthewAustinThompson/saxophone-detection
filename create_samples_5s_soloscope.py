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

# This is a special list with only one file, a 51 minute live concert of solo
# tenor sax by Sonny Rollins
FILE_LIST = './data/selected_files/tenor_booster_pack_03.txt'
OUT_DIR = './data/5s/wav_sample/Batch_TBP_03'

TEMP_FILENAME = 'tempfile.wav'

SAMPLE_DURATION = 5  # seconds
SAMPLES_START_AT = [30 + x for x in range(0, 50*60, 30)]
SAMPLE_RATE = 44100


def extract_fragments(data_, album_, artist_, trackname_):
    """Takes data read by Scipy from WAV file and extracts smaller fragments """
    for idx, sample_start_s in enumerate(SAMPLES_START_AT):
        start_sample = sample_start_s * SAMPLE_RATE
        end_sample = (sample_start_s + SAMPLE_DURATION) * SAMPLE_RATE
        try:
            snip_data = Audio.snip(data_, start_sample, end_sample)
            snip_data = Audio.s2m(snip_data)  # Mixes stereo down to mono
            snip_name = os.path.join(OUT_DIR, "{}_{}_{}_{}.wav"
                                     .format(artist_, album_, trackname_, idx))
            scipy.io.wavfile.write(snip_name, SAMPLE_RATE, snip_data)
        except SourceTooShortException:
            # e.g., Audio source too short for snip!
            return


if __name__ == '__main__':

    logger.info("======== STARTING NEW RUN ========")
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)
    tmp = os.path.join(OUT_DIR, TEMP_FILENAME)

    r = re.compile(r'^CD[0-9]+$|^cd[0-9]+$|^[D,d]isk[0-9]+$|^[D,d]isc[0-9]+$')

    with open(FILE_LIST) as flh:
        file_path = flh.readline()
        while file_path:
            file_path = file_path.strip()
            print(file_path)
            if os.path.isfile(file_path):
                disc = None
                p, file_name = os.path.split(file_path)
                p, album = os.path.split(p)
                if r.match(album):
                    disc = album
                    p, album = os.path.split(p)
                p, artist = os.path.split(p)

                src = file_path  # this is the source file
                try:
                    input_ = ffmpeg.input(src)  # read source file
                    out = ffmpeg.output(input_, tmp)
                    stdout, stderr = out.run(overwrite_output=True)
                except Exception as e:
                    logger.error(str(e))
                    logger.error("FFMPEG error on {}".format(src))
                    try:
                        os.remove(tmp)
                    except FileNotFoundError:
                        pass
                    file_path = flh.readline()
                    continue
                if stderr is not None:
                    try:
                        os.remove(tmp)
                    except FileNotFoundError:
                        continue
                else:
                    try:
                        # Scipy reads audio data from WAV files
                        rate, data = scipy.io.wavfile.read(tmp)
                        # rate is the sample rate, data is the data
                        assert rate == SAMPLE_RATE
                    except ValueError as err:
                        # E.g., Unsupported bit depth: the wav file has
                        # 24-bit data or File format b''... not understood.
                        print("Non-conforming file")
                        print(file_path)
                        logger.error("Non-conforming file {}".format(file_path))
                        file_path = flh.readline()
                        continue
                    except AssertionError:
                        print("Non-conforming sample rate {}".format(rate))
                        print(file_path)
                        logger.error("Non-conforming sample rate in {}"
                                     .format(file_path))
                        file_path = flh.readline()
                        continue
                    # try:
                    #    # Try to take three samples
                    trackname, ext = os.path.splitext(file_name)
                    if disc is not None:
                        album = album + '_' + disc
                    extract_fragments(data, album, artist, file_name)
                    # except Exception as e:
                    #     print(str(e))
                    #     continue
            else:
                print("File not found: {}".format(file_path))
            file_path = flh.readline()

    if os.path.isfile(tmp):
        os.remove(tmp)