"""
Convert stereo WAV files to mono WAV files
"""

import numpy as np


class SourceTooShortException(Exception):
    pass


class Audio:
    """Utilities for audio data (all data as Numpy array) """

    @staticmethod
    def s2m(data):
        """
        Convert stereo data to mono data
        """
        assert data.dtype == np.int16  # The type of data
        # assert data.max() <= 32767
        # assert data.min() >= -32768
        # Mix down stereo to mono
        data = data.astype(float)
        data = (data[:, 0] + data[:, 1]) / 2
        data = data.astype(np.int16)
        # For some reason, writing floats does not work. I get terrible
        # distortion and artifacts. But casting to np.int16 works just fine.
        # I think this is because scipy.io.wavefile.write chooses bits-per-
        # sample and PCM/float based on data type, and we want to stick to a
        # uniform format. See docs for scipy.io.wavfile.write
        return data

    @staticmethod
    def snip(data, start, end):
        """Take a snip from a data stream """
        if end > len(data):
            raise SourceTooShortException("Audio source too short for snip!")
        return data[start:end]

    @staticmethod
    def sec2sample(sec, rate):
        """Convert whole second to sample """
        return sec * rate
