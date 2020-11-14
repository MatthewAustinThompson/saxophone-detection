# C. Cafiero

from scipy.signal import butter, sosfilt
import scipy.io.wavfile


def def_butter_high(cutoff, fs, order=2):
    """
    Define a Butterworth highpass filter
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    return butter(order, normal_cutoff, btype='high',
                  analog=False, output='sos')


def apply_butter_high(data, cutoff, fs, order=2):
    """
    Apply a Butterworth highpass filter to signal
    """
    sos = def_butter_high(cutoff, fs, order=order)
    return sosfilt(sos, data)


if __name__ == '__main__':
    # Scipy reads audio data from WAV files
    rate, data = scipy.io.wavfile.read('./test.wav')
    # rate is the sample rate, data is the data
    # NOTE: Sample rate of my input is 44100
    assert rate == 44100

    data = data.astype('float64')
    # print(data.dtype)

    fs = rate

    # Filter requirements.
    order = 2
    cutoff = 40  # desired cutoff frequency of the filter, Hz

    filtered = apply_butter_high(data, cutoff, fs, order=order)
    filtered = filtered.astype('int16')
    scipy.io.wavfile.write('filtered.wav', rate, filtered)
