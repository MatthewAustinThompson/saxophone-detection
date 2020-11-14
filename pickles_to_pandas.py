"""
Script to read data from pickes, marry it up with features from CSV,
and produce a Pandas dataframe. Place burden on files for now. Expect a
1-to-1 relationship between pickles and rows in CSV files containing labels,
otherwise chaos will ensue.

C. Cafiero
"""

import os
import csv
import pickle

import pandas as pd

DIR = './data/5s/labeled/features_r02'
LABEL_FILENAME = 'labels.csv'


def pickles_to_pandas(dir_=DIR):
    """Read from a bunch of pickles out in disk land and make a lovely
    Pandas data frame """
    labels = []
    # Read csv file which contains filename and labels
    with open(os.path.join(dir_, LABEL_FILENAME)) as fh:
        reader = csv.reader(fh)
        next(reader)  # skip the header
        for row in reader:
            if row:  # exclude blanks, if any
                labels.append(row)

    # If this is the first time through, we'll want to handle column headings
    # for the Pandas dataframe, hence this flag
    first = True

    col_names = []
    data = []
    for f in os.scandir(dir_):
        # Since we may delete local .wav files, drive this from the pickles
        if f.path.endswith(".dat"):
            name = f.name.replace('_PICKLE.dat', '.wav')
            # ^ This is what should match name in the labels array for lookup

            rrow = []  # create placeholder for this row's records
            with open(os.path.join(f.path), "rb") as pfh:
                pdata = pickle.load(pfh)  # load the pickle
                if first:
                    col_names.append('filename')
                rrow.append(name)
                if first:
                    col_names.append('zeros')
                rrow.append(pdata['zero_crossings'])
                for band_num, mfcc_band in enumerate(pdata['mfcc']):
                    for t, val in enumerate(mfcc_band):
                        if first:
                            col_names.append('mfcc_{}_{}'.format(band_num, t))
                        rrow.append(val)
                for t, val in enumerate(pdata['spectral_centroids'][0]):
                    if first:
                        col_names.append('spc_{}'.format(t))
                    rrow.append(val)
                for t, val in enumerate(pdata['spectral_rolloff'][0]):
                    if first:
                        col_names.append('spr_{}'.format(t))
                    rrow.append(val)
                for t, val in enumerate(pdata['spectral_bandwidth'][0]):
                    if first:
                        col_names.append('spb_{}'.format(t))
                    rrow.append(val)
                for t, val in enumerate(pdata['spectral_contrast'][0]):
                    if first:
                        col_names.append('spx_{}'.format(t))
                    rrow.append(val)
                for t, val in enumerate(pdata['spectral_flatness'][0]):
                    if first:
                        col_names.append('spf_{}'.format(t))
                    rrow.append(val)
                if first:
                    for x in ['sop', 'alto', 'tenr', 'tora', 'bari',
                              'clrt', 'othr', 'trmp', 'trmb', 'otrb',
                              'ext', 'excl', 'batch']:
                        col_names.append(x)

                found_match = False
                for row in labels:
                    # Iterate through rows in labels to find match.
                    # Not efficient, I know. Beat me up in code review.
                    if row[0] == name:
                        found_match = True
                        for x in row[1:]:
                            rrow.append(x)
                if not found_match:  # warn
                    print("Could not find match in labels for {}".format(name))
                    for x in row[1:]:
                        rrow.append(-1)
            data.append(rrow)
            first = False

    df = pd.DataFrame(data, columns=col_names)
    return df


if __name__ == '__main__':
    df = pickles_to_pandas()
    df = df[df['excl'] == '0']
    # DON'T use .dat extension if saving to same directory as other pickles!
    df.to_pickle('./data/5s/labeled/features_r02/all_data.pkl')
    print(df.head(10))
