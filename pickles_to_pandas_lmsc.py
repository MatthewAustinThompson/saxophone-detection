"""
Script to read data from pickes, marry it up with features from CSV,
and produce a Pandas dataframe. Place burden on files for now. Expect a
1-to-1 relationship between pickles and rows in CSV files containing labels,
otherwise chaos will ensue.

WE MAY NEED TO SAVE ACROSS MULTIPLE PICKLES, maybe 2000 records per pickle.

C. Cafiero
"""

import os
import csv
import pickle

import pandas as pd

DIR = './data/5s/labeled/features_r02'
LABEL_FILENAME = 'labels.csv'


def lms_pickles_to_pandas(dir_=DIR):
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
    out_file_counter = 0
    counter = 0
    for f in os.scandir(dir_):
        # Since we may delete local .wav files, drive this from the pickles
        if f.path.endswith(".lmsc"):
            counter +=1
            # print(f.name)
            name = f.name.replace('_PICKLE.lmsc', '.wav')
            # ^ This is what should match name in the labels array for lookup

            rrow = []  # create placeholder for this row's records
            with open(os.path.join(f.path), "rb") as pfh:
                pdata = pickle.load(pfh)  # load the pickle
                if first:
                    col_names.append('filename')
                rrow.append(name)
                for band_num, lms_band in enumerate(pdata):
                    for t, val in enumerate(lms_band):
                        if first:
                            col_names.append('lmsc_{}_{}'.format(band_num, t))
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
            if not counter % 1000:
                print(counter)
                print("Trying to make DataFrame...")
                df = pd.DataFrame(data, columns=col_names)
                print("Filtering...")
                df = df[df['excl'] == '0']
                print("Saving to pickle {}...".format(out_file_counter))
                df.to_pickle('./data/5s/labeled/features_r02/lmsc_data_{}.pkl'
                             .format(out_file_counter))
                out_file_counter += 1
                data = []


if __name__ == '__main__':
    print("Starting...")
    lms_pickles_to_pandas()
    print("Done")
