"""
Import from this file if you're doing stuff with

        TENOR LOG MEL SPECTROGRAM
        COMPRESSED to 96 x 96
		NO OTHER REEDS, BUT BRASS OK

C. Cafiero
"""

import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

from _common import NUM_LABEL_COLS

TEST_SIZE = 0.20  # of total data
VALIDATION_SIZE = 0.2  # of training set
BANDS = 96
TIME_SLICES = 96

pnums = [0, 1, 2, 3, 4, 5, 6, 7, 8]
DIR = './data/5s/labeled/features_r02/'

for n in pnums:
    fn = 'lmsc_data_{}.pkl'.format(n)
    print("Reading lms_data_{}.pkl...".format(n))
    df = pd.read_pickle(os.path.join(DIR, fn))

    # exclude records we want to exclude
    df = df[df['sop'] == '0']
    df = df[df['alto'] == '0']
    df = df[df['tora'] == '0']
    df = df[df['bari'] == '0']
    df = df[df['clrt'] == '0']
    df = df[df['othr'] == '0']

    print(df.shape)
    if n == 0:
        master = df
    else:
        print("Appending {}...".format(n))
        master = master.append(df)


print("Making labels...")
target = master['tenr'].to_numpy().ravel()
# ^ these are the labels

print(master.shape)
print("Selecting columns...")
lmss = master.iloc[:, 1:]
num_x_cols = lmss.shape[1] - NUM_LABEL_COLS
lmss = lmss.iloc[:, 0:num_x_cols]
print(lmss.shape)

# print(lmss.head())
print("Trying to make numpy...")
data = lmss.to_numpy()

print("Applying scaler...")
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

print("Done")
