"""
Import from this file if you're doing stuff with

        ALL SAXES MFCC
		NO OTHER REEDS, BUT BRASS OK

C. Cafiero and M. Thompson
"""

import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

from _common import NUM_LABEL_COLS

TEST_SIZE = 0.20  # of total data
VALIDATION_SIZE = 0.2  # of training set
BANDS = 20
TIME_SLICES = 431

df = pd.read_pickle('./data/5s/labeled/features_r02/all_data.pkl')
df_filtered = df

# exclude records we want to exclude
df_filtered = df_filtered[df_filtered['clrt'] == '0']
df_filtered = df_filtered[df_filtered['othr'] == '0']

mfccs = df_filtered.filter(regex=('mfcc_*'))

data = mfccs.to_numpy()

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

df_filtered['sax'] = df_filtered[['sop', 'alto', 'tenr', 'tora', 'bari']].max(axis=1)
df_filtered['sax'] = df_filtered['sax'].astype(int)
target = df_filtered['sax'].to_numpy().ravel()
# ^ these are the labels

