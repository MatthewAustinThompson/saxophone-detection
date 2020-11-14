"""
Import from this file if you're doing stuff with TENOR MFCC

C. Cafiero
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0.20  # of total data
VALIDATION_SIZE = 0.2  # of training set
BANDS = 20
TIME_SLICES = 431

df = pd.read_pickle('./data/5s/labeled/features_r02/all_data.pkl')
df_filtered = df

# exclude records we want to exclude
df_filtered = df_filtered[df_filtered['sop'] == '0']
df_filtered = df_filtered[df_filtered['alto'] == '0']
df_filtered = df_filtered[df_filtered['tora'] == '0']
df_filtered = df_filtered[df_filtered['bari'] == '0']
df_filtered = df_filtered[df_filtered['clrt'] == '0']
df_filtered = df_filtered[df_filtered['othr'] == '0']
df_filtered = df_filtered[df_filtered['trmp'] == '0']
df_filtered = df_filtered[df_filtered['trmb'] == '0']
df_filtered = df_filtered[df_filtered['otrb'] == '0']

mfccs = df_filtered.filter(regex=('mfcc_*'))

data = mfccs.to_numpy()

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

target = df_filtered['tenr'].to_numpy().ravel()
# ^ these are the labels
