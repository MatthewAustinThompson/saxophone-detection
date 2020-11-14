"""
Import from this file if you're doing stuff with TENOR
Be sure to verify all constants and update NUM_COMPONENTS with result
of most recent PCA for TENOR

C. Cafiero
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from _common import NUM_LABEL_COLS, RANDOM_SEED

NUM_COMPONENTS = 36
TEST_SIZE = 0.20  # of total data
VALIDATION_SIZE = 0.2  # of training set

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

num_x_cols = df_filtered.shape[1] - NUM_LABEL_COLS - 1
# ^ last bit to adjust for zero indexing
data = df_filtered.iloc[:, 1:num_x_cols].to_numpy()
# ^ these are the features (we start at 1 to leave out track name)
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

target = df_filtered['tenr'].to_numpy().ravel()
# ^ these are the labels

pca = PCA(n_components=NUM_COMPONENTS)
pca.fit(data)
d = pca.transform(data)
