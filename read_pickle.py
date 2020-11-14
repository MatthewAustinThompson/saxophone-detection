"""
Read a pickle. Just a test.

C. Cafiero
"""

import os
import pickle

import numpy as np

DIR = "./data/5s/wav_sample/test"

for f in os.scandir(DIR):
    if f.path.endswith(".lmsc"):
        with open(os.path.join(f.path), "rb") as pfh:
            data = pickle.load(pfh)
            for key, value in data.items():
                print("-" * 40)
                print(key)
                print(type(value))
                if isinstance(value, np.ndarray):
                    print(value.shape)
