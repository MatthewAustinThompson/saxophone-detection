# M. Thompson

import os
import csv
import pickle

DIR = './data/5s/labeled/features_r02'
LABEL_FILENAME = 'labels.csv'

def label_helper(dir_=DIR):
    labels = []

    with open(os.path.join(dir_, LABEL_FILENAME)) as fh:
        reader = csv.reader(fh)
        next(reader)  # skip the header
        for row in reader:
            if row:  # exclude blanks, if any
                labels.append(row[0])

    for f in os.scandir(dir_):
        if f.path.endswith(".dat"):
            name = f.name.replace('_PICKLE.dat', '.wav')
        for row in labels:
            if row == name:
                if name.endswith(".wav"):
                    name = f.name.replace('.wav', '_PICKLE.dat')
                os.rename(DIR + '/' + name, DIR + '/selected/' + name)
                print("Success!")
            else:
                print("File not found")
if __name__ == '__main__':
    label_helper(DIR)
