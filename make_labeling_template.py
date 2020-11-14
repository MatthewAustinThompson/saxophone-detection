"""Loop through all samples in a directory and create a CSV file that can
be used for human labeling of files. Note that this produces an unordered
 list (since os.listdir yields abitrary order). So you'll probably want
 to sort the CSV in Numbers or Excel or something before labeling samples.

 C. Cafiero
 """

import os
import csv

SAMPLE_DIR = './data/5s/wav_sample/Batch_TBP_04'
SHORT_LABEL = 'tpb04'
out_filename = os.path.join(SAMPLE_DIR, 'labels_batch_tpb_04.csv')

with open(out_filename, "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['filename',
                     'sop', 'alto', 'tenr', 'tora', 'bari', 'clrt', 'othr',
                     'trmp', 'trmb', 'otrb',
                     'ext', 'excl', 'batch'])
    for fn in os.listdir(SAMPLE_DIR):
        name, ext = os.path.splitext(fn)
        if ext == '.wav':
            print(fn)
            writer.writerow([fn, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, SHORT_LABEL])
