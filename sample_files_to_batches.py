"""Docstring or regret

C. Cafiero

"""

import os

FILE_LIST = './data/selected_files/to_do_shuffled.txt'
START_NUM = 14
OUT_DIR = './data/selected_files/'

with open(FILE_LIST) as flh:
    lines = flh.readlines()
    num_lines = len(lines)
    if num_lines % 100:
        num_batches = (num_lines // 100) + 1
    else:
        num_batches = num_lines // 100

    start = 0
    end = 99
    for i in range(num_batches):
        batch_num = START_NUM + i
        print(batch_num)
        outfile = 'sample_files_shuffled_BATCH_{}.txt'.format(batch_num)
        outfile = os.path.join(OUT_DIR, outfile)
        with open(outfile, 'w') as ofh:
            ofh.writelines(lines[start:end])
        start = start + 100
        end = end + 100
