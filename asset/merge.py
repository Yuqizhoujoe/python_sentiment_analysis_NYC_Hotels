import os

os.getcwd()
filenames = ['text_merge.txt', 'Hotel Pennsylvania.txt']
with open('merge.txt', 'w') as f:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                f.write(line)