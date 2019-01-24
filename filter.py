import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-c')
args = parser.parse_args()

validate = 'validate_sample'

if os.path.isdir(validate):
    shutil.rmtree(validate)
os.mkdir(validate)

count = args.c

for index, fn in enumerate(os.listdir('multipleBackgrounds')):
    if fn[0] == '.':
        continue

    if fn.endswith('csv'):
        continue

    if index > count:
        print('copy finished')
        exit()

    shutil.move(os.path.join('multipleBackgrounds', fn), validate)