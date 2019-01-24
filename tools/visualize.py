import os
import sys
import matplotlib.pyplot as plt
import cv2
import csv
import argparse
import shutil


def logProcess(num, total):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write('{} files processed, totally {}  files'.format(str(num), str(total)))
    sys.stdout.flush()
    sys.stdout.write('\r')

parser = argparse.ArgumentParser()
parser.add_argument('--sample')

path = parser.parse_args().sample

output = 'output'
if os.path.isdir(output):
    shutil.rmtree(output)
os.mkdir(output)

count = 0
total = len([sourcefile for sourcefile in os.listdir(path) if sourcefile.lower().endswith('jpg')])

for filename in os.listdir(path):
    if filename[0] == '.':
        continue
    if not filename.lower().endswith('jpg'):
        continue

    img = cv2.imread(os.path.join(path, filename))

    # Create figure and axes
    # fig,ax = plt.subplots(1)

    # Display the image
    plt.imshow(img)

    # read csv
    csvfile = open(os.path.join(path, filename) + '.csv', 'r')
    spamwritter = csv.reader(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    qt = []
    for row in spamwritter:
        if len(row) > 0:
            qt.append(row)

    # Create a Rectangle patch
    pgon = plt.Polygon(qt, color='r', alpha=0.5, fill=True)

    # rect = plt.Rectangle((20, 750), 400, 150, color='r', alpha=0.5)

    # Add the patch to the Axes
    plt.gca().add_patch(pgon)
    # plt.gca().add_patch(rect)

    plt.savefig(os.path.join(output, filename))
    plt.close()
    count += 1
    logProcess(count, total)
