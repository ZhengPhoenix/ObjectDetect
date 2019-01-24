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

output = 'visual_output'
if os.path.isdir(output):
    shutil.rmtree(output)
os.mkdir(output)

count = 0
total = len([sourcefile for sourcefile in os.listdir(path) if sourcefile.lower().endswith('jpg')])

if not os.path.isfile(os.path.join(path, 'merged.csv')):
    print('csv file not exist')
    exit(0)

spamreader = csv.reader(open(os.path.join(path, 'merged.csv')), delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

for row in spamreader:
    if len(row) <= 0:
        continue

    filename = str(row[0])
    coord = [float(i) for i in row[1:-1]]
    coord = [coord[i*2:(i+1)*2] for i in range(round(len(coord)/2))]

    if not os.path.isfile(os.path.join(path, filename)):
        continue

    img = cv2.imread(os.path.join(path, filename))

    # Create figure and axes
    # fig,ax = plt.subplots(1)

    # Display the image
    plt.imshow(img)

    # Create a Rectangle patch
    pgon = plt.Polygon(coord, color='r', alpha=0.5, fill=True)

    # rect = plt.Rectangle((20, 750), 400, 150, color='r', alpha=0.5)

    # Add the patch to the Axes
    plt.gca().add_patch(pgon)
    # plt.gca().add_patch(rect)

    plt.savefig(os.path.join(output, filename))
    plt.close()
