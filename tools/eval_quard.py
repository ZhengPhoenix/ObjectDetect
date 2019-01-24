import os
import sys
import argparse
import shutil

import cv2
import numpy as np

from net import Net

import matplotlib.pyplot as plt


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',
                        help='specify eval jpg')
    parser.add_argument('--model',
                        help='specify model path')
    parser.add_argument('--dest',
                        help='specify output path')

    args = parser.parse_args()

    if not args.source or not args.model:
        print('invalid argument')
        exit()

    if os.path.isdir(args.dest):
        shutil.rmtree(args.dest)
    os.mkdir(args.dest)
    dest = os.path.join(os.getcwd(), args.dest)
    
    err = os.path.join(os.getcwd(), args.source+'_err')
    os.mkdir(err)

    modelH5 = ''
    modelNpy = ''
    for name in os.listdir(args.model):
        if name.endswith('h5'):
            modelH5 = os.path.join(args.model, name)
        if name.endswith('npy'):
            modelNpy = os.path.join(args.model, name)

    if not modelNpy or not modelH5:
        print('model not found')
        exit()

    model = Net(modelH5, modelNpy)

    totalSample = len([sourcefile for sourcefile in os.listdir(args.source) if sourcefile[0] != '.'])
    trueP = 0
    os.chdir(args.source)
    for index, fileNode in enumerate(os.listdir('.')):
        image = cv2.imread(fileNode, 3)

        if image is None:
            continue

        shape = [image.shape[1], image.shape[0]]
        points, time = model.predict_points(image)

        if not checkContainQuard(points, shape):
            source = cv2.imread(fileNode)
            plt.imshow(source)
            pgon = plt.Polygon(points, color='r', alpha=0.5, fill=True)
            plt.gca().add_patch(pgon)
            plt.savefig(os.path.join(dest, fileNode))
            plt.close()
            trueP += 1
        else:
            shutil.copy(fileNode, err)

        logProcess(index, totalSample)

    print('parsed finished with {} true positive files'.format(trueP))


def logProcess(num, total):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write('{} files processed, totally {}  files'.format(str(num), str(total)))
    sys.stdout.flush()
    sys.stdout.write('\r')

def checkContainQuard(points, shape):
    result = 0
    [row,col] = points.shape

    for i in range(row):
        if all(v == 0 for v in points[i]):
            result += 1
            continue
        if np.array_equal(shape, points[i]):
            result += 1
            continue

    return result > 1


if __name__ == '__main__':
    main(sys.argv)