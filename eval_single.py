import os
import sys
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tools.net import Net

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',
                        help='specify eval jpg')
    parser.add_argument('--model',
                        help='specify model path')

    args = parser.parse_args()

    if not args.file or not args.model:
        print('invalid argument')
        exit()

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

    image = cv2.imread(args.file, 3)

    points, time = model.predict_points(image)

    print(np.sort(points, 0))

    source = cv2.imread(args.file)
    plt.imshow(source)
    pgon = plt.Polygon(points, color='r', alpha=0.5, fill=True)
    plt.gca().add_patch(pgon)
    plt.show()


if __name__ == '__main__':
    main(sys.argv)