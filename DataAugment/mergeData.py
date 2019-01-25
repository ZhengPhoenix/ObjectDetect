import os
import argparse
import shutil
import numpy as np
import csv
import cv2

def parserArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Image path')
    parser.add_argument('-o', help='Output path')
    args = parser.parse_args()
    print('reading image from {}'.format(args.i))
    if not args.o:
        print('output dir not specify, exit')
        exit(0)
    return args.i, args.o


def main():
    image_path, output_path = parserArgument()

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    csvfile = open(os.path.join(output_path, 'merged.csv'), 'w')

    for file in os.listdir(image_path):
        if not file.endswith('jpg') and not file.endswith('JPG'):
            continue

        if not os.path.exists(os.path.join(image_path, file + '.csv')):
            continue

        spamwriter = csv.reader(open(os.path.join(image_path, file + '.csv')), delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        img = cv2.imread(os.path.join(image_path, file))
        mul_h = 300.0 / img.shape[1]
        mul_w = 300.0 / img.shape[0]
        img = cv2.resize(img, (300, 300))

        coordinate = []
        for row in spamwriter:
            if len(row) <= 0:
                continue
            coordinate.append([float(row[0]) * mul_h, float(row[1]) * mul_w])

        coordinate_resize = np.array(coordinate)
        coord = [str(i) for i in coordinate_resize.reshape(-1).tolist()]

        cv2.imwrite(os.path.join(output_path, file), img)

        csvfile.write('{} {} 300 0\n'.format(file, ' '.join(coord)))


if __name__ == '__main__':
    main()