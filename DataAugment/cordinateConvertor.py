import os
import csv
import sys
import argparse

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',
                        help='specify source path')
    parser.add_argument('-f',
                        help='specify test file')

    args = parser.parse_args()
    isSingleFile = args.f is not None

    if args.source is None:
        print('invalid argument source')
        exit()

    dest = os.path.join(os.path.abspath(os.path.join(os.getcwd(), args.source)), 'cord_output')

    os.chdir(args.source)
    if isSingleFile:
        edgeFile = args.f
        parseSingleFile(edgeFile, dest)


def parseSingleFile(fileNode, outputPath):
    data = open(fileNode, 'r').read()



if __name__ == '__main__':
    main(sys.argv)