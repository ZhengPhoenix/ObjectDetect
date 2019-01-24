import os
import sys

import cv2
import numpy as np
import re

def interpret_name(file_name):
    info_lst = os.path.splitext(file_name)[0]
    kw = re.compile(r'coordinate_1__([0-9E/.-]+)_([0-9E/.-]+)-coordinate_2__([0-9E/.-]+)_([0-9E/.-]+)-coordinate_3__([0-9E/.-]+)_([0-9E/.-]+)-coordinate_4__([0-9E/.-]+)_([0-9E/.-]+)')
    return  list(kw.findall(info_lst)[0])


folder_input_path = sys.argv[1]
label_txt_path = sys.argv[2]

file_labels = open(label_txt_path, 'w')

for i, fn in enumerate(os.listdir(folder_input_path)):
    print fn
    # print interpret_name(fn)
    coordinates = ' '.join(interpret_name(fn))
    line_label = '{} {}\n'.format(fn, coordinates)
    file_labels.write(line_label)

