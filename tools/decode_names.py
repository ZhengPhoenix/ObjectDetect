import os
import sys

import cv2
import numpy as np
import re

input_path = sys.argv[1]
# output_path = 'results'
output_path = sys.argv[2]

fn_lst = os.listdir(input_path)
for fn in fn_lst:

    path = os.path.join(input_path, fn)
    image = cv2.imread(path)

    info_lst = os.path.splitext(fn)[0]
    kw = re.compile(r'coordinate_1__([0-9E/.-]+)_([0-9E/.-]+)-coordinate_2__([0-9E/.-]+)_([0-9E/.-]+)-coordinate_3__([0-9E/.-]+)_([0-9E/.-]+)-coordinate_4__([0-9E/.-]+)_([0-9E/.-]+)')
    print info_lst
    if len(info_lst) < 40:
        continue

    points = kw.findall(info_lst)
    for p in points[0]:
        print p
    points = [int(float(p)) for p in points[0]]
    points = np.array(points).astype('int32').reshape(-1,2)
    result = cv2.drawContours(image, [points], -1, (0, 255, 0), 4)
    save_path = os.path.join(output_path, fn)
    cv2.imwrite(save_path, result)
    # cv2.imshow('result', result)
    # k = cv2.waitKey(0)
    # if ord('q') == 0xFF & k:
    #     break

