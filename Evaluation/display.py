import os
import sys
import time

import cv2
import numpy as np

from net import Net

current_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(current_dir,".."))

test_path = sys.argv[1]
model_path = sys.argv[2]
mean_train_path = os.path.join(os.sep.join(model_path.split(os.sep)[:-2]),
                               "document_mean_train.npy")
output_dir = sys.argv[3]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model = Net(model_path,mean_train_path)

image_names = os.listdir(test_path)
total_l1_loss = 0
total_l2_loss = 0
total_overlap = 0
total_overlap_score = 0
total_time = 0
num_of_images = 0

for fn in image_names:
    print(fn)
    image = cv2.imread(os.path.join(test_path, fn), 3)

    points, time = model.predict_points(image)

    image = cv2.drawContours(image, [points.astype(np.int32)], -1, (255, 0, 255), 10)

    # image = cv2.resize(image, dsize=None, fx=0.2, fy=0.2)
    # cv2.imshow('result', image)
    # k = cv2.waitKey(0)
    # if 0xFF & k == ord('q') or 0xFF & k == ord('Q'):
    #     break
    output_path = os.path.join(output_dir, fn)
    cv2.imwrite(output_path, image)
