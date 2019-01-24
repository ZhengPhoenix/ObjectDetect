import os
import sys
import time

import cv2
import numpy as np

from net import Net

from Evaluator import Evaluator

current_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(current_dir,".."))

test_path = sys.argv[1]
labels_file_path = sys.argv[2]
model_path = sys.argv[3]
mean_train_path = os.path.join(os.sep.join(model_path.split(os.sep)[:-2]),
                               "document_mean_train.npy")
bad_sample_saving_path = sys.argv[6] if len(sys.argv) > 6 else None
test_postproc = int(sys.argv[5]) > 0 if len(sys.argv) > 6 else 0

if not bad_sample_saving_path is None:
    if (not os.path.isdir(bad_sample_saving_path)) :
        os.mkdir(bad_sample_saving_path)

if test_postproc:
    from net_postproc import Net

model = Net(model_path,mean_train_path)

evaluator = Evaluator(labels_file_path)

image_names = os.listdir(test_path)
total_l1_loss = 0
total_l2_loss = 0
total_overlap = 0
total_overlap_score = 0
total_time = 0
num_of_images = 0

for fn in image_names:

    if not (fn.endswith("jpg") or fn.endswith("JPG")):
        continue

    print(fn)
    image = cv2.imread(os.path.join(test_path, fn), 3)

    points, time = model.predict_points(image)
    label = evaluator.get_label_by_image_name(fn)

    # l1_loss, _ = evaluator.error_func['L1'](label, points)
    # print( 'L1_loss = {}'.format(l1_loss))

    # l2_loss, _ = evaluator.error_func['L2'](label, points)
    # print( 'L2_loss = {}'.format(l2_loss))
    try:
        overlap, overlap_score = evaluator.error_func['overlap'](points, label, threshold=0.8)
    except:
        overlap = 0
        overlap_score =0 
    print( 'overlap = {}'.format(overlap))

    # total_l1_loss += l1_loss
    # total_l2_loss += l2_loss
    total_overlap += overlap
    total_overlap_score += overlap_score
    total_time += time
    print( 'runtime = {}ms'.format(time*1000))

    num_of_images += 1

    if (not bad_sample_saving_path is None) and (0 == overlap_score):
        # command = "cp {} {}".format(os.path.join(test_path, fn), bad_sample_saving_path)
        # os.system(command)
        image = cv2.drawContours(image, [label], -1, (0, 255, 255), 3)
        image = cv2.drawContours(image, [points.astype('int32')], -1, (255, 0, 255), 3)
        cv2.imwrite(os.path.join(bad_sample_saving_path, fn),image)

    # image = cv2.drawContours(image, [points.astype('int32')], -1, (255, 0, 255), 10)
    # image = cv2.drawContours(image, [label], -1, (0, 255, 255), 10)
    # image = cv2.resize(image, dsize=None, fx=0.2, fy=0.2)
    # cv2.imshow('result', image)
    # k = cv2.waitKey(0)
    # if 0xFF & k == ord('q') or 0xFF & k == ord('Q'):
    #     break

# print('average L1 loss = {}'.format( float(total_l1_loss) / len(image_names) ))
# print('average L2 loss = {}'.format( float(total_l2_loss) / len(image_names) ))
print('average overlap = {}'.format( float(total_overlap) / num_of_images ))
print('hitRate for Overlap = {}'.format( float(total_overlap_score) / num_of_images ))
print('average runtime = {}'.format( float(total_time*1000) / num_of_images ))
