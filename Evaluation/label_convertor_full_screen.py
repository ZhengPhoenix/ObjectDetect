import os
from sys import argv
import numpy as np
import cv2
import csv


neg_dir = argv[1]
label_txt_path = argv[2]
file_labels = open(label_txt_path, 'w')

for image in os.listdir(neg_dir):

    if image.endswith("jpg") or image.endswith("JPG"):
        
        image_name = image
        img_path = os.path.join(neg_dir, image_name)
        img = cv2.imread(img_path)

        gt = ['0', '0',
              str(img.shape[1]), '0',
              str(img.shape[1]), str(img.shape[0]),
              '0', str(img.shape[0])]

        file_labels.write('{} {}\n'.format(image_name, ' '.join(gt)))


