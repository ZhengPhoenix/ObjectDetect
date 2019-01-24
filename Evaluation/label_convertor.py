import os
from sys import argv
import numpy as np
import cv2
import csv


dir_path = argv[1]
label_txt_path = argv[2]
file_labels = open(label_txt_path, 'w')

for image in os.listdir(dir_path):

    if image.endswith("jpg") or image.endswith("JPG"):
        if os.path.isfile(os.path.join(dir_path, image+".csv")):
            with open(os.path.join(dir_path,image+ ".csv"), 'r') as csvfile:
                spamwriter = csv.reader(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

                image_name = image
                coordinates = []
                for row in spamwriter:
                    if len(row) == 0:
                        continue                            
                    coordinates.append(row[0])
                    coordinates.append(row[1])

                coordinates = ' '.join(coordinates)
                line_label = '{} {}\n'.format(image_name, coordinates)
                file_labels.write(line_label)


