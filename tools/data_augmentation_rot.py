import os
import sys

import re
import cv2
import numpy as np

import rect

train_path = sys.argv[1]
augment_train_path = sys.argv[2]

prefix = ['fu', 'fl', 'fd','fr',  'mu', 'ml', 'md', 'mr']

for i, fn in enumerate(os.listdir(train_path)):
    print fn
    
    path_fn = os.path.join(train_path, fn)
    image = cv2.imread(path_fn)

    height, width, _ = image.shape
    
    name_hash_angle = re.findall(r'(.+)-coordinate_1', fn)[0]


    info_lst = os.path.splitext(fn)[0]
    kw = re.compile(r'coordinate_1__([0-9E/.-]+)_([0-9E/.-]+)-coordinate_2__([0-9E/.-]+)_([0-9E/.-]+)-coordinate_3__([0-9E/.-]+)_([0-9E/.-]+)-coordinate_4__([0-9E/.-]+)_([0-9E/.-]+)')
    points = kw.findall(info_lst)
    points = [int(float(p)) for p in points[0]]
    points = np.array(points).astype('float32').reshape(-1, 2)
    points = rect.rectify(points)
    points = np.hstack((points, np.ones(4).reshape(4, 1))).transpose(1, 0)

    # front
    angle = 0
    for j, op in enumerate(prefix[:4]):
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        if 0 == (j % 2):
            dst = cv2.warpAffine(image, M, (width, height), cv2.INTER_CUBIC)
        else:
            M[0,2] +=(height-width)/2 
            M[1,2] +=(width-height)/2 
            dst = cv2.warpAffine(image, M, (height, width), cv2.INTER_CUBIC)
            
        angle += 90
        new_points = np.dot(M, points).transpose(1, 0)
        new_points = rect.rectify(new_points).astype('int32')
        info_pos = ''
        for k in range(4):
            info_pos += '-coordinate_{}__{}_{}'.format(k+1, new_points[k][0], new_points[k][1])

        fn_save = '{}_{}{}.jpg'.format(op, name_hash_angle, info_pos)
        cv2.imwrite(os.path.join(augment_train_path, fn_save), dst)
        # # print fn_save
        
    # mirror
    angle = 0
    mirror_image = cv2.flip(image, 1)
    reflect_matrix = np.array([[-1, 0, width],[0, 1, 0]])
    # print reflect_matrix
    # print points
    points = np.dot(reflect_matrix, points).transpose(1, 0)
    points = rect.rectify(points)
    points = np.hstack((points, np.ones(4).reshape(4, 1))).astype('int32').transpose(1, 0)
    # print points

    for j, op in enumerate(prefix[4:]):
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        if 0 == (j % 2):
            dst = cv2.warpAffine(mirror_image, M, (width, height), cv2.INTER_CUBIC)
        else:
            M[0,2] +=(height-width)/2 
            M[1,2] +=(width-height)/2 
            dst = cv2.warpAffine(mirror_image, M, (height, width), cv2.INTER_CUBIC)
            
        angle += 90
        new_points = np.dot(M, points).transpose(1, 0)
        new_points = rect.rectify(new_points).astype('int32')
        info_pos = ''
        for k in range(4):
            info_pos += '-coordinate_{}__{}_{}'.format(k+1, new_points[k][0], new_points[k][1])

        fn_save = '{}_{}{}.jpg'.format(op, name_hash_angle, info_pos)
        cv2.imwrite(os.path.join(augment_train_path, fn_save), dst)
               
