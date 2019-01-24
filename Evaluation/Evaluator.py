import os
import sys
import time

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

curren_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(curren_dir,".."))
sys.path.append(root_dir)

import rect

class Evaluator(object):
    def __init__(self, labels_file_path):
        
        self.error_func = {'L1': self.compute_L1_loss,
                           'L2': self.compute_L2_loss,
                           'houghL2': None,
                           'overlap': self.compute_overlap}

        self.data_matrix = pd.read_csv(labels_file_path,
                                       header=None, sep=' ')
        self.names = self.data_matrix.ix[:, 0].tolist()

    def get_label_by_image_name(self, image_name):
        idx = self.names.index(image_name)
        label = np.array(self.data_matrix.ix[idx, 1:]).reshape((4, 2)).astype('int32')
        return label

    def compute_L1_loss(self, predict_points, label_points, threshold = 0.):
        # label = rect.rectify(label_points)
        label = label_points
        predicted_label = rect.rectify(predict_points)

        diff = label - predicted_label

        square_diff = np.abs(diff)
        dists = square_diff.sum(axis=1, dtype=np.int32)
        avg_loss = np.sum(dists)/4

        score = 1 if avg_loss > threshold else 0
        return avg_loss, score


    def compute_L2_loss(self, predict_points, label_points, threshold = 0.):
        # label = rect.rectify(label_points)
        label = label_points
        predicted_label = rect.rectify(predict_points)

        diff = label - predicted_label
       
        square_diff = diff**2
        dists = square_diff.sum(axis=1, dtype=np.int32)
        dists = np.sqrt(dists)
        avg_loss = np.sum(dists)/4

        score = 1 if avg_loss > threshold else 0
        return avg_loss, score

    def compute_overlap(self, predict_points, label_points, threshold = 0.):
        # label_points = rect.rectify(label_points)
        # predict_points = rect.rectify(predict_points)

        predicted_polygon = Polygon(predict_points)
        labeled_polygon = Polygon(label_points)
        # print(predicted_polygon)
        # print(labeled_polygon)
        intersection = predicted_polygon.intersection(labeled_polygon)
        union = predicted_polygon.union(labeled_polygon)
        
        overlap = intersection.area / union.area

        score = 1 if overlap > threshold else 0
        return overlap, score
    

if __name__ == '__main__':
    fake_labeled_points = np.array([[0, 0], [0, 2], [2, 0], [2, 2]])
    fake_predicted_points = np.array([[1, 1], [1, 3], [3, 1], [3, 3]])

    evaluator = Evaluator(sys.argv[1])

    print( 'L1_loss = {}'.format(evaluator.error_func['L1'](fake_labeled_points,
                                                            fake_predicted_points)) )

    print( 'L2_loss = {}'.format(evaluator.error_func['L2'](fake_labeled_points,
                                                            fake_predicted_points)) )

    print( 'overlap = {}'.format(evaluator.error_func['overlap'](fake_labeled_points,
                                                                 fake_predicted_points)) )


