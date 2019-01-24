import sys

import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

sys.path.append('..')
sys.path.append('../utils')

class Net(object):
    
    def __init__(self, model_path):
        
        
        graph_def = graph_pb2.GraphDef()
        with open(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            _ = importer.import_graph_def(graph_def, name="")

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="prefix",
                op_dict=None,
                producer_op_list=None
            )
            
        self.sess = tf.Session(graph=graph)
        self.output_node = self.sess.graph.get_tensor_by_name("prefix/dense_1/BiasAdd:0")
        self.input_tensor = self.sess.graph.get_tensor_by_name("prefix/input_1:0")

        
    def predict_points(self, image):

        points = [];
        start_time = time.clock()

        small_image = cv2.resize(image, (227, 227))

        network_input = small_image[np.newaxis, :, :, :]

        
        network_output = self.sess.run(self.output_node,
                                       feed_dict={self.input_tensor: network_input})[0]

        print(network_output)
        
        classes = network_output[[8,9]]

        c = np.argmax(classes)

        if 0 == c: # positive case
        
            x = network_output[[0,2,4,6]]
            y = network_output[[1,3,5,7]]

        elif 1 == c: # negative case

            x = np.array([0, 0, 1, 1])
            y = np.array([0, 1, 1, 0])
            
        else:
            pass
            
        x = x * image.shape[1]
        y = y * image.shape[0]

        corners = np.vstack((x,y)).transpose(1,0)

        for corner in corners:
            points.append(corner)

        end_time = time.clock()

        runtime = end_time - start_time

        return np.array(points), runtime



