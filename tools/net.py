import sys

import cv2
import time
import numpy as np
import tensorflow as tf

from keras.models import *
from keras.layers import *

from keras import backend as K
from keras_squeezenet import SqueezeNet


sys.path.append('..')
sys.path.append('../utils')

def network(mean_train):
    shape = 227
    label_num = 10
    mean_train = np.expand_dims(mean_train, axis=0)
    mean_train = np.expand_dims(mean_train, axis=0)
    mean_train = np.expand_dims(mean_train, axis=0)
    input_tensor = Input((shape, shape, 3))
    mean_train_tensor = Input(tensor=tf.constant(mean_train,dtype=tf.float32))
    subtracted = Subtract()([input_tensor, mean_train_tensor])
    squeezeNet = SqueezeNet(subtracted)
    output = Dense(label_num, activation=None)(squeezeNet)

    return Model(inputs=[input_tensor, mean_train_tensor], outputs=output)

def custom_objective(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.subtract(y_pred, y_true)))


class Net(object):

    
    def __init__(self, model_path, mean_train_path):
        mean_train = np.load(mean_train_path)
        self.model = network(mean_train)
        self.model.load_weights(model_path)

        
    def predict_points(self, image):

        points = [];
        start_time = time.clock()

        small_image = cv2.resize(image, (227, 227))

        network_input = small_image[np.newaxis, :, :, :]
        network_output = self.model.predict(network_input)[0]

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



