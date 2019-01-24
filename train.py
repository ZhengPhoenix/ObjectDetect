# coding: utf-8

import re
import os
import sys

import cv2
import numpy as np
import pandas as pd
import keras
import argparse
from keras.models import *
from keras.layers import *
import tensorflow as tf
from keras.optimizers import Adam
from keras import backend as K
from keras_squeezenet import SqueezeNet
from sklearn.cross_validation import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.backend.tensorflow_backend import set_session
import re


shape = 227
label_num = 8 + 2
batch_size = 256


def fetch_epoch_num(s):
    r = re.match('.*\.(\d+)-.*', s)
    if r == None:
        return 0
    else:
        return int(r.groups()[0])


# configure argument
parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Training Path')
parser.add_argument('-o', help='Training Output Path')

args = parser.parse_args()
training_path = args.o
image_dir = args.i

if not os.path.exists(training_path):
    os.makedirs(training_path)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

labels_path = os.path.join(image_dir, 'gt.csv')
data_pairs = pd.read_csv(labels_path, header=None, sep=' ')
filenames = [os.path.join(image_dir,name) for name in data_pairs.loc[:, 0].tolist()]
labels = data_pairs.loc[:, 1:].values.astype(float) / 300.

number_samples = len(filenames)

save_model_path = os.path.join(training_path,'squeezenet_model')
if (not os.path.isdir(save_model_path)):
    os.mkdir(save_model_path)
    
training_img_path = os.path.join(training_path, 'training_images')
if (not os.path.isdir(training_img_path)):
    os.mkdir(training_img_path)
    
training_log_dir = os.path.join(training_path, 'squeezenet_train')

train_filenames, val_filenames, train_labels, val_labels = train_test_split(np.array(filenames), labels, test_size=0.01)
print('train_size:', train_filenames.shape[0])
print('val_size:', val_filenames.shape[0])

sum_image = np.zeros([shape, shape, 3]).astype('float32')
for name in train_filenames:
    image = cv2.imread(name)
    image = cv2.resize(image, (shape, shape))
    sum_image = sum_image + image.astype('float32')

mean_train = np.mean(sum_image, axis=(0, 1)) / train_filenames.shape[0]
np.save(os.path.join(training_path, 'document_mean_train.npy'), mean_train)


print(mean_train)
mean_train = np.expand_dims(mean_train, axis=0)
mean_train = np.expand_dims(mean_train, axis=0)
mean_train = np.expand_dims(mean_train, axis=0)


def custom_objective(y_true, y_pred):
    '''Just another crossentropy'''
    classes_true = y_true[: , -2:]
    classes_pred = y_pred[: , -2:]
    coordinates_true = y_true[ : , :-2]
    coordinates_pred = y_pred[ : , :-2]

    vec_classes_true = tf.reshape(classes_true[:,0], [1,-1])
    abs_diff = tf.abs(tf.subtract(coordinates_true, coordinates_pred))
    loss_cross_entropy = tf.losses.softmax_cross_entropy(classes_true,classes_pred)

    return tf.reduce_mean(tf.matmul(vec_classes_true,abs_diff)) + loss_cross_entropy
   

def step_decay(epoch):
    """
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    """
    lrate = 0.001
    if epoch > 25:
        lrate = 0.0001
    elif epoch > 50:
        lrate = 0.0001
    elif epoch > 75:
        lrate = 0.00001
    elif epoch > 100:
        lrate = 0.00001
    return lrate


input_tensor = Input((shape, shape, 3))
mean_train_tensor = Input(tensor=tf.constant(mean_train,dtype=tf.float32))
subtracted = Subtract()([input_tensor, mean_train_tensor])
squeezeNet = SqueezeNet(subtracted)
output = Dense(label_num, activation=None)(squeezeNet)

model = Model(inputs=[input_tensor, mean_train_tensor], outputs=output)
epoch = 0


def _parse_function(filename, label):
    image_string = tf.read_file(filename)

    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    image = tf.image.resize_images(image_decoded, [shape, shape])

    return image[:,:,::-1], label


def data_generator(_filenames, _labels):

    filenames = tf.constant(_filenames, name='filenames_list')
    labels = tf.constant(_labels, name='labels_list')

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(number_samples)


    dataset = dataset.map(_parse_function,
                          num_parallel_calls=batch_size)

    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(4)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    train_images, train_labels = iterator.get_next()
    sess = tf.keras.backend.get_session()

    while(True):
        train_image_data, train_label_data = sess.run([train_images, train_labels])
        #print(train_label_data.shape)
        yield train_image_data, train_label_data


class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        weights_name = 'weights.Epoch_{}-Loss_{}-ValLoss_{}.h5'.format(epoch, loss, val_loss)

        model.save_weights(os.path.join(save_model_path, weights_name))


opt = Adam(lr=0.001)
model.compile(loss=custom_objective, optimizer=opt, metrics=['accuracy'])
model.summary()

learningrate = keras.callbacks.LearningRateScheduler(step_decay)
early_stopping = EarlyStopping(patience=25, min_delta=10)
tensorboard = TensorBoard(log_dir=training_log_dir)
checkpoint = MyCallback()
callbacks_list = [checkpoint, learningrate, tensorboard]

val_img_list = []
for name in val_filenames:
  image = cv2.imread(name)
  image = cv2.resize(image,(shape,shape))
  val_img_list.append(image)

validation_images = np.array(val_img_list)
history = model.fit_generator(data_generator(train_filenames, train_labels),
                              callbacks=callbacks_list,
                              steps_per_epoch = int(train_filenames.shape[0] / batch_size) + 1, epochs=500,
                              validation_data = (validation_images, val_labels),
                              validation_steps = int(val_filenames.shape[0] / batch_size) + 1,
                              initial_epoch=epoch)

model.save(os.path.join(save_model_path, 'model.h5'))
