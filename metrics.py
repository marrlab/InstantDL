from __future__ import print_function
import numpy as np
from keras.losses import *
from keras.losses import categorical_crossentropy as cct
import tensorflow as tf
from sklearn.metrics import accuracy_score
#import keras as K

def dice_coef(image, prediction):
    image_f = K.flatten(image)
    y_pred = K.cast(prediction, 'float32')
    prediction_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = image_f * prediction_f
    score = 2. * K.sum(intersection) / (K.sum(image_f) + K.sum(prediction_f))
    return score

'''def dice_loss(y_true, y_pred):
    num_labels = K.backend.int_shape(y_pred)[-1]
    probabilities = K.backend.reshape(y_pred, [-1, num_labels])
    y_true_flat = K.backend.reshape(y_true, [-1] )
    onehots_true = tf.one_hot(tf.cast(y_true_flat, tf.int32 ), num_labels)
    numerator = tf.reduce_sum(onehots_true * probabilities, axis=0)
    denominator = tf.reduce_sum(onehots_true + probabilities, axis=0)
    loss = 1.0 - 2.0 * (numerator + 1) / (denominator + 1)
    return tf.keras.backend.mean(loss)'''

def dice_loss(image, prediction):
    smooth = 1.
    image_f = K.flatten(image)
    prediction_f = K.flatten(prediction)
    intersection = image_f * prediction_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(K.square(image_f),-1) + K.sum(K.square(prediction_f),-1) + smooth)
    return tf.keras.backend.mean(1. - score)

def dice_coef_loss_crossentropy(image, prediction):
    total_loss = dice_loss(image, prediction) + cct(image, prediction)
    return total_loss

def softmax(X):
    X = np.array(X)
    exps = np.exp(X-np.max(X))
    exps = exps /np.sum(exps)
    return exps

def DW_crossentropy(image, prediction):
    m = prediction.shape[0]
    p = softmax(image)
    log_likelihood = -np.log(p[range(m)],prediction)
    loss = np.sum(log_likelihood) / m
    return loss


def total_variation_loss(prediction):
    if K.ndim(prediction) == 5:
        img_nrows = prediction.shape[1]
        img_ncols = prediction.shape[2]
        if K.image_data_format() == 'channels_first':
            print("Total Variation Loss Channel first")
            a = K.square(prediction[:, :, :img_nrows - 1, :img_ncols - 1, :] - prediction[:, :, 1:, :img_ncols - 1, :])
            b = K.square(prediction[:, :, :img_nrows - 1, :img_ncols - 1, :] - prediction[:, :, :img_nrows - 1, 1:, :])
        else:
            print("Total Variation Loss Channel last")
            a = K.square(prediction[:, :img_nrows - 1, :img_ncols - 1, :, :] - prediction[:, 1:, :img_ncols - 1, :, :])
            b = K.square(prediction[:, :img_nrows - 1, :img_ncols - 1, :, :] - prediction[:, :img_nrows - 1, 1:, :, :])
        return K.sum(K.pow(a + b, 1.25))
    if K.ndim(prediction) == 4:
        img_nrows = prediction.shape[1]
        img_ncols = prediction.shape[2]
        if K.image_data_format() == 'channels_first':
            print("Total Variation Loss Channel first")
            a = K.square(prediction[:, :, :img_nrows - 1, :img_ncols - 1] - prediction[:, :, 1:, :img_ncols - 1, :])
            b = K.square(prediction[:, :, :img_nrows - 1, :img_ncols - 1] - prediction[:, :, :img_nrows - 1, 1:])
        else:
            print("Total Variation Loss Channel last")
            a = K.square(prediction[:, :img_nrows - 1, :img_ncols - 1, :] - prediction[:, 1:, :img_ncols - 1, :])
            b = K.square(prediction[:, :img_nrows - 1, :img_ncols - 1, :] - prediction[:, :img_nrows - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

''' Combine the total variation loss with the mean squarred error loss. The tot_var should be 2% of the mse loss
according to https://arxiv.org/abs/1803.11293 '''

def mse(y_true, y_pred):
    print("y_pred", y_pred, "y_true", y_true)
    return K.mean(K.square(y_pred - y_true))

def total_variaton_loss_mse(image, prediction):
    tot_var = total_variation_loss(prediction)
    mse = mean_squared_error(image, prediction)
    total_loss = mse + (0.01 * mse * tot_var)
    return total_loss

def Huber_loss(y_pred, y_true):
    loss = tf.losses.huber_loss(y_pred, y_true)
    return loss

def dice_mse(y_true, y_pred):
    return mse(y_true, y_pred) +  0.5 * dice_loss(y_true, y_pred)

def binary_crossentropy_mse(y_true, y_pred):
    return mse(y_true, y_pred) + binary_crossentropy(y_true, y_pred)

def binary_crossentropy_dice(y_true, y_pred):
    return dice_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred)