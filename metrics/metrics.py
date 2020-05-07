'''
InstantDL
Metrics for training
Written by Dominik Waibel
´'''

from __future__ import print_function
import numpy as np
from keras.losses import *
from keras.losses import categorical_crossentropy as cct
import tensorflow as tf
from sklearn.metrics import accuracy_score

def dice_coef(y_true, y_pred):
    """The dice coefficient needed to calculate the dice loss
    # Arguments
        y_true: tensor,  the groundtrouth or label, loaded from the groundtruth folder
        y_pred: tensor, the prediction or output of the network
        returns: float
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.cast(y_pred, 'float32')
    prediction_f = K.cast(K.greater(K.flatten(y_pred_f), 0.5), 'float32')
    intersection = y_true_f * prediction_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(prediction_f))
    return score

def dice_loss(y_true, y_pred):
    """The dice loss to calculate the similarity between two samples
    # Arguments
        y_true: tensor,  the groundtrouth or label, loaded from the groundtruth folder
        y_pred: tensor, the prediction or output of the network
        returns: float
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(K.square(y_true_f),-1) + K.sum(K.square(y_pred_f),-1) + smooth)
    return tf.keras.backend.mean(1. - score)

def dice_coef_loss_crossentropy(image, prediction):
    """The sum of the dice loss and categorical crossentropy to calculate the similarity between two samples
    # Arguments
        y_true: tensor, the groundtrouth or label, loaded from the groundtruth folder
        y_pred: tensor, the prediction or output of the network
        returns: float
    """
    total_loss = dice_loss(image, prediction) + cct(image, prediction)
    return total_loss


def total_variation_loss(prediction):
    """The total variation loss
    # Arguments
        prediction: tensor, the prediction or output of the network
        returns: float
    https://arxiv.org/abs/1803.11293
    https://arxiv.org/abs/1605.01368
    """
    if K.ndim(prediction) == 5:
        img_nrows = prediction.shape[1]
        img_ncols = prediction.shape[2]
        if K.image_data_format() == 'channels_first':
            logging.info("Total Variation Loss Channel first")
            a = K.square(prediction[:, :, :img_nrows - 1, :img_ncols - 1, :] - prediction[:, :, 1:, :img_ncols - 1, :])
            b = K.square(prediction[:, :, :img_nrows - 1, :img_ncols - 1, :] - prediction[:, :, :img_nrows - 1, 1:, :])
        else:
            logging.info("Total Variation Loss Channel last")
            a = K.square(prediction[:, :img_nrows - 1, :img_ncols - 1, :, :] - prediction[:, 1:, :img_ncols - 1, :, :])
            b = K.square(prediction[:, :img_nrows - 1, :img_ncols - 1, :, :] - prediction[:, :img_nrows - 1, 1:, :, :])
        return K.sum(K.pow(a + b, 1.25))
    if K.ndim(prediction) == 4:
        img_nrows = prediction.shape[1]
        img_ncols = prediction.shape[2]
        if K.image_data_format() == 'channels_first':
            logging.info("Total Variation Loss Channel first")
            a = K.square(prediction[:, :, :img_nrows - 1, :img_ncols - 1] - prediction[:, :, 1:, :img_ncols - 1, :])
            b = K.square(prediction[:, :, :img_nrows - 1, :img_ncols - 1] - prediction[:, :, :img_nrows - 1, 1:])
        else:
            logging.info("Total Variation Loss Channel last")
            a = K.square(prediction[:, :img_nrows - 1, :img_ncols - 1, :] - prediction[:, 1:, :img_ncols - 1, :])
            b = K.square(prediction[:, :img_nrows - 1, :img_ncols - 1, :] - prediction[:, :img_nrows - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

def mse(y_true, y_pred):
    """The mean squarred error loss to similarity between two samples
    # Arguments
        y_true: tensor, the groundtrouth or label, loaded from the groundtruth folder
        y_pred: tensor, the prediction or output of the network
        returns: float
    """
    logging.info("y_pred", y_pred, "y_true", y_true)
    return K.mean(K.square(y_pred - y_true))

def total_variaton_loss_mse(image, prediction):
    '''
    The sum of mean squarred error and total variation loss to similarity between two samples
    # Arguments
        y_true: tensor, he groundtrouth or label, loaded from the groundtruth folder
        y_pred: tensor, the prediction or output of the network
        returns: float
    Combine the total variation loss with the mean squarred error loss. The tot_var should be 2% of the mse loss
    according to https://arxiv.org/abs/1803.11293 '''
    tot_var = total_variation_loss(prediction)
    mse = mean_squared_error(image, prediction)
    total_loss = mse + (0.01 * mse * tot_var)
    return total_loss

def dice_mse(y_true, y_pred):
    '''
     The sum of mean squarred error and dice loss to similarity between two samples
     # Arguments
         y_true: tensor, the groundtrouth or label, loaded from the groundtruth folder
         y_pred: tensor, the prediction or output of the network
         returns: float
    '''
    return mse(y_true, y_pred) +  0.5 * dice_loss(y_true, y_pred)

def binary_crossentropy_mse(y_true, y_pred):
    '''
     The sum of mean squarred error and binary crossetnropy loss to similarity between two samples
     # Arguments
         y_true: tensor, the groundtrouth or label, loaded from the groundtruth folder
         y_pred: tensor, the prediction or output of the network
         returns: float
    '''
    return mse(y_true, y_pred) + binary_crossentropy(y_true, y_pred)

def binary_crossentropy_dice(y_true, y_pred):
    '''
     The sum of binary crossentropy and dice loss to similarity between two samples
     # Arguments
         y_true: tensor, the groundtrouth or label, loaded from the groundtruth folder
         y_pred: tensor, the prediction or output of the network
         returns: float
    '''
    return dice_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred)