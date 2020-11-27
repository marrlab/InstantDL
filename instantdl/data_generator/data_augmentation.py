'''
InstantDL
Data augmentations which can be executed on the fly
Written by Dominik Waibel
'''

import numpy as np
import skimage as sk
import random
from instantdl.data_generator.data import plot2images
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter
import math
import os

def data_augentation(X, Y, data_gen_args, data_path_file_name):

    if "horizontal_flip" in data_gen_args:
        """ Flip image and groundtruth horizontally by a chance of 33% 
        # Arguments
            X: tensor, the input image
            Y: tensor, the groundtruth
            returns: two tensors, X, Y, which have the same dimensions as the input 
        """
        if data_gen_args["horizontal_flip"] == True & random.choice([True, False, False]) == True:
            X = np.flip(X, len(np.shape(X))-1)
            Y = np.flip(Y, len(np.shape(Y))-1)

    if "vertical_flip" in data_gen_args:
        """ Flip image and groundtruth vertically by a 33% chance
        # Arguments
            X: tensor, the input image
            Y: tensor, the groundtruth
            returns: two tensors, X, Y, which have the same dimensions as the input
        """
        if data_gen_args["vertical_flip"] == True & random.choice([True, False, False]) == True:
            X = np.flip(X, len(np.shape(X))-2)
            Y = np.flip(Y, len(np.shape(Y))-2)

    if "rotation_range" in data_gen_args and data_gen_args["rotation_range"] > 0:
        """ Rotate image and groundtruth by a random angle within the rotation range 
        # Arguments
            X: tensor, the input image
            Y: tensor, the groundtruth
            returns: two tensors, X, Y, which have the same dimensions as the input
        """
        angle =  np.random.choice(int(data_gen_args["rotation_range"]*100))/100
        if len(np.shape(X))-1 == 3:
            X = np.nan_to_num(rotate(X, angle, mode='nearest', axes=(2, 3), reshape=False))
            Y = np.nan_to_num(rotate(Y, angle, mode='nearest', axes=(2, 3), reshape=False))
        else:
            X = np.nan_to_num(rotate(X, angle, mode='nearest', axes=(1, 2), reshape=False))
            Y = np.nan_to_num(rotate(Y, angle, mode='nearest', axes=(1, 2), reshape=False))
        # Zoom so that there are no empty edges
        shape_X = np.shape(X)
        shape_Y = np.shape(Y)
        length_X = np.max(shape_X)
        zoom_length_X = length_X / (np.cos(math.radians(angle)) + np.sin(math.radians(angle)))
        zoom_length_Y = length_X / (np.cos(math.radians(angle)) + np.sin(math.radians(angle)))
        X = X[..., int((length_X - zoom_length_X) / 2) : int(length_X -  ((length_X - zoom_length_X) / 2)), int((length_X - zoom_length_X) / 2) : int(length_X - ((length_X - zoom_length_X) / 2)), :]
        X = np.nan_to_num(sk.transform.resize(X, shape_X))
        Y = Y[..., int((length_X - zoom_length_Y) / 2) : int(length_X -  ((length_X - zoom_length_Y) / 2)), int((length_X - zoom_length_Y) / 2) : int(length_X - ((length_X - zoom_length_Y) / 2)), :]
        Y = np.nan_to_num(sk.transform.resize(Y, shape_Y))


    if "width_shift_range" in data_gen_args and data_gen_args["width_shift_range"] > 0 & random.choice([True, False]) == True:
        """ Shift the image and groundtruth width by a number within the width shift range by a 50% chance
        # Arguments
            X: tensor, the input image
            Y: tensor, the groundtruth
            width shift range: float, amound of width shift
            returns: two tensors, X, Y, which have the same dimensions as the input
        """
        width_shift = np.random.choice(int(data_gen_args["width_shift_range"] * 100))
        shape_X = np.shape(X)
        shape_Y = np.shape(Y)
        if len(np.shape(X))-1 == 3:
            size = np.size(X[0, 0, :, 0, 0])
        else:
            size = np.size(X[0, 0, :, 0,])
        start_width_shift = np.random.choice(int(size - (size - width_shift)+1))

        X = X[:, :, start_width_shift: int(size - int(size * width_shift / 100)),...]
        X = np.nan_to_num(sk.transform.resize(X, shape_X))
        Y = Y[:, :, start_width_shift: int(size - int(size * width_shift / 100)),...]
        Y = np.nan_to_num(sk.transform.resize(Y, shape_Y))

    if "height_shift_range" in data_gen_args and data_gen_args["height_shift_range"] > 0 & random.choice([True, False]) == True:
        """ Shift the image and groundtruth height by a number within the width shift range by a 50% chance
        # Arguments
            X: tensor, the input image
            Y: tensor, the groundtruth
            height shift range: float, amound of height shift
            returns: two tensors, X, Y, which have the same dimensions as the input
        """
        height_shift = np.random.choice(int(data_gen_args["height_shift_range"] * 100))
        shape_X = np.shape(X)
        shape_Y = np.shape(Y)
        if len(np.shape(X))-1 == 3:
            size = np.size(X[0, 0, :, 0, 0])
        else:
            size = np.size(X[0, 0, :, 0,])
        start_heigth_shift = np.random.choice(int(size - (size - height_shift)+1))
        X = X[..., start_heigth_shift: int(size - int(size * height_shift / 100)), :]
        X = np.nan_to_num(sk.transform.resize(X, shape_X))
        Y = Y[..., start_heigth_shift: int(size - int(size * height_shift / 100)), :]
        Y = np.nan_to_num(sk.transform.resize(Y, shape_Y))

    if "zoom_range" in data_gen_args and data_gen_args["zoom_range"] > 0 & random.choice([True, False]) == True:
        """ Zooms the image and groundtruth to a random magnification in the zoom range and to a random position in the image by a 50% chance
        # Arguments
            X: tensor, the input image
            Y: tensor, the groundtruth
            zoom range: float, amound of zoom
            returns: two tensors, X, Y, which have the same dimensions as the input
        """
        zoom = np.random.choice(int(data_gen_args["zoom_range"] * 100)) + 1
        shape_X = np.shape(X)
        shape_Y = np.shape(Y)
        if len(np.shape(X))-1 == 3:
            size = np.size(X[0, 0, :, 0, 0])
        else:
            size = np.size(X[0, 0, :, 0,])
        x_position = np.random.choice(zoom)
        y_position = np.random.choice(zoom)
        X = X[..., x_position: int(x_position + size - zoom), y_position: int(y_position + size - zoom), :]
        X = np.nan_to_num(sk.transform.resize(X, shape_X))
        Y = Y[..., x_position:int(x_position + size - zoom), y_position: int(y_position + size - zoom), :]
        Y = np.nan_to_num(sk.transform.resize(Y, shape_Y))

    if "gaussian_noise" in data_gen_args and data_gen_args["gaussian_noise"] > 0 and random.choice([True, False, False]) == True:
        """ Adds gaussian noise to the image by a one-third chance
        # Arguments
            X: tensor, the input image
            gaussian_noise: float, amound of gaussian noise added
            returns: one tensors, X, which have the same dimensions as the input
        """
        value = data_gen_args["gaussian_noise"]
        X = X + np.random.normal(0, value)

    if "gaussian_blur_image" in data_gen_args and data_gen_args["gaussian_blur_image"] > 0 and random.choice([True, False, False]) == True:
        """ Blurs the input image in gaussian fashin by a 33% chance
        # Arguments
            X: tensor, the input image
            gaussian_blur: float, amound of gaussian blur added
            returns: one tensor, X, which have the same dimensions as the input
        """
        value = data_gen_args["gaussian_blur_image"]
        X = gaussian_filter(X, sigma=value)

    if "gaussian_blur_label" in data_gen_args and data_gen_args["gaussian_blur_label"] > 0 and random.choice([True, False, False]) == True:
        """ Blurs the groundtruth image in gaussian fashin by a 33% chance
        # Arguments
            Y: tensor, the groundtruth image
            gaussian_blur: float, amound of gaussian blur added
            returns: one tensor, Y, which have the same dimensions as the input
        """
        value = data_gen_args["gaussian_blur_label"]
        Y = gaussian_filter(Y, sigma=value)

    if "contrast_range" in data_gen_args and random.choice([True, False, False]) == True:
        """ Increases or decreases the contrast of the input in by the range given by a 33% chance
        # Arguments
            X: tensor, the groundtruth image
            contrast_range: float, amound of contrast added or removed
            returns: one tensor, X, which have the same dimensions as the input
        """
        range = np.random.uniform(-1,1) * data_gen_args["contrast_range"] + 1
        min_X = np.min(X)
        X = (X - min_X) * range + min_X

    if "brightness_range" in data_gen_args and random.choice([True, False]) == True:
        """ Increases or decreases the brightness of the input in by the range given by a 50% chance
        # Arguments
            X: tensor, the groundtruth image
            brightness_range: float, amound of gaussian noise added
            returns: one tensor, X, which have the same dimensions as the input
        """
        range = np.random.uniform(-1,1) * data_gen_args["brightness_range"]
        X = X + range * X

    if "threshold_background_image" in data_gen_args and random.choice([True, False, False]) == True:
        """ Thresholds the input image at the mean and sets every pixelvalue below the mean to zero by a 33% chance
        # Arguments
            X: tensor, the groundtruth image
            threshold_background_image: True or False
            returns: one tensor, X, which have the same dimensions as the input
        """
        mean_X = np.mean(X)
        X[X < mean_X] = 0

    if "threshold_background_groundtruth" in data_gen_args and random.choice([True, False, False]) == True :
        """ Thresholds the groundtruth image at the mean and sets every pixelvalue below the mean to zero by a 33% chance
        # Arguments
            Y: tensor, the groundtruth image
            threshold_background_groundtruth: True or False
            returns: one tensor, Y, which have the same dimensions as the input
        """
        mean_Y = np.mean(Y) * 0.8
        Y[Y < mean_Y] = 0

    if "binarize_mask" in data_gen_args and data_gen_args["binarize_mask"] == True:
        """ Binarize the groundtruth image at the mean and sets every pixelvalue below the mean to zero and every above to one
        # Arguments
            Y: tensor, the groundtruth image
            binarize_mask: True or False
            returns: one tensor, Y, which have the same dimensions as the input
        """
        mean_Y = np.mean(Y)
        Y[Y <= mean_Y] = 0
        Y[Y > mean_Y] = 1
        Y = np.nan_to_num(Y)

    if "save_augmented_images" in data_gen_args and data_gen_args["save_augmented_images"] == True:
        """ save all augmented images to a folder named "Augmentations" in the project folder to make augmentations trackable
        # Arguments
            X: tensor, the input image
            Y: tensor, the groundtruth image
            save_augmented_images: True or False
            returns: Saves the image pair as .png to the Augmentations folder
        """
        data_path, file_name = os.path.split(data_path_file_name)
        Aug_path = (data_path + '/Augmentations/' )
        os.makedirs(Aug_path, exist_ok=True)
        #TODO: Check if the folder and saving really works with the path
        title = file_name.split("'")[1]
        title = os.path.splitext(title)[0]
        #datacomb = np.concatenate((X, Y), axis=2).astype("uint8")
        #sk.io.imsave(Aug_path + title, datacomb[0,...])
        if len(np.shape(X)) == 5:
            plot2images(X[0, 10, ... ,0], Y[0, 10, :, :,0], Aug_path, title)
        elif len(np.shape(X)) == 4 and np.shape(X)[-1] == 3:
            plot2images(X[0, ...], Y[0,...], Aug_path, title)
        elif len(np.shape(X)) == 4 and np.shape(X)[-1] == 1:
            plot2images(X[0, ..., 0], Y[0, :, :, 0], Aug_path, title)
    #logging.info("Augmented Dimensions:", np.shape(X), np.shape(Y))
    return X, Y