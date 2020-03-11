import numpy as np
from PIL import Image, ImageEnhance
import skimage as sk
import random
from data import plot2images
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter
import math
import os

def data_augentation(X, Y, data_gen_args, data_path_file_name):

    if "horizontal_flip" in data_gen_args:
        if data_gen_args["horizontal_flip"] == True & random.choice([True, False, False]) == True:
            X = np.flip(X, len(np.shape(X))-1)
            Y = np.flip(Y, len(np.shape(Y))-1)

    if "vertical_flip" in data_gen_args:
        if data_gen_args["vertical_flip"] == True & random.choice([True, False, False]) == True:
            X = np.flip(X, len(np.shape(X))-2)
            Y = np.flip(Y, len(np.shape(Y))-2)

    if "rotation_range" in data_gen_args and data_gen_args["rotation_range"] > 0:
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

    '''Zoom to a random position in the image within the zoom range of zoom_range'''
    if "zoom_range" in data_gen_args and data_gen_args["zoom_range"] > 0 & random.choice([True, False]) == True:
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
        value = data_gen_args["gaussian_noise"]
        X = X + np.random.normal(0, value)

    if "gaussian_blur_image" in data_gen_args and data_gen_args["gaussian_blur"] > 0 and random.choice([True, False, False]) == True:
        value = data_gen_args["gaussian_blur"]
        X = gaussian_filter(X, sigma=value)

    if "gaussian_blur_label" in data_gen_args and data_gen_args["gaussian_blur"] > 0 and random.choice([True, False, False]) == True:
        value = data_gen_args["gaussian_blur"]
        Y = gaussian_filter(Y, sigma=value)

    if "contrast_range" in data_gen_args and random.choice([True, False, False]) == True:
            range = np.random.uniform(-1,1) * data_gen_args["contrast_range"] + 1
            min_X = np.min(X)
            X = (X - min_X) * range + min_X

    if "brightness_range" in data_gen_args and random.choice([True, False]) == True:
            range = np.random.uniform(-1,1) * data_gen_args["brightness_range"]
            X = X + range * X

    if "threshold_background_image" in data_gen_args and random.choice([True, False, False]) == True:
        mean_X = np.mean(X)
        X[X < mean_X] = 0

    if "threshold_background_groundtruth" in data_gen_args and random.choice([True, False, False]) == True :
        mean_Y = np.mean(Y) * 0.8
        Y[Y < mean_Y] = 0

    if "binarize_mask" in data_gen_args and data_gen_args["binarize_mask"] == True:
        mean_Y = np.mean(Y)
        Y[Y < mean_Y] = 0
        Y[Y >= mean_Y] = 1

    if "save_augmented_images" in data_gen_args:
        data_path, file_name = os.path.split(data_path_file_name)
        Aug_path = (data_path + '/Augmentations/' )
        os.makedirs("./" + (Aug_path), exist_ok=True)
        title = file_name + ".tif"
        #datacomb = np.concatenate((X, Y), axis=2).astype("uint8")
        #sk.io.imsave(Aug_path + title, datacomb[0,...])
        if len(np.shape(X)) == 5:
            plot2images(X[0, 10, ... ,0], Y[0, 10, :, :,0], Aug_path, title)
        elif len(np.shape(X)) == 4 and np.shape(X)[-1] == 3:
            plot2images(X[0, ...], Y[0,...], Aug_path, title)
        elif len(np.shape(X)) == 4 and np.shape(X)[-1] == 1:
            plot2images(X[0, ..., 0], Y[0, :, :, 0], Aug_path, title)
    #print("Augmented Dimensions:", np.shape(X), np.shape(Y))
    return X, Y