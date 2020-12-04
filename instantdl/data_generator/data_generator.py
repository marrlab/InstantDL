'''
Author: Dominik Waibel

This file contains the functions with which the data is improted and exported before training and testing and after testing.
'''

from instantdl.data_generator.data import *
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from instantdl.data_generator.data_augmentation import data_augentation
import os
import csv as csv
import sys
import copy
import glob
import logging
from keras.utils import to_categorical
from skimage.color import gray2rgb
import warnings

def get_min_max(data_path, folder_name, image_files):
    '''
    This function gets the minimum and maximum pixel values for the folder
    
    Args:
        data_path (str): path to project folder
        folder_name (str): one of the data folder names, e.g. image or groundtruth
        train_image_files (list): list of image files
    
    return: 
        min_value: the minimum pixel value of the dataset 
        max_value: the maximum pixel value of the dataset
    '''
    num_img = len(image_files)
    Xmin = np.empty(num_img)
    Xmax = np.empty(num_img)
    for i, img_file in enumerate(image_files):
        if img_file.endswith(".npy"):
            Xmin[i] = np.min(np.array(np.load(data_path + folder_name + img_file)))
            Xmax[i] = np.max(np.array(np.load(data_path + folder_name + img_file)))
        else:
            Xmin[i] = np.min(np.array(imread(data_path + folder_name + img_file)))
            Xmax[i] = np.max(np.array(imread(data_path + folder_name + img_file)))
    
    min_value = np.min(Xmin)
    max_value = np.max(Xmax)

    logging.info("min value of %s is %s and the max value is %s" % \
                                            (folder_name, min_value, max_value))
    return min_value, max_value

def import_image(path_name):
    '''
    This function loads the image from the specified path
    NOTE: The alpha channel is removed (if existing) for consistency

    Args:
        path_name (str): path to image file
    
    return: 
        image_data: numpy array containing the image data in at the given path. 
    '''
    if path_name.endswith('.npy'):
        image_data = np.array(np.load(path_name))
    else:
        image_data = imread(path_name)
        # If has an alpha channel, remove it for consistency
    if np.array(np.shape(image_data))[-1] == 4:
        image_data = image_data[:,:,0:3]
    return image_data


def image_generator(    Training_Input_shape, batchsize, num_channels, 
                        train_image_file, folder_name, data_path, 
                        X_min, X_max, use_algorithm):
    '''
    This function normalizes the imported images, resizes them and create batches

    Args:
        Training_Input_shape: The dimensions of one image used for training. Can be set in the config.json file
        batchsize: the batchsize used for training
        num_channels: the number of channels of one image. Typically 1 (grayscale) or 3 (rgb)
        train_image_file: the file name
        folder_name: the folder name of the file to be imported
        data_path: the project directory
        X_min: the minimum pixel value of this dataset
        X_max: the maximum pixel value of this dataset
    return: 
        X: a batch of image data with dimensions (batchsize, x-dim, y-dim, [z-dim], channels)
    '''
    X = []
    for i in range(0, batchsize):
        img_file = train_image_file[i]
        image_data = import_image(data_path + folder_name + img_file)
        if np.shape(image_data) != tuple(Training_Input_shape):
            #The Resizing fundtion changes the array values, therefore shift them back to the original range
            min_value = np.min(image_data)
            max_value = np.max(image_data)
            image_data = resize(image_data, Training_Input_shape)
            newmin = np.min(image_data)
            newmax = np.max(image_data)
            image_data = ((image_data - newmin) / (newmax - newmin)) * (max_value - min_value) + min_value
        image_data = (image_data - X_min) / (X_max - X_min)
        X.append(image_data)
    X = np.stack(X, axis = 0)
    if np.shape(X)[-1] != num_channels:
       X= X[..., np.newaxis]
    return X


def training_data_generator(Training_Input_shape, batchsize, num_channels, 
                            num_channels_label, train_image_files, 
                            data_gen_args, data_dimensions,data_path, use_algorithm):
    '''
    Generate the data for training and return images and groundtruth 
    for regression and segmentation

    Args
        Training_Input_shape: The dimensions of one image used for training. Can be set in the config.json file
        batchsize: the batchsize used for training
        num_channels: the number of channels of one image. Typically 1 (grayscale) or 3 (rgb)
        num_channels_label: the number of channels of one groundtruth image. Typically 1 (grayscale) or 3 (rgb)
        train_image_files: list of files in the training dataset
        data_gen_args: augmentation arguments
        data_dimensions: the dimensions of one image or the dimension to which the image should be resized
        data_path: path to the project directory
        use_algorithm: the selected network (UNet, ResNet50 or MRCNN)
    
    return: 
        X_train: batched training data
        Y: label or ground truth
    '''
    Folder_Names = ["/groundtruth/", "/image/", "/image1/", "/image2/", "/image3/", "/image4/", "/image5/", "/image6/", "/image7/"]
    X_min = np.zeros((len(Folder_Names)))
    X_max = np.zeros((len(Folder_Names)))
    for i, folder_name in enumerate(Folder_Names):
        if os.path.isdir(data_path + folder_name) == True:
            X_min[i], X_max[i] = get_min_max(data_path, folder_name, train_image_files)

    while True:
        def grouped(train_image_files, batchsize):
            return zip(*[iter(train_image_files)] * batchsize)
        for train_image_file in grouped(train_image_files, batchsize):
            for index, folder_name in enumerate(Folder_Names):
                if os.path.isdir(data_path + folder_name) == True:
                    if "groundtruth" in folder_name:
                        GT_Input_image_shape = np.array(copy.deepcopy(Training_Input_shape))
                        GT_Input_image_shape[-1] = num_channels_label
                        GT_Input_image_shape = tuple(GT_Input_image_shape)
                        if GT_Input_image_shape[-1] == 0:
                            GT_Input_image_shape = GT_Input_image_shape[0:-1]
                        Y = image_generator(GT_Input_image_shape, batchsize, num_channels_label, 
                                                train_image_file, folder_name, data_path, X_min[index], X_max[index], use_algorithm)
                    if "image" in folder_name and index > 1:
                        imp = image_generator(Training_Input_shape, batchsize, num_channels, train_image_file, 
                                                folder_name, data_path, X_min[index], X_max[index], use_algorithm)
                        X = np.concatenate([X, imp], axis = -1)
                    if "image" in folder_name and index == 1:
                        X = image_generator(Training_Input_shape, batchsize, num_channels, 
                        train_image_file, folder_name, data_path, X_min[index], X_max[index], use_algorithm)
            X_train, Y = data_augentation(X, Y, data_gen_args, data_path + str(train_image_file))
            X_train = np.nan_to_num(X_train)
            Y = np.nan_to_num(Y)
            yield (X_train, Y)


def training_data_generator_classification(Training_Input_shape, 
                                            batchsize, num_channels, 
                                            num_classes, train_image_files, 
                                            data_gen_args, data_path, use_algorithm):
    '''
    Generate the data for training and return images and groundtruth for classification

    Args
        Training_Input_shape: The dimensions of one image used for training. Can be set in the config.json file
        batchsize: the batchsize used for training
        num_channels: the number of channels of one image. Typically 1 (grayscale) or 3 (rgb)
        num_classes: the number of classes of the dataset, set in the config.json
        train_image_files: list of filenames in the training dataset
        data_gen_args: augmentation arguments
        data_path: path to the project directory
        use_algorithm: the selected network (UNet, ResNet50 or MRCNN)
    
    return: 
        X: batched training data
        Y: groundtruth labels
    '''
    csvfilepath = os.path.join(data_path + '/groundtruth/', 'groundtruth.csv')
    Folder_Names = ["/image/", "/image1/", "/image2/", "/image3/", "/image4/", "/image5/", "/image6/", "/image7/"]
    X_min = np.zeros((len(Folder_Names)))
    X_max = np.zeros((len(Folder_Names)))
    for i, folder_name in enumerate(Folder_Names):
        if os.path.isdir(data_path + folder_name) == True:
            X_min[i], X_max[i] = get_min_max(data_path, folder_name, train_image_files)
    logging.info("array of min values: %s" % X_min)
    logging.info("array of max values: %s" % X_max)
    while True:
        def grouped(train_image_files, batchsize):
            return zip(*[iter(train_image_files)] * batchsize)
        for train_image_file in grouped(train_image_files, batchsize):
            for index, folder_name in enumerate(Folder_Names):
                if os.path.isdir(data_path + folder_name) == True:
                    if "image" in folder_name and index > 0:
                        imp = image_generator(Training_Input_shape, batchsize, num_channels, 
                                                        train_image_file, folder_name, data_path, 
                                                        X_min[index], X_max[index], use_algorithm)
                                                        
                        X = np.concatenate([X, imp], axis = -1)
                    if "image" in folder_name and index == 0:
                        X = image_generator(Training_Input_shape, batchsize, num_channels, 
                        train_image_file, folder_name, data_path, X_min[index], X_max[index], use_algorithm)
            
            X, Trash = data_augentation(X, X, data_gen_args, data_path + str(train_image_file))
            label = np.zeros((batchsize))
            for j in range(0, batchsize):
                img_file = train_image_file[j]
                with open(csvfilepath) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row['filename'] == img_file:
                            label[j] = row['groundtruth']
            label = to_categorical(label, num_classes)
            yield (X, label)

 
def testGenerator(Input_image_shape, path, num_channels, test_image_files, use_algorithm):
    '''
    Generate test images for segmentation, regression and classification

    Args
        Input_image_shape: The dimensions of one image used for training. Can be set in the config.json file
        path: path to the project directory
        num_channels: the number of channels of one image. Typically 1 (grayscale) or 3 (rgb)
        test_image_files: list of filenames in the test dataset
        use_algorithm: the selected network (UNet, ResNet50 or MRCNN)
    
    return: 
        X: one image on which a model prediction is executed
    '''
    test_path = path + "/test/"
    batchsize = 1
    Folder_Names = ["/image/", "/image1/", "/image2/", "/image3/", 
                        "/image4/", "/image5/", "/image6/", "/image7/"]
    X_min = np.zeros((len(Folder_Names)))
    X_max = np.zeros((len(Folder_Names)))
    for i, folder_name in enumerate(Folder_Names):
        if os.path.isdir(test_path + folder_name) == True:
            X_min[i], X_max[i] = get_min_max(test_path, folder_name, test_image_files)
    logging.info(test_image_files)
    logging.info("len test files %s" % len(test_image_files))
    while True:
        for test_file in test_image_files:
            test_file = [test_file]
            for index, folder_name in enumerate(Folder_Names):
                if os.path.isdir(test_path + folder_name) == True:
                    imp = image_generator(Input_image_shape, batchsize, num_channels, 
                                    test_file, folder_name, test_path, X_min[index], X_max[index], use_algorithm)
                    if index > 0 :
                        X = np.concatenate([X, imp], axis = -1)
                    else:
                        X = imp
            yield X


def saveUncertainty(path, test_image_files, epi_uncertainty, ali_uncertainty):
    '''
    saves the predicted uncertainty estimates to a .csv file
    Args:
        path: path to the project directory
        test_image_files: list of filenames in the test dataset
        results: the predicted segmentation or image
    return: None
    '''
    logging.info("Save result")
    os.makedirs(path, exist_ok=True)
    with open(path + 'uncertainty.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['filename', 'summed uncertainty', 'mean epistemic uncertainty', 'median epistemic uncertainty', 'mean aleatoric uncertainty', 'median aleatoric uncertainty'])
        for i in range(0, len(epi_uncertainty)-1):
            writer.writerow([test_image_files[i], np.mean(epi_uncertainty[i,...])+np.mean(ali_uncertainty[i,...]), np.mean(epi_uncertainty[i,...]), np.median(epi_uncertainty[i,...]), np.mean(ali_uncertainty[i,...]), np.median(ali_uncertainty[i,...])])


def saveResult(path, test_image_files, results, Input_image_shape):
    '''
    saves the predicted segmentation or image to the Results 
                            folder in the project directory
    
    Args:
        path: path to the project directory
        test_image_files: list of filenames in the test dataset
        results: the predicted segmentation or image
        Input_image_shape: The dimensions of one image used for 
                            training. Can be set in the config.json file
    
    return: None
    '''
    results = results * 255.
    logging.info("Save result")
    os.makedirs(path, exist_ok=True)
    logging.info("shape npyfile %s" % (np.shape(results),))
    logging.info("test_image_files %s" % len(test_image_files))
    for i in range(len(test_image_files)):
        logging.info("filename %s" % test_image_files[i])
        titlenpy = (test_image_files[i] + "_predict")
        logging.info(i)
        if np.shape(results)[-1] == 0:
            results = results[:,:,:,0]
        minnpy = np.min(results[i, ...])
        maxnpy = np.max(results[i, ...])
        npy_resized = np.squeeze(resize(results[i, ...], Input_image_shape))
        newmin = np.min(npy_resized)
        newmax = np.max(npy_resized)
        npy_resized = ((npy_resized - newmin) / (newmax - newmin)) * (maxnpy - minnpy) + minnpy
        plottestimage_npy(npy_resized.astype("uint8"), path, titlenpy)

def saveResult_classification(path, test_image_files, results):
    '''
    saves a .csv file to the results folder in the project directory
            containing the predicted labels on the testset

        path: path to the project directory
        test_image_files: list of filenames in the test dataset
        results: list of predicted labels
    
    return: None
    '''
    logging.info("Save result")
    save_path = (path + '/results/')
    os.makedirs(os.getcwd() + (save_path), exist_ok=True)
    with open(save_path + 'results.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['filename', 'prediction', 'Probability for each possible outcome'])
        for i in range(0, len(results)-1):
            writer.writerow([test_image_files[i], np.argmax(results[i,...]), results[i,...]])

def saveResult_classification_uncertainty(path, test_image_files, results, 
                                                    MCprediction, combined_uncertainty):
    '''
    saves a .csv file to the results folder in the project directory
            containing the predicted labels on the testset
    
    Args:        
        path: path to the project directory
        test_image_files: list of filenames in the test dataset
        results: list of predicted labels
        MCprediction: list of predicted labels based on the likeliest prediction using MC Dropout
        combined_uncertainty: list of uncertaintys for each image
    
    return: None
    '''
    logging.info("Save result")
    save_path = (path + '/results/')
    os.makedirs(os.getcwd() + (save_path), exist_ok=True)
    with open(save_path + 'results.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['filename', 'prediction', 'Probability for each possible outcome',
                                    'MC Prediction', 'Certertainty: 0 is certain, high is uncertain'])
        for i in range(0, len(results)-1):
            writer.writerow([test_image_files[i], np.argmax(results[i,...]), \
                                results[i,...], MCprediction[i], np.abs(combined_uncertainty[i])])


def training_validation_data_split(data_path):
    '''
    Import filenames and split them into train and validation set according to the variable -validation_split = 20%
    Splits files in the train folder into a training and validation dataset and returns both lists containing the filenames
    
    Args
        
        data_path: path to the project directory
    
    return: 
        train_image_files: list of filenames of training data files
        val_image_files: list of filenames of validaton data files
    '''
    image_files = os.listdir(os.path.join(data_path + "/image"))
    lenval = int(len(image_files) * 0.2)
    validation_spilt_id = np.array(list(range(0, len(image_files), int(len(image_files) / lenval))))
    logging.info(validation_spilt_id)
    train_image_files = []
    val_image_files = []
    for i in range(0, len(image_files)):
        if i in validation_spilt_id:
            val_image_files.append(image_files[i])
        if i not in validation_spilt_id:
            train_image_files.append(image_files[i])
    train_image_files = np.random.permutation(train_image_files)
    logging.info("Found: %s images in training set" % len(train_image_files))
    logging.info("Found: %s images in validation set" % len(val_image_files))
    return train_image_files, val_image_files


def get_input_image_sizes(path, use_algorithm):
    '''
    Get the size of the input images and check dimensions

    Args:
        path: path to project directory
        use_algorithm: the selected network (UNet, ResNet50 or MRCNN)
    
    return: 
        Training_Input_shape: the shape of the training data
        num_channels: number of channels
        Input_image_shape: the shape of the input image
    '''
    data_path = path + '/train'
    img_file = os.listdir(data_path + "/image/")[0]
    Input_image_shape = np.array(np.shape(np.array(import_image(data_path + "/image/" + img_file))))
    logging.info("Input shape Input_image_shape %s" % Input_image_shape)
    if use_algorithm in ["Regression", "Segmentation"]:
        logging.info("Input_image_shape %s" % Input_image_shape)
        if int(Input_image_shape[0]) not in [int(16), int(32),int(64),int(128),int(256),int(512),int(1024),int(2048)]:
            if int(Input_image_shape[1]) not in [int(16), int(32),int(64),int(128),int(256),int(512),int(1024),int(2048)]:
                sys.exit("The Input data needs to be of pixel dimensions: 16, 32, 64, 128, 256, 512, 1024 or 2048 in each dimension")

    # If has an alpha channel, remove it for consistency
    Training_Input_shape = copy.deepcopy(Input_image_shape)
    if Training_Input_shape[-1] == 4:
        logging.info("Removing alpha channel")
        Training_Input_shape[-1] = 3

    if all([Input_image_shape[-1] != 1, Input_image_shape[-1] != 3]):
        logging.info("Adding an empty channel dimension to the image dimensions")
        Training_Input_shape = np.array(tuple(Input_image_shape) + (1,))

    if use_algorithm == "Classification":
        Training_Input_shape[-1] = 3

    num_channels = Training_Input_shape[-1]
    input_size = tuple(Training_Input_shape)
    logging.info("Input size is: %s" % (input_size,))
    return tuple(Training_Input_shape), num_channels, Input_image_shape
