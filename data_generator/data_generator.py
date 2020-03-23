'''
Author: Dominik Waibel

This file contains the functions with which the data is improted and exported before training and testing and after testing.
'''

from data import *
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from data_generator.data_augmentation import data_augentation
import os
import csv as csv
import sys
import copy
import glob
from keras.utils import to_categorical
from skimage.color import gray2rgb
import warnings

'''Get the minimum and maximum value of a dataset for normalization'''
def get_min_max(data_path, folder_name, train_image_files):
    num_img = len(train_image_files)
    Xmin = np.empty(num_img)
    Xmax = np.empty(num_img)
    for i, img_file in enumerate(train_image_files):
        if img_file.endswith(".npy"):
            Xmin[i] = np.min(np.array(np.load(data_path + folder_name + img_file)))
            Xmax[i] = np.max(np.array(np.load(data_path + folder_name + img_file)))
        else:
            Xmin[i] = np.min(np.array(imread(data_path + folder_name + img_file)))
            Xmax[i] = np.max(np.array(imread(data_path + folder_name + img_file)))
    min = np.min(Xmin)
    max = np.max(Xmax)
    print("min value of", folder_name,"is", min, " max valueis", max)
    return min, max

'''Import an image from the directory and remove the alpha channel if existing'''
def import_image(path_name):
    if path_name.endswith('.npy'):
        image_data = np.array(np.load(path_name))
    else:
        image_data = imread(path_name)
        # If has an alpha channel, remove it for consistency
    if np.array(np.shape(image_data))[-1] == 4:
        image_data = image_data[:,:,0:3]
    return image_data

'''Normalize the improted images, resize them and create batches'''
def image_generator(Training_Input_shape, batchsize, num_channels, train_image_file, folder_name, data_path, X_min, X_max, use_algorithm):
    X = []
    for i in range(0, batchsize):
        img_file = train_image_file[i]
        image_data = import_image(data_path + folder_name + img_file)
        if np.shape(image_data) != tuple(Training_Input_shape):
            #The Resizing fundtion changes the array values, therefore shift them back to the original range
            min = np.min(image_data)
            max = np.max(image_data)
            image_data = resize(image_data, Training_Input_shape)
            newmin = np.min(image_data)
            newmax = np.max(image_data)
            image_data = ((image_data - newmin) / (newmax - newmin)) * (max - min) + min
        image_data = (image_data - X_min) / (X_max - X_min)
        X.append(image_data)
    X = np.stack(X, axis = 0)
    if np.shape(X)[-1] != num_channels:
       X= X[..., np.newaxis]
    return X

'''Generate the data for training and return images and groundtruth for regression and segmentation'''
def training_data_generator(Training_Input_shape, batchsize, num_channels, num_channels_label, train_image_files, data_gen_args, data_dimensions,data_path, use_algorithm):
    Folder_Names = ["/groundtruth/", "/image/", "/image1/", "/image2/", "/image3/", "/image4/", "/image5/", "/image6/", "/image7/"]
    X_min = np.zeros((len(Folder_Names)))
    X_max = np.zeros((len(Folder_Names)))
    for i, folder_name in enumerate(Folder_Names):
        if os.path.isdir(data_path + folder_name) == True:
            X_min[i], X_max[i] = get_min_max(data_path, folder_name, train_image_files)
    print("array of min values:", X_min)
    print("array of max values:", X_max)
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
                        Y = image_generator(GT_Input_image_shape, batchsize, num_channels_label, train_image_file, folder_name, data_path, X_min[index], X_max[index], use_algorithm)
                    if "image" in folder_name and index > 1:
                        imp = image_generator(Training_Input_shape, batchsize, num_channels, train_image_file, folder_name, data_path, X_min[index], X_max[index], use_algorithm)
                        X = np.concatenate([X, imp], axis = -1)
                    if "image" in folder_name and index == 1:
                        X = image_generator(Training_Input_shape, batchsize, num_channels, train_image_file, folder_name, data_path, X_min[index], X_max[index], use_algorithm)
            X_train, Y = data_augentation(X, Y, data_gen_args, data_path + str(train_image_file))
            #print("Training data", np.shape(X_train), np.mean(X_train), np.max(X_train), np.shape(Y), np.mean(Y), np.max(Y))
            yield (X_train, Y)

'''Generate the data for training and return images and groundtruth for classification'''
def training_data_generator_classification(Training_Input_shape, num_channels, batchsize, num_classes, train_image_files, data_gen_args, data_path, use_algorithm):
    csvfilepath = os.path.join(data_path + '/groundtruth/', 'groundtruth.csv')
    Folder_Names = ["/image/", "/image1/", "/image2/", "/image3/", "/image4/", "/image5/", "/image6/", "/image7/"]
    X_min = np.zeros((len(Folder_Names)))
    X_max = np.zeros((len(Folder_Names)))
    for i, folder_name in enumerate(Folder_Names):
        if os.path.isdir(data_path + folder_name) == True:
            X_min[i], X_max[i] = get_min_max(data_path, folder_name, train_image_files)
    print("array of min values:", X_min)
    print("array of max values:", X_max)
    while True:
        def grouped(train_image_files, batchsize):
            return zip(*[iter(train_image_files)] * batchsize)
        for train_image_file in grouped(train_image_files, batchsize):
            for index, folder_name in enumerate(Folder_Names):
                if os.path.isdir(data_path + folder_name) == True:
                    if "image" in folder_name and index > 0:
                        imp = image_generator(Training_Input_shape, batchsize, num_channels, train_image_file, folder_name, data_path, X_min[index], X_max[index], use_algorithm)
                        #imp = image_generator(Training_Input_shape, batchsize, num_channels, train_image_file, folder_name, data_path, 0, 255)
                        X = np.concatenate([X, imp], axis = -1)
                    if "image" in folder_name and index == 0:
                        X = image_generator(Training_Input_shape, batchsize, num_channels, train_image_file, folder_name, data_path, X_min[index], X_max[index], use_algorithm)
            X, Trash = data_augentation(X, X, data_gen_args, data_path + str(train_image_file))
            label = np.zeros((batchsize))
            for j in range(0, batchsize):
                img_file = train_image_file[j]
                with open(csvfilepath) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row['filename'] == img_file:
                            label[j] = row['label']
                        else:
                            warnings.warn("Ńo classification label found for image")
            label = to_categorical(label, num_classes)
            #print("Training data", np.shape(X), np.max(X), label)
            yield (X, label)

'''Generate test images for segmentation, regression and classification'''
def testGenerator(Input_image_shape, path, num_channels, test_image_files, use_algorithm):
    test_path = path + "/test/"
    batchsize = 1
    Folder_Names = ["/image/", "/image1/", "/image2/", "/image3/", "/image4/", "/image5/", "/image6/", "/image7/"]
    X_min = np.zeros((len(Folder_Names)))
    X_max = np.zeros((len(Folder_Names)))
    for i, folder_name in enumerate(Folder_Names):
        if os.path.isdir(test_path + folder_name) == True:
            X_min[i], X_max[i] = get_min_max(test_path, folder_name, test_image_files)
    print(test_image_files)
    print("len test files", len(test_image_files))
    while True:
        for test_file in test_image_files:
            test_file = [test_file]
            for index, folder_name in enumerate(Folder_Names):
                if os.path.isdir(test_path + folder_name) == True:
                    imp = image_generator(Input_image_shape, batchsize, num_channels, test_file, folder_name, test_path, 0, 255, use_algorithm)
                    if index > 0 :
                        X = np.concatenate([X, imp], axis = -1)
                    else:
                        X = imp
            print("Test", np.shape(X))
            yield X

'''Save the result for regression and segmentation as images and .npy files'''
def saveResult(path, test_image_files, npyfile, Input_image_shape):
    npyfile = npyfile * 255
    print("Save result")
    os.makedirs(path, exist_ok=True)
    print("shape npyfile", np.shape(npyfile))
    print("test_image_files", len(test_image_files))
    for i in range(len(test_image_files)):
        print("filename", test_image_files[i])
        titlenpy = (test_image_files[i] + "_predict")
        print(i)
        if np.shape(npyfile)[-1] == 0:
            npyfile = npyfile[:,:,:,0]
        minnpy = np.min(npyfile[i, ...])
        maxnpy = np.max(npyfile[i, ...])
        npy_resized = np.squeeze(resize(npyfile[i, ...], Input_image_shape))
        newmin = np.min(npy_resized)
        newmax = np.max(npy_resized)
        npy_resized = ((npy_resized - newmin) / (newmax - newmin)) * (maxnpy - minnpy) + minnpy
        plottestimage_npy(npy_resized.astype("uint8"), path, titlenpy)

'''Save the result for classification to the result.csv file'''
def saveResult_classification(path, test_image_files, results):
    print("Save result")
    save_path = (path + '/results/')
    os.makedirs("./" + (save_path), exist_ok=True)
    with open(save_path + 'results.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['Filename', 'Prediciton', 'Probability for each possible outcome'])
        for i in range(0, len(results)-1):
            writer.writerow([test_image_files[i], np.argmax(results[i,...]), results[i,...]])

def saveResult_classification_uncertainty(path, test_image_files, results, MCprediction, combined_uncertainty):
    print("Save result")
    save_path = (path + '/results/')
    os.makedirs("./" + (save_path), exist_ok=True)
    with open(save_path + 'results.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['Filename', 'Prediciton', 'Probability for each possible outcome', 'MC Prediction', 'Certertainty: 0 is certain, high is uncertain'])
        for i in range(0, len(results)-1):
            writer.writerow([test_image_files[i], np.argmax(results[i,...]), results[i,...], MCprediction[i], combined_uncertainty[i]])

'''Import filenames and split them into train and validation set according to the variable -validation_split = 20%    '''
def training_validation_data_split(data_path):
    image_files = os.listdir(os.path.join(data_path + "/image"))
    lenval = int(len(image_files) * 0.2)
    validation_spilt_id = np.array(list(range(0, len(image_files), int(len(image_files) / lenval))))
    print(validation_spilt_id)
    train_image_files = []
    val_image_files = []
    for i in range(0, len(image_files)):
        if i in validation_spilt_id:
            val_image_files.append(image_files[i])
        if i not in validation_spilt_id:
            train_image_files.append(image_files[i])
    train_image_files = np.random.permutation(train_image_files)
    print("Found:", len(train_image_files), "images in training set")
    print("Found:", len(val_image_files), "images in validation set")
    return train_image_files, val_image_files

'''Get the size of the input iamges and check dimensions'''
def get_input_image_sizes(path, use_algorithm):
    data_path = path + '/train'
    img_file = os.listdir(data_path + "/image/")[0]
    Input_image_shape = np.array(np.shape(np.array(import_image(data_path + "/image/" + img_file))))
    print("Input shape Input_image_shape", str(Input_image_shape))
    if use_algorithm is "Regression" or use_algorithm is "Segmentation":
        print(Input_image_shape)
        if int(Input_image_shape[0]) not in [int(16), int(32),int(64),int(128),int(256),int(512),int(1024),int(2048)]:
            if int(Input_image_shape[1]) not in [int(16), int(32),int(64),int(128),int(256),int(512),int(1024),int(2048)]:
                sys.exit("The Input data needs to be of pixel dimensions: 16, 32, 64, 128, 256, 512, 1024 or 2048 in each dimension")

    # If has an alpha channel, remove it for consistency
    Training_Input_shape = copy.deepcopy(Input_image_shape)
    if Training_Input_shape[-1] == 4:
        print("Removing alpha channel")
        Training_Input_shape[-1] = 3

    if all([Input_image_shape[-1] != 1, Input_image_shape[-1] != 3]):
        print("Adding an empty channel dimension to the image dimensions")
        Training_Input_shape = np.array(tuple(Input_image_shape) + (1,))

    if use_algorithm == "Classification":
        Training_Input_shape[-1] = 3

    num_channels = Training_Input_shape[-1]
    #print("num channels are: ", num_channels)
    #Training_Input_shape[-1] = Training_Input_shape[-1] * num_channels
    input_size = tuple(Training_Input_shape)
    print("Input size is: ", input_size)
    return tuple(Training_Input_shape), num_channels, Input_image_shape
