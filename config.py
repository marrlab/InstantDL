'''
InstantDL
Written by Dominik Waibel
Config file for the deep learning pipeline.
Set parameters here.
'''

from main import start_learning
from metrics import *
from data_generator.data import write_logbook

'''
you can only set one of the following: Regression, Segmentation, Classification or ObjectDetection
'''
use_algorithm = "Regression"
'''
Set Network parameters: 
num_classes = Number of classes in your Regression
'''
batchsize = 2
Iterations_Over_Dataset = 200
num_classes = 1

''' 
If None it will automatically determine the image size. 
If set to a tuple, please use: (Dim1, Dim2, [Dim3,] Number Channels)
'''
Image_size = None#(256,256,3)
'''
Set the loss function. All functions from keras.metrics are possible, 
as well as custom funcions in "metrics.py
'''
loss_function = "mse"

'''
Data Augmentation: Select the relevant augmentations for your procjet by removing the hashtag
The numbers indicate the amount of augmentation applied
'''

data_gen_args = dict(#save_augmented_images = True,
                    #resample_images=(0.5,0.5,0.5),
                    #std_normalization = False,
                    #feature_scaling = False,
                    horizontal_flip=True,
                    vertical_flip=True,
                    #poission_noise = 1,
                    #rotation_range = 45, #angle in degrees.
                    #zoom_range = 2, #Factor of zoom
                    #contrast_range = 0.2, #Percentage for contrast change [0,1]
                    #brightness_range = 0.2, #percentage of brightness range [0,1]
                    #gamma_shift = 0,
                    #threshold_background_image = True,
                    #threshold_background_groundtruth = True,
                    #gaussian_blur_image = 2, #Standard deviation for Gaussian kernel,
                    #gaussian_blur_label = 2, #Standard deviation for Gaussian kernel,
                    #binarize_mask = True
                    )
'''pretrained_weights =  put None or (path + "BinaryCrosspretrained_weights_KaggleNerve.hdf5") here '''
use_pretrained_weights = True
'''Set calculate uncertainty to True or False'''
path = 'data/CPC25'

if use_pretrained_weights == True:
    pretrained_weights = (path + "/logs/pretrained_weights_CPC25.hdf5")
else:
    pretrained_weights = None

calculate_uncertainty = False

'''
write_logbook(path, epochs, loss_function, data_gen_args)
'''
start_learning(use_algorithm, path, pretrained_weights, batchsize, Iterations_Over_Dataset, data_gen_args, loss_function, num_classes, Image_size, calculate_uncertainty)
