import sys
import logging
import argparse
import os
import json
from data_generator.data_generator import *
from data_generator.auto_evaluation_classification import classification_evaluation
from data_generator.auto_evaluation_segmentation_regression import segmentation_regression_evaluation
from segmentation.UNet_models import UNetBuilder
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import time
import tensorflow as tf
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from classification.ResNet50 import ResNet50
from segmentation.RCNNSettings import RCNNInferenceConfig, train, detect
import segmentation.RCNNmodel as RCNNmodel
from segmentation.RCNNSettings import RCNNConfig
tf.config.experimental.list_physical_devices('GPU')
from classification.ResNet50 import ResNet50 #, get_imagenet_weights
import glob
from keras.optimizers import Adam, SGD
from data_generator.data import write_logbook
import logging
logging.basicConfig(level=logging.DEBUG)

def load_json(file_path):
    with open(file_path, 'r') as stream:
        return json.load(stream)
