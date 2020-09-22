import sys
import logging
import argparse
import os
import json
from instantdl.data_generator.data_generator import *
from instantdl.data_generator.auto_evaluation_classification import classification_evaluation
from instantdl.data_generator.auto_evaluation_segmentation_regression import segmentation_regression_evaluation
from instantdl.segmentation.UNet_models import UNetBuilder
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import time
import tensorflow as tf
tf.get_logger().setLevel('WARNING')
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from instantdl.classification.ResNet50 import ResNet50
from instantdl.segmentation.RCNNSettings import RCNNInferenceConfig, train, detect
import instantdl.segmentation.RCNNmodel as RCNNmodel
from instantdl.segmentation.RCNNSettings import RCNNConfig
tf.config.experimental.list_physical_devices('GPU')
from instantdl.classification.ResNet50 import ResNet50 #, get_imagenet_weights
import glob
from keras.optimizers import Adam, SGD
from instantdl.data_generator.data import write_logbook
logging.basicConfig(level=logging.INFO)

def load_json(file_path):
    with open(file_path, 'r') as stream:
        return json.load(stream)
