import pytest
import numpy as np
from instantdl.data_generator.data_generator import *
from skimage.io import imsave, imread
import os
import pandas as pd

def test_get_min_max():
    os.makedirs("./data_generator/testimages/", exist_ok=True)
    imsave("./data_generator/testimages/image.jpg", np.zeros((128,128,3)))
    imsave("./data_generator/testimages/image1.jpg", 255.*np.ones((128, 128, 3)))
    assert get_min_max("./", "./data_generator/testimages/", ["image.jpg", "image1.jpg"]) == (0.0, 255.0)

def test_import_image():
    os.makedirs("./data_generator/testimages/", exist_ok=True)
    imsave("./data_generator/testimages/image1.jpg", 255.*np.ones((128,128,3)))
    assert np.shape(import_image("./data_generator/testimages/image1.jpg")) == (128,128,3)
    assert np.max(import_image("./data_generator/testimages/image1.jpg")) == 255.0
    assert np.mean(import_image("./data_generator/testimages/image1.jpg")) == 255.0
    assert np.mean(import_image("./data_generator/testimages/image1.jpg")) == 255.0

def test_image_generator():
    os.makedirs("./data_generator/testimages/", exist_ok=True)
    imsave("./data_generator/testimages/image.jpg", np.zeros((128,128,3)))
    imsave("./data_generator/testimages/image1.jpg", 255.*np.ones((128, 128, 3)))
    assert np.shape(
        image_generator((128, 128, 3), 2, 3, ["image.jpg", "image1.jpg"], "/", "./data_generator/testimages/", 0., 255., "Segmentation")) == (2, 128, 128, 3)
    assert np.max(
        image_generator((128, 128, 3), 2, 3, ["image.jpg", "image1.jpg"], "/", "./data_generator/testimages/", 0., 255., "Segmentation")) == 1.
    assert np.min(
        image_generator((128, 128, 3), 2, 3, ["image.jpg", "image1.jpg"], "/", "./data_generator/testimages/", 0., 255., "Segmentation")) == 0.

def test_saveResult():
    os.makedirs("./data_generator/testimages/", exist_ok=True)
    image = np.zeros((1,128,128,2))
    image[:10:50,10:50,:] = 255.
    saveResult("./data_generator/testimages/", ["result_image"], image, (128,128,3))
    assert np.shape(imread("./data_generator/testimages/result_image_predict.tif")) == (128,128,3)
    assert np.min(imread("./data_generator/testimages/result_image_predict.tif")) == 0.
    assert np.max(imread("./data_generator/testimages/result_image_predict.tif")) == 1.

def test_saveResult_classification():
    results = np.ones((3,2))
    saveResult_classification("./data_generator/testimages/", ["image.jpg", "image1.jpg"], results)
    res = pd.read_csv("./data_generator/testimages/results/results.csv")
    assert res["prediciton"].values[0] == 0
    assert res["prediciton"].values[1] == 0
    assert res["filename"].values[0] == "image.jpg"
    assert res["filename"].values[1] == "image1.jpg"
    assert res["Probability for each possible outcome"].values[0] == "[1. 1.]"
    assert res["Probability for each possible outcome"].values[1] == "[1. 1.]"

def test_saveUncertainty():
    epi_uncertainty = np.zeros((1,128,128,1))
    ali_uncertainty = 255.*np.ones((1,128, 128, 1))
    saveUncertainty("./data_generator/testimages/", ["image.jpg", "image1.jpg"], epi_uncertainty, ali_uncertainty)
    assert (imread("./data_generator/testimages/image.jpg") == epi_uncertainty).all
    assert (imread("./data_generator/testimages/image1.jpg") == ali_uncertainty).all

def test_get_input_image_sizes():
    os.makedirs("./data_generator/testimages/train/image/", exist_ok=True)
    imsave("./data_generator/testimages/train/image/image.jpg", np.zeros((128,128,3)))
    imsave("./data_generator/testimages/train/image/image1.jpg", 255.*np.ones((128, 128, 3)))
    Training_Input_shape, num_channels, Input_image_shape = get_input_image_sizes("./data_generator/testimages/", "Segmentation")
    assert Training_Input_shape == (128,128,3)
    assert num_channels == 3
    assert (Input_image_shape == ([128,128,3])).all()