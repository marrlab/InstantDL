import pytest
import numpy as np
from instantdl.data_generator.data_generator import *
from skimage.io import imsave, imread
import os

def test_get_min_max():
    os.makedirs("./testimages/", exist_ok=True)
    imsave("./testimages/image.jpg", np.zeros((128,128,3)))
    imsave("./testimages/image1.jpg", 255.*np.ones((128, 128, 3)))
    assert get_min_max("./", "./testimages/", ["image.jpg", "image1.jpg"]) == (0.0, 255.0)

def test_import_image():
    os.makedirs("./testimages/", exist_ok=True)
    imsave("./testimages/image1.jpg", 255.*np.ones((128,128,3)))
    assert np.shape(import_image("./testimages/image1.jpg")) == (128,128,3)
    assert np.max(import_image("./testimages/image1.jpg")) == 255.0
    assert np.mean(import_image("./testimages/image1.jpg")) == 255.0
    assert np.mean(import_image("./testimages/image1.jpg")) == 255.0

def test_image_generator():
    os.makedirs("./testimages/", exist_ok=True)
    imsave("./testimages/image.jpg", np.zeros((128,128,3)))
    imsave("./testimages/image1.jpg", 1.*np.ones((128, 128, 3)))
    assert np.shape(
        image_generator((128, 128, 3), 2, 3, ["image.jpg", "image1.jpg"], "/", "./testimages/", 0., 255., "Segmentation")) == (2, 128, 128, 3)
    assert np.max(
        image_generator((128, 128, 3), 2, 3, ["image.jpg", "image1.jpg"], "/", "./testimages/", 0., 255., "Segmentation")) == 1.
    assert np.min(
        image_generator((128, 128, 3), 2, 3, ["image.jpg", "image1.jpg"], "/", "./testimages/", 0., 255., "Segmentation")) == 0.

def test_saveResult():
    os.makedirs("./testimages/", exist_ok=True)
    image = np.zeros((1,128,128,3))
    image[:10:50,10:50,:] = 1.
    saveResult("./testimages/", ["result_image"], image, (128,128,3))
    assert np.shape(imread("./testimages/result_image_predict.tif")) == (128,128,3)
    assert np.min(imread("./testimages/result_image_predict.tif")) == 0.
    assert np.max(imread("./testimages/result_image_predict.tif")) == 255.