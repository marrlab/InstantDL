import pytest
import numpy as np
from instantdl.data_generator.data_augmentation import data_augentation
from instantdl.data_generator.data_generator import get_min_max, import_image, image_generator
from skimage.io import imsave
import os

def test_data_augentation():
    shape = (128,128,3)
    X = np.zeros(shape)
    data_gen_args = {"save_augmented_images": False,
                    "resample_images": True,
                    "std_normalization": True,
                    "feature_scaling": True,
                    "horizontal_flip": True,
                    "vertical_flip": True,
                    "poission_noise": True,
                    "rotation_range": True,
                    "zoom_range": True,
                    "contrast_range": True,
                    "brightness_range": True,
                    "gamma_shift": True,
                    "threshold_background_image": True,
                    "threshold_background_groundtruth": True,
                    "binarize_mask": True
    }
    data_path_file_name = "Random"
    assert np.shape(data_augentation(X, X, data_gen_args, data_path_file_name)[0]) == shape
    assert np.max(data_augentation(X, X, data_gen_args, data_path_file_name)[0]) == 0.0
    assert np.min(data_augentation(X, X, data_gen_args, data_path_file_name)[0]) == 0.0

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
    imsave("./testimages/image1.jpg", 255.*np.ones((128, 128, 3)))
    assert np.shape(
        image_generator((128, 128, 3), 2, 3, ["image.jpg", "image1.jpg"], "/", "./testimages/", 0., 255.)) == (
           2, 128, 128, 3)
    assert np.max(
        image_generator((128, 128, 3), 2, 3, ["image.jpg", "image1.jpg"], "/", "./testimages/", 0., 255.)) == 1.
    assert np.min(
        image_generator((128, 128, 3), 2, 3, ["image.jpg", "image1.jpg"], "/", "./testimages/", 0., 255.)) == 0.