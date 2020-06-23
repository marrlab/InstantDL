import pytest
import numpy as np
from instantdl.data_generator.data_augmentation import data_augentation

def test_data_augentation():
    shape = (128,128,3)
    X = np.zeros(shape)
    data_gen_args = {"save_augmented_images": False,
                    "resample_images": True,
                    "std_normalization": True,
                    "feature_scaling": True,
                    "horizontal_flip": True,
                    "vertical_flip": True,
                    "poission_noise": 0.2,
                    "rotation_range": 10,
                    "zoom_range": 2,
                    "contrast_range": 2,
                    "brightness_range": 2,
                    "gamma_shift": 1,
                    "threshold_background_image": True,
                    "threshold_background_groundtruth": True,
                    "binarize_mask": True
    }
    data_path_file_name = "Random"
    assert np.shape(data_augentation(X, X, data_gen_args, data_path_file_name)[0]) == shape
    assert np.max(data_augentation(X, X, data_gen_args, data_path_file_name)[0]) == 0.
    assert np.min(data_augentation(X, X, data_gen_args, data_path_file_name)[0]) == 0.0
    X[10:50, 10:50, :] = 255.
    data_gen_args = {"save_augmented_images": False,
                     "resample_images": False,
                     "std_normalization": False,
                     "feature_scaling": False,
                     "horizontal_flip": True,
                     "vertical_flip": True,
                     "poission_noise": False,
                     "rotation_range": False,
                     "zoom_range": 1.5,
                     "contrast_range": False,
                     "brightness_range": False,
                     "gamma_shift": False,
                     "threshold_background_image": False,
                     "threshold_background_groundtruth": False,
                     "binarize_mask": False
                     }
    assert np.shape(data_augentation(X, X, data_gen_args, data_path_file_name)[0]) == shape
    assert np.max(data_augentation(X, X, data_gen_args, data_path_file_name)[0]) == 255.
    assert np.min(data_augentation(X, X, data_gen_args, data_path_file_name)[0]) == 0.