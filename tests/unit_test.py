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