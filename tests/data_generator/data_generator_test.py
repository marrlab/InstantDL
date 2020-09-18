"""
InstantDL
Utils for data evaluation
Written by Dominik Waibel
"""

from instantdl.data_generator.data_generator import *
from skimage.io import imsave, imread
import os
import pandas as pd
import shutil

def test_get_min_max():
    os.makedirs(os.getcwd()+"/tests/data_generator/testimages/", exist_ok=True)
    imsave(os.getcwd()+"/tests/data_generator/testimages/image.jpg", np.zeros((128,128,3)))
    imsave(os.getcwd()+"/tests/data_generator/testimages/image1.jpg", 255.*np.ones((128, 128, 3)))
    assert get_min_max(os.getcwd(), "/tests/data_generator/testimages/", ["image.jpg", "image1.jpg"]) == (0.0, 255.0)


def test_import_image():
    os.makedirs(os.getcwd()+"/tests/data_generator/testimages/", exist_ok=True)
    imsave(os.getcwd()+"/tests/data_generator/testimages/image1.jpg", 255.*np.ones((128,128,3)))
    assert np.shape(import_image(os.getcwd()+"/tests/data_generator/testimages/image1.jpg")) == (128,128,3)
    assert np.max(import_image(os.getcwd()+"/tests/data_generator/testimages/image1.jpg")) == 255.0
    assert np.mean(import_image(os.getcwd()+"/tests/data_generator/testimages/image1.jpg")) == 255.0
    assert np.mean(import_image(os.getcwd()+"/tests/data_generator/testimages/image1.jpg")) == 255.0

def test_image_generator():
    os.makedirs(os.getcwd()+"/tests/data_generator/testimages/", exist_ok=True)
    imsave(os.getcwd()+"/tests/data_generator/testimages/image.jpg", np.zeros((128,128,3)))
    imsave(os.getcwd()+"/tests/data_generator/testimages/image1.jpg", 255.*np.ones((128, 128, 3)))
    assert np.shape(
        image_generator((128, 128, 3), 2, 3, ["image.jpg", "image1.jpg"], "/", os.getcwd()+"/tests/data_generator/testimages/", 0., 255., "Segmentation")) == (2, 128, 128, 3)
    assert np.max(
        image_generator((128, 128, 3), 2, 3, ["image.jpg", "image1.jpg"], "/", os.getcwd()+"/tests/data_generator/testimages/", 0., 255., "Segmentation")) == 1.
    assert np.min(
        image_generator((128, 128, 3), 2, 3, ["image.jpg", "image1.jpg"], "/", os.getcwd()+"/tests/data_generator/testimages/", 0., 255., "Segmentation")) == 0.

def test_training_data_generator():
    os.makedirs(os.getcwd()+"/tests/data_generator/testimages/train/image/", exist_ok=True)
    os.makedirs(os.getcwd()+"/tests/data_generator/testimages/train/groundtruth/", exist_ok=True)
    X_true = np.ones((128,128,3))
    #Y_true = 255.*np.ones((128,128,3))
    Y_true = np.zeros((128,128,1))
    Y_true[50:60,50:60,:] = 255
    Y_true[80:90,80:90,:] = 255
    imsave(os.getcwd()+"/tests/data_generator/testimages/train/image/image.jpg", X_true)
    imsave(os.getcwd()+"/tests/data_generator/testimages/train/groundtruth/image.jpg", Y_true)
    imsave(os.getcwd()+"/tests/data_generator/testimages/train/image/image1.jpg", X_true)
    imsave(os.getcwd()+"/tests/data_generator/testimages/train/groundtruth/image1.jpg", Y_true)
    imsave(os.getcwd()+"/tests/data_generator/testimages/train/image/image2.jpg", X_true)
    imsave(os.getcwd()+"/tests/data_generator/testimages/train/groundtruth/image2.jpg", Y_true)
    generator = training_data_generator((3,128,128,3), 1, 3, 1, ["image.jpg","image1.jpg", "image2.jpg"],
                                   {}, 3,os.getcwd()+"/tests/data_generator/testimages/train/", "Regression",'mse')
    assert ((next(generator)[0])== X_true).all
    assert ((next(generator)[1])== Y_true).all
    assert ((next(generator)[0])== X_true).all
    assert ((next(generator)[1])== Y_true).all
    
    # test malis loss case
    Y_malis = np.zeros((128,128,1))
    Y_malis[50:60,50:60,:] = 1
    Y_malis[80:90,80:90,:] = 2
    generator = training_data_generator((128,128,3), 1, 3, 1, ["image.jpg","image1.jpg", "image2.jpg"],
                                   {}, 2,os.getcwd()+"/tests/data_generator/testimages/train/", "SemanticSegmentation",'malis loss')
    assert ((next(generator)[0])== X_true).all
    assert ((next(generator)[1])== Y_malis).all
    assert ((next(generator)[0])== X_true).all
    assert ((next(generator)[1])== Y_malis).all
    

def test_training_data_generator_classification():
    os.makedirs(os.getcwd()+"/tests/data_generator/testimages_classification/train/image/", exist_ok=True)
    os.makedirs(os.getcwd()+"/tests/data_generator/testimages_classification/train/groundtruth/", exist_ok=True)
    X_true = np.zeros((128, 128, 3))
    imsave(os.getcwd()+"/tests/data_generator/testimages_classification/train/image/image.jpg", X_true)
    imsave(os.getcwd()+"/tests/data_generator/testimages_classification/train/image/image1.jpg", X_true)
    imsave(os.getcwd()+"/tests/data_generator/testimages_classification/train/image/image2.jpg", X_true)
    labels = {"filename": ['image.jpg', 'iamge1.jpg', 'image2.jpg'], 'groundtruth': [0, 1, 2]}
    gt = pd.DataFrame(labels, columns=['filename', 'groundtruth'])
    gt.to_csv(os.getcwd()+"/tests/data_generator/testimages_classification/train/groundtruth/groundtruth.csv")
    class_generator = training_data_generator_classification((3,128,128,3), 1, 3, 3,
                                                             ["image.jpg","image1.jpg", "image2.jpg"],  {},
                                                             os.getcwd()+"/tests/data_generator/testimages_classification/train/", "Classification")
    assert ((next(class_generator)[0])== X_true).all
    assert ((next(class_generator)[1])== [1.,0.,0.,]).all
    assert ((next(class_generator)[0])== X_true).all
    assert ((next(class_generator)[1])== [1.,0.,0.,]).all

#def test_testGenerator(Input_image_shape, path, num_channels, test_image_files, use_algorithm):
#    testGenerator(Input_image_shape, path, num_channels, test_image_files, use_algorithm)

def test_saveResult():
    os.makedirs(os.getcwd()+"/tests/data_generator/testimages/", exist_ok=True)
    image = np.zeros((1,128,128,2))
    image[:10:50,10:50,:] = 255.
    saveResult(os.getcwd()+"/tests/data_generator/testimages/", ["result_image"], image, (128,128,3))
    assert np.shape(imread(os.getcwd()+"/tests/data_generator/testimages/result_image_predict.tif")) == (128,128,3)
    assert np.min(imread(os.getcwd()+"/tests/data_generator/testimages/result_image_predict.tif")) == 0.
    assert np.max(imread(os.getcwd()+"/tests/data_generator/testimages/result_image_predict.tif")) == 1.


def test_saveResult_classification():
    results = np.ones((3,2))
    os.makedirs(os.getcwd()+"/tests/data_generator/testimages/", exist_ok=True)
    os.makedirs(os.getcwd()+"/tests/data_generator/testimages/results/", exist_ok=True)
    saveResult_classification(os.getcwd()+"/tests/data_generator/testimages/", ["image.jpg", "image1.jpg"], results)
    res = pd.read_csv(os.getcwd()+"/tests/data_generator/testimages/results/results.csv")
    assert res["prediction"].values[0] == 0
    assert res["prediction"].values[1] == 0
    assert res["filename"].values[0] == "image.jpg"
    assert res["filename"].values[1] == "image1.jpg"
    assert res["Probability for each possible outcome"].values[0] == "[1. 1.]"
    assert res["Probability for each possible outcome"].values[1] == "[1. 1.]"

def test_saveResult_classification_uncertainty():
    results = np.ones((3, 2))
    MCprediction = np.ones((3, 2))
    combined_uncertainty = np.ones((3, 1))
    os.makedirs(os.getcwd() + "/tests/data_generator/testimages/", exist_ok=True)
    saveResult_classification(os.getcwd()+"/tests/data_generator/testimages/", ["image.jpg", "image1.jpg"], results)
    os.makedirs(os.getcwd() + "/tests/data_generator/testimages/", exist_ok=True)
    saveResult_classification_uncertainty(os.getcwd()+"/tests/data_generator/testimages/", ["image.jpg", "image1.jpg"], results,
                                          MCprediction, combined_uncertainty)
    res = pd.read_csv(os.getcwd()+"/tests/data_generator/testimages/results/results.csv")
    assert res["prediction"].values[0] == 0
    assert res["prediction"].values[1] == 0
    assert res["filename"].values[0] == "image.jpg"
    assert res["filename"].values[1] == "image1.jpg"
    assert res["Probability for each possible outcome"].values[0] == "[1. 1.]"
    assert res["Probability for each possible outcome"].values[1] == "[1. 1.]"
    assert res["Certertainty: 0 is certain, high is uncertain"].values[0] == "[1.]"
    assert res["Certertainty: 0 is certain, high is uncertain"].values[1] == "[1.]"
    assert res["MC Prediction"].values[0] == "[1. 1.]"
    assert res["MC Prediction"].values[1] == "[1. 1.]"

def test_saveUncertainty():
    epi_uncertainty = np.zeros((1,128,128,1))
    ali_uncertainty = 255.*np.ones((1,128, 128, 1))
    saveUncertainty(os.getcwd()+"/tests/data_generator/testimages/", ["image.jpg", "image1.jpg"], epi_uncertainty, ali_uncertainty)
    assert (imread(os.getcwd()+"/tests/data_generator/testimages/image.jpg") == epi_uncertainty).all
    assert (imread(os.getcwd()+"/tests/data_generator/testimages/image1.jpg") == ali_uncertainty).all

def test_get_input_image_sizes():
    os.makedirs(os.getcwd()+"/tests/data_generator/testimages/train/image/", exist_ok=True)
    imsave(os.getcwd()+"/tests/data_generator/testimages/train/image/image.jpg", np.zeros((128,128,3)))
    imsave(os.getcwd()+"/tests/data_generator/testimages/train/image/image1.jpg", 255.*np.ones((128, 128, 3)))
    Training_Input_shape, num_channels, Input_image_shape = get_input_image_sizes(os.getcwd()+"/tests/data_generator/testimages/", "Segmentation")
    assert Training_Input_shape == (128,128,3)
    assert num_channels == 3
    assert (Input_image_shape == ([128,128,3])).all()

    #Delete created test data
    if os.path.exists(os.getcwd()+"/tests/data_generator/testimages") and os.path.isdir(os.getcwd()+"/tests/data_generator/testimages"):
        shutil.rmtree(os.getcwd()+"/tests/data_generator/testimages")
    if os.path.exists(os.getcwd()+"/tests/data_generator/testimages_classification") and os.path.isdir(os.getcwd()+"/tests/data_generator/testimages_classification"):
        shutil.rmtree(os.getcwd()+"/tests/data_generator/testimages_classification")
