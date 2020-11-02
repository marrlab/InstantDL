
"""
InstantDL
Utils for data evaluation
Written by Dominik Waibel
"""

from instantdl import GetPipeLine
import shutil
from instantdl.utils import *

def test_2DRegression():
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/train/image/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/train/groundtruth/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/test/image/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/test/groundtruth/", exist_ok=True)
    X_true = np.ones((32, 32, 3))
    Y_true = 255. * np.ones((32, 32, 3))
    for i in range(0,20):
        imsave(os.getcwd() + "/tests/segmentation/testimages/train/image/image"+str(i)+".jpg", X_true)
        imsave(os.getcwd() + "/tests/segmentation/testimages/train/groundtruth/image"+str(i)+".jpg", Y_true)
    for i in range(0,5):
        imsave(os.getcwd() + "/tests/segmentation/testimages/test/image/image"+str(i)+".jpg", X_true)
        imsave(os.getcwd() + "/tests/segmentation/testimages/test/groundtruth/image"+str(i)+".jpg", Y_true)

    configs = {"use_algorithm": "Regression",
               "path": "./tests/segmentation/testimages/",
               "pretrained_weights": False,
               "batchsize": 1,
               "image_size": False,
               "iterations_over_dataset": 1,
               "evaluation": False}

    pipeline = GetPipeLine(configs)

    pipeline.run()
    K.clear_session()
    assert (os.path.isfile(os.getcwd() + "/tests/segmentation/testimages/logs/pretrained_weights.hdf5"))
    #Make sure the networks has changed something
    for i in range(0, 5):
        assert (X_true != np.array(imread(os.getcwd() + "/tests/segmentation/testimages/results/image"+str(i)+".jpg_predict.tif").astype("uint8"))).all

    if os.path.exists(os.getcwd()+"/tests/segmentation/testimages") and os.path.isdir(os.getcwd()+"/tests/segmentation/testimages"):
        shutil.rmtree(os.getcwd()+"/tests/segmentation/testimages")

def test_3DRegression():
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/train/image/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/train/groundtruth/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/test/image/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/test/groundtruth/", exist_ok=True)
    X_true = np.ones((32, 32, 16))
    Y_true = 255. * np.ones((32, 32, 16))
    for i in range(0, 20):
        imsave(os.getcwd() + "/tests/segmentation/testimages/train/image/image" + str(i) + ".tif", X_true)
        imsave(os.getcwd() + "/tests/segmentation/testimages/train/groundtruth/image" + str(i) + ".tif", Y_true)
    for i in range(0, 5):
        imsave(os.getcwd() + "/tests/segmentation/testimages/test/image/image" + str(i) + ".tif", X_true)
        imsave(os.getcwd() + "/tests/segmentation/testimages/test/groundtruth/image" + str(i) + ".tif", Y_true)

    configs = {"use_algorithm": "Regression",
               "path": "./tests/segmentation/testimages/",
               "pretrained_weights": False,
               "batchsize": 1,
               "image_size": False,
               "iterations_over_dataset": 1,
               "evaluation": False}

    pipeline = GetPipeLine(configs)

    pipeline.run()
    K.clear_session()
    assert (os.path.isfile(os.getcwd() + "/tests/segmentation/testimages/logs/pretrained_weights.hdf5"))
    # Make sure the networks has changed something
    for i in range(0, 5):
        assert (X_true != np.array((imread(os.getcwd() + "/tests/segmentation/testimages/results/image"+str(i)+".tif_predict.tif")).astype("uint8"))).all
    #Delete created test data
    if os.path.exists(os.getcwd()+"/tests/segmentation/testimages") and os.path.isdir(os.getcwd()+"/tests/segmentation/testimages"):
        shutil.rmtree(os.getcwd()+"/tests/segmentation/testimages")
