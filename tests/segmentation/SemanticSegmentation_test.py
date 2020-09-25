"""
InstantDL
Utils for data evaluation
Written by Dominik Waibel
"""

from instantdl import GetPipeLine
import shutil
from instantdl.utils import *

def test_2DSegmentation():
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

    configs = {"use_algorithm": "SemanticSegmentation",
               "path": "./tests/segmentation/testimages/",
               "pretrained_weights": False,
               "batchsize": 1,
               "iterations_over_dataset": 1,
               "evaluation": False}

    pipeline = GetPipeLine(**configs)

    pipeline.run()
    K.clear_session()
    # Make sure the networks has changed something
    for i in range(0, 5):
        assert (X_true != imread(os.getcwd() + "/tests/segmentation/testimages/results/image"+str(i)+".jpg_predict.tif")).all
    #Delete created test data
    if os.path.exists(os.getcwd()+"/tests/segmentation/testimages") and os.path.isdir(os.getcwd()+"/tests/segmentation/testimages"):
        shutil.rmtree(os.getcwd()+"/tests/segmentation/testimages")

def test_3DSegmentation():
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

    configs = {"use_algorithm": "SemanticSegmentation",
               "path": "./tests/segmentation/testimages/",
               "pretrained_weights": False,
               "batchsize": 1,
               "iterations_over_dataset": 1,
               "evaluation": False}

    pipeline = GetPipeLine(**configs)

    pipeline.run()
    K.clear_session()
    # Make sure the networks has changed something
    for i in range(0, 5):
        assert (X_true != np.array((imread(os.getcwd() + "/tests/segmentation/testimages/results/image"+str(i)+".tif_predict.tif")).astype("uint8"))).all
    #Delete created test data
    if os.path.exists(os.getcwd()+"/tests/segmentation/testimages") and os.path.isdir(os.getcwd()+"/tests/segmentation/testimages"):
        shutil.rmtree(os.getcwd()+"/tests/segmentation/testimages")