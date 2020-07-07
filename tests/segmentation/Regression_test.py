from instantdl import GetPipeLine
import shutil
from instantdl.utils import *


def test_2DRegression():
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/train/image/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/train/groundtruth/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/test/image/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/test/groundtruth/", exist_ok=True)
    X_true = np.ones((128, 128, 3))
    Y_true = 255. * np.ones((128, 128, 3))
    for i in range(0,20):
        imsave(os.getcwd() + "/tests/segmentation/testimages/train/image/image"+str(i)+".jpg", X_true)
        imsave(os.getcwd() + "/tests/segmentation/testimages/train/groundtruth/image"+str(i)+".jpg", Y_true)
    for i in range(0,5):
        imsave(os.getcwd() + "/tests/segmentation/testimages/test/image/image"+str(i)+".jpg", X_true)
        imsave(os.getcwd() + "/tests/segmentation/testimages/test/groundtruth/image"+str(i)+".jpg", Y_true)

    pipeline = GetPipeLine("Regression",
                           os.getcwd() + "/tests/segmentation/testimages/",
                           False,
                           1,
                           1,
                           {},
                           "mse",
                           1,
                           None,
                           False,
                           False)

    pipeline.run()
    K.clear_session()
    #Delete created test data
    if os.path.exists(os.getcwd()+"/tests/segmentation/testimages") and os.path.isdir(os.getcwd()+"/tests/segmentation/testimages"):
        shutil.rmtree(os.getcwd()+"/tests/segmentation/testimages")


def test_3DRegression():
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/train/image/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/train/groundtruth/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/test/image/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/segmentation/testimages/test/groundtruth/", exist_ok=True)
    X_true = np.ones((128, 128, 32, 1))
    Y_true = 255. * np.ones((128, 128, 32, 1))
    for i in range(0, 20):
        imsave(os.getcwd() + "/tests/segmentation/testimages/train/image/image" + str(i) + ".tif", X_true)
        imsave(os.getcwd() + "/tests/segmentation/testimages/train/groundtruth/image" + str(i) + ".tif", Y_true)
    for i in range(0, 5):
        imsave(os.getcwd() + "/tests/segmentation/testimages/test/image/image" + str(i) + ".tif", X_true)
        imsave(os.getcwd() + "/tests/segmentation/testimages/test/groundtruth/image" + str(i) + ".tif", Y_true)

    pipeline = GetPipeLine("Regression",
                           os.getcwd() + "/tests/segmentation/testimages/",
                           False,
                           1,
                           1,
                           {},
                           "mse",
                           1,
                           None,
                           False,
                           False)

    pipeline.run()
    K.clear_session()

    #Delete created test data
    if os.path.exists(os.getcwd()+"/tests/segmentation/testimages") and os.path.isdir(os.getcwd()+"/tests/segmentation/testimages"):
        shutil.rmtree(os.getcwd()+"/tests/segmentation/testimages")