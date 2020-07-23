
"""
InstantDL
Utils for data evaluation
Written by Dominik Waibel
"""
#ToDo: Finish Test
'''from instantdl import GetPipeLine
import shutil
from instantdl.utils import *

def test_InstanceSegmentation():
    X_true = np.ones((64, 64, 3))
    X_true[10:20, 20:30] = 255
    Y_true = np.zeros((64, 64))
    Y_true[10:20,20:30] = 1
    for i in range(0,5):
        os.makedirs(os.getcwd() +"/tests/segmentation/Instancetestimages/train/image"+str(i)+"/image/", exist_ok=True)
        os.makedirs(os.getcwd() +"/tests/segmentation/Instancetestimages/train/image" + str(i) + "/mask/", exist_ok=True)
        imsave(os.getcwd() + "/tests/segmentation/Instancetestimages/train/image"+str(i)+"/image/image"+str(i)+".jpg", X_true)
        imsave(os.getcwd() + "/tests/segmentation/Instancetestimages/train/image" + str(i) + "/mask/image" + str(i) + ".jpg",
               Y_true)
    for i in range(0,5):
        os.makedirs(os.getcwd() +"/tests/segmentation/Instancetestimages/test/image"+str(i)+"/image/", exist_ok=True)
        os.makedirs(os.getcwd() +"/tests/segmentation/Instancetestimages/test/image" + str(i) + "/mask/", exist_ok=True)
        imsave(os.getcwd() + "/tests/segmentation/Instancetestimages/test/image"+str(i)+"/image/image"+str(i)+".jpg", X_true)
        imsave(os.getcwd() + "/tests/segmentation/Instancetestimages/test/image"+str(i)+"/mask/image"+str(i)+".jpg",Y_true)

    pipeline = GetPipeLine("InstanceSegmentation",
                           os.getcwd() + "/tests/segmentation/Instancetestimages/",
                           None,
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
    #Make sure the networks has changed something
    for i in range(0, 5):
        assert (X_true != imread(os.getcwd() + "/tests/segmentation/Instancetestimages/results/image"+str(i)+".jpg_predict.npy")).all
    #Delete created test data
    #if os.path.exists(os.getcwd()+"/tests/segmentation/Instancetestimages") and os.path.isdir(os.getcwd()+"/tests/segmentation/testimages"):
    #    shutil.rmtree(os.getcwd()+"/tests/segmentation/Instancetestimages")'''