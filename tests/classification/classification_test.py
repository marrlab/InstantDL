
"""
InstantDL
Written by Dominik Waibel
"""

from instantdl import GetPipeLine
import shutil
from instantdl.utils import *
import csv

def test_Classification():
    os.makedirs(os.getcwd() + "/tests/classification/testimages/train/image/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/classification/testimages/train/groundtruth/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/classification/testimages/test/image/", exist_ok=True)
    os.makedirs(os.getcwd() + "/tests/classification/testimages/test/groundtruth/", exist_ok=True)
    X_true = np.ones((32, 32, 3))

    with open(os.getcwd() + "/tests/classification/testimages/train/groundtruth/" + 'groundtruth.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['filename', 'groundtruth'])
        for i in range(0,20):
            imsave(os.getcwd() + "/tests/classification/testimages/train/image/image" + str(i) + ".jpg", X_true)
            writer.writerow(["image" + str(i) + ".jpg", str(1)])

    with open(os.getcwd() + "/tests/classification/testimages/test/groundtruth/" + 'groundtruth.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['filename', 'groundtruth'])
        for i in range(0,5):
            imsave(os.getcwd() + "/tests/classification/testimages/test/image/image" + str(i) + ".jpg", X_true)
            writer.writerow(["image" + str(i) + ".jpg", str(1)])

    configs = {"use_algorithm": "Classification",
               "path": "./tests/classification/testimages/",
               "batchsize": 1,
               "iterations_over_dataset": 1,
               "num_classes": 2,
               "evaluation": False}

    pipeline = GetPipeLine(configs)

    pipeline.run()
    K.clear_session()
    # Make sure the networks has changed something
    assert (os.path.isfile(os.getcwd() + "/tests/classification/testimages/results/results.cvs"))
    #Delete created test data
    if os.path.exists(os.getcwd()+"/tests/classification/testimages") and os.path.isdir(os.getcwd()+"/tests/classification/testimages"):
        shutil.rmtree(os.getcwd()+"/tests/classification/testimages")
