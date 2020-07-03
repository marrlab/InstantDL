"""
InstantDL
Utils for data evaluation
Written by Dominik Waibel
"""

from instantdl.data_generator.auto_evaluation_classification import *
import os
import pandas as pd
import shutil

def test_get_auc():
    y_test = np.array([[1, 0], [0, 1]])
    y_score = np.array([[0.8, 0], [0.5, 0]])
    n_classes = 2
    os.makedirs("./data_generator/testimages_classification/insights/", exist_ok=True)
    auc = get_auc("./data_generator/testimages_classification/", y_test, y_score, n_classes)
    print(auc)
    assert auc == [1.0, 0.5]

def test_load_data():
    os.makedirs("./data_generator/testimages_classification/test/groundtruth/", exist_ok=True)
    labels = {"filename": ['image.jpg', 'image1.jpg', 'image2.jpg'], 'groundtruth': [0, 1, 1]}
    gt = pd.DataFrame(labels, columns=['filename', 'groundtruth'])
    gt.to_csv("./data_generator/testimages_classification/test/groundtruth/groundtruth.csv")

    os.makedirs("./data_generator/testimages_classification/results/", exist_ok=True)
    labels = {"filename": ['image.jpg', 'image1.jpg', 'image2.jpg'], 'prediction': [0, 1, 1],
              "Probability for each possible outcome": [[.8], [1.], [.2]]}
    gt = pd.DataFrame(labels, columns=['filename', 'prediction', "Probability for each possible outcome"])
    gt.to_csv("./data_generator/testimages_classification/results/results.csv")
    Groundtruth, Results, Sigmoid_output = load_data("./data_generator/testimages_classification/")
    print(Groundtruth)
    print(Results)
    print(Sigmoid_output)
    assert Groundtruth.all() == np.array([[0], [1], [1]]).all()
    assert Results.all() == np.array([0, 1, 1]).all()
    print(Sigmoid_output[0])
    assert Sigmoid_output[0] == ['0.8']
    assert Sigmoid_output[1] == ['1.0']
    assert Sigmoid_output[2] == ['0.2']

    # Delete created test data
    if os.path.exists("./data_generator/testimages") and os.path.isdir("./data_generator/testimages"):
        shutil.rmtree("./data_generator/testimages")
    if os.path.exists("./data_generator/testimages_classification") and os.path.isdir("./data_generator/testimages_classification"):
        shutil.rmtree("./data_generator/testimages_classification")
