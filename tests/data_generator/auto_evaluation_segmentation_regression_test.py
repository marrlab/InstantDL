"""
InstantDL
Utils for data evaluation
Written by Dominik Waibel
"""

from instantdl.data_generator.auto_evaluation_segmentation_regression import *

def test_threshold():
    img = np.ones((10,10))
    img[4:6,4:6] = 255
    img_thresh_true = copy.deepcopy(img)
    img_thresh_true[4:6, 4:6] = 0
    img_thresh = threshold(img)
    assert (img_thresh_true == img_thresh).all

def test_binarize():
    img = np.ones((10,10))
    img[4:6,4:6] = 255
    img_binary_true = np.zeros((10,10))
    img_binary_true[4:6, 4:6] = 1
    img_binary = binarize(img)
    assert (img_binary_true == img_binary).all

def test_normalize():
    img = np.ones((10,10))
    img[4:6,4:6] = 255
    img_norm_true = copy.deepcopy(img)
    img_norm_true = img_norm_true / 255.
    img_norm = normalize(img)
    assert (img_norm_true == img_norm).all

def test_AUC():
    pred = np.ones((10, 10))
    pred[4:6, 4:6] = 0
    gt = np.ones((10, 10))
    gt[5:6, 4:6] = 0
    auc_res = AUC(pred, gt)
    assert auc_res == 0.9897959183673469

def test_getPearson():
    gt = np.ones((10, 10))
    np.fill_diagonal(gt, 5)
    pearson, pearson_all = getPearson(gt, gt)
    assert pearson == 1.0

