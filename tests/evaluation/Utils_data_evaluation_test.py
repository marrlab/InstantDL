"""
InstantDL
Utils for data evaluation
Written by Dominik Waibel
"""

from instantdl.evaluation.Utils_data_evaluation import *
import copy
from skimage.io import imsave
import shutil

def test_normalize():
    img = np.ones((10,10))
    img[4:6,4:6] = 255
    img_norm_true = copy.deepcopy(img)
    img_norm_true = img_norm_true / 255.
    img_norm = normalize(img)
    assert (img_norm_true == img_norm).all

def test_import_images():
    X_true = np.ones((128,128,3))
    os.makedirs("./data_generator/testimages/train/image/", exist_ok=True)
    imsave("./data_generator/testimages/train/image/image.jpg", X_true)
    imsave("./data_generator/testimages/train/image/image1.jpg", X_true)
    imsave("./data_generator/testimages/train/image/image2.jpg", X_true)
    data, names = import_images("./data_generator/testimages/train/image/", ["image", "image1", "image2"], ".jpg")
    assert (data == (255*np.ones((3,128,128,3)))).all()

    # Delete created test data
    if os.path.exists("./data_generator/testimages") and os.path.isdir("./data_generator/testimages"):
        shutil.rmtree("./data_generator/testimages")
    if os.path.exists("./data_generator/testimages_classification") and os.path.isdir("./data_generator/testimages_classification"):
        shutil.rmtree("./data_generator/testimages_classification")
