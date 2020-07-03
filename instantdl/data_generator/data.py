from __future__ import print_function
import matplotlib.pylab as plt
import numpy as np
import os
from skimage.io import imsave
import datetime
import logging

def plottestimage_npy(image, path, title):
    '''
    saves the networks prediction to the results folder

    Args
        image: the networks prediction
        path: results path to where the image is saved
        title: image name
    
    return: None
    '''
    logging.info("shape of image %s" % (np.shape(image),))
    imsave(path + title + ".tif", image)

def plot2images(image, mask, path, title):
    '''
    saves a pair of images (image and mask) to directory

    Args   
        image: image data
        mask: mask data
        path: path to where the image pair is saved
        title: image name
    
    return: None
    '''
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.subplot(1, 2, 2)
    if len(np.shape(mask)) > 2:
        mask = mask[:,:,0]
    plt.imshow(mask)
    plt.title("Groundtruth")
    plt.savefig(os.path.join(path, str(title) + ".png"), dpi=50)
    plt.tight_layout()
    plt.close()


def write_logbook(path, epochs, loss, data_gen_args):
    f = open(path + '/Logbook.txt', 'a')
    f.write('\n' + "Run started at: " + str(datetime.datetime.now())[:16])
    f.write('\n' + "With lossfunction: " + str(loss) + " for : " + str(epochs) + " epochs")
    f.write('\n' + "The augmentations are: " + str(data_gen_args))
    f.write('\n')
    f.close()
