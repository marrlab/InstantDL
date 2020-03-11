from __future__ import print_function
import matplotlib.pylab as plt
import numpy as np
import os
from skimage.io import imsave
import datetime
import skimage

def plottestimage(image, path, title):
    if np.size(np.shape(image)) == 3:
        plt.imshow(image[:,:,0])
        plt.savefig(os.path.join(path + title))
    if np.size(np.shape(image)) == 4:
        imsave(os.path.join(path + title), np.array(image[:,:,:,0]))

def plottestimage_npy(image, path, title):
    print(np.shape(image))
    imsave(path + title + ".tif", image)
    #np.save(path + title + ".npy", image)

def plot2images(image, mask, path,title):
    #if np.shape(mask)[-1] != 3:
    #    mask = mask[:,:,0]
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
    f.write('\n')
    f.write('\n' + "Run started at: " + str(datetime.datetime.now())[:16])
    f.write('\n' + "With lossfunction : " + str(loss) + "for: " + str(epochs) + "epochs")
    f.write('\n' + "The augmentations are: " + str(data_gen_args))
    f.close()
