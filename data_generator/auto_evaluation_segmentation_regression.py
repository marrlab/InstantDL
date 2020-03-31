'''Import the dependencies'''
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Computer Modern Roman"
import os, fnmatch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Computer Modern Roman"
from scipy.stats import pearsonr
from scipy.spatial.distance import jaccard
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Computer Modern Roman"
import copy
from matplotlib.colorbar import Colorbar
from skimage.color import rgb2gray, gray2rgb

from evaluation.Utils_data_evaluation import prepare_data_for_evaluation

def threshold(img):
    img[img < np.mean(img)] = 0
    return img

def binarize(data):
    data_mean = np.mean(data)
    data[data <= data_mean] = 0
    data[data > data_mean] = 1
    return data

def normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)

def getPearson(gt, pred):
    pearson_all = np.zeros(np.shape(gt)[0])
    for index in range(len(pearson_all)):

        gtp = np.array(gt[index].flatten())
        predp = np.array(pred[index].flatten())
        pearson_all[index], pvalue = pearsonr(gtp,predp)

    pearson = np.median(pearson_all)
    return pearson, pearson_all

def save_2Dimages(data, names, z_index, path_name, image_name):
    print("plot len data", len(data))
    plt.figure(figsize = (len(data)*8,10))
    grid = plt.GridSpec(4, len(data), height_ratios=[0.01,0.3,0.02,0.01], width_ratios=len(data) * [1])
    print("shape data save", np.shape(data))
    for img, image in enumerate(data):
        shape = np.shape(image)
        print("shape plot image", np.shape(image))
        plts = plt.subplot(grid[0:2, img])
        if names[img] == "abs_errormap":
            max_val = np.max(np.abs(image))
            pltcberror = plt.title("errormap", size = 50)
            pltcberror = plt.imshow(rgb2gray(image), cmap='seismic', vmin = -max_val, vmax = max_val)
        elif names[img] == "uncertainty":
            plts = plt.title(names[img] + str(np.mean(image)), size = 50)
            max_val = np.max(np.abs(image))
            pltcbunc = plt.title("uncertainty \n"+ "Average:" +str(np.mean(image))[0:4], size = 50)
            pltcbunc = plt.imshow(np.max(image)-rgb2gray(image), cmap='Greys', vmin = -max_val, vmax = max_val)
        else:
            plts = plt.imshow(rgb2gray(image), cmap='Greys')
            plts = plt.title(names[img], size = 50)
            pltz = plt.xlabel("x (Pixel)", size=50)
        if img == 0:
            pltz = plt.ylabel("y (Pixel)", size=50)

        plts = plt.tick_params(labelsize=50)
        if img > 0:
            plts = plt.tick_params(
            axis='y',
            labelleft = False)
            plts = plt.tick_params(labelsize=50)
        if names[img] == "abs_errormap":
            print("colorbar")
            if "uncertainty" in names:
                cbax = plt.subplot(grid[2:3,-2])
            else:
                cbax = plt.subplot(grid[2:3,-1])
            cbax = Colorbar(ax = cbax, mappable = pltcberror,orientation="horizontal")
            cbax.set_ticks([])
            cbax.set_label('Over-     Under- \n prediction', size = 50)
        if names[img] == "uncertainty":
            print("colorbar uncertanty")
            cbay = plt.subplot(grid[2:3,-1])
            cbay = Colorbar(ax = cbay,  mappable = pltcbunc, orientation="horizontal")
            cbay.set_ticks([])
            cbay.set_label('Low         High', size = 50)
        plt.savefig(os.path.join(path_name, z_index + image_name + ".png"), dpi=50,bbox_inches='tight')
    plt.show()
    plt.close()


def save_3Dimages(data, names, z_index, path_name, image_name):
    plt.figure(figsize=(len(data) * 8, 12))
    print("plot len data", len(data))
    grid = plt.GridSpec(5, len(data), height_ratios=[0.01, 0.3, 0.1, 0.02, 0.01], width_ratios=len(data) * [1])

    print("shape data save", np.shape(data))
    for img, image in enumerate(data):
        shape = np.shape(image)
        print("shape image", np.shape(image))
        x_image = image[int(shape[0] / 2), ...]
        z_image = image[:, int(shape[1] / 2), :]
        print("shape plot image", np.shape(x_image))
        plts = plt.subplot(grid[0:2, img])
        plts = plt.plot(np.arange(np.shape(x_image)[1]), [np.shape(x_image)[1] / 2] * np.shape(x_image)[1], color='0.5')
        plts = plt.title(names[img], size=50)
        plts = plt.tick_params(labelsize=50)
        plts = plt.xticks([])
        if img > 0:
            plts = plt.yticks([])
        if img == 0:
            plts = plt.ylabel("y (Pixel)", size=50)
            plts = plt.yticks(size=40)
        if names[img] == "abs_errormap":
            max_val = np.max(np.abs(x_image))
            pltcberror = plt.imshow(rgb2gray(x_image), cmap='seismic', vmin=-max_val, vmax=max_val)
            pltcberror = plt.subplot(grid[2, img])
            pltcberror = plt.plot(np.arange(np.shape(z_image)[1]), [np.shape(z_image)[0] / 2] * np.shape(z_image)[1],
                                  color='0.5')
        elif names[img] == "uncertainty":
            max_val = np.max(np.abs(x_image))
            pltcbunc = plt.title("uncertainty \n" + "Average:" + str(np.mean(image))[0:4], size=50)
            pltcunc = plt.imshow(rgb2gray(x_image), cmap='Greys', vmin=-max_val, vmax=max_val)
            pltcunc = plt.subplot(grid[2, img])
            pltcunc = plt.plot(np.arange(np.shape(z_image)[1]), [np.shape(z_image)[0] / 2] * np.shape(z_image)[1],
                               color='0.5')
        else:
            pltxz = plt.imshow(x_image, cmap='Greys')
            pltxz = plt.subplot(grid[2, img])
            pltxz = plt.plot(np.arange(np.shape(z_image)[1]), [np.shape(z_image)[0] / 2] * np.shape(z_image)[1],
                             color='0.5')
        if img > 0:
            pltxy = plt.tick_params(axis='y', labelleft=False)
            pltxy = plt.tick_params(labelsize=40)
            pltxz = plt.yticks([])
        if img == 0:
            pltxz = plt.ylabel("y (Pixel)", size=50)
        if names[img] == "abs_errormap":
            max_val = np.max(np.abs(x_image))
            pltcerror = plt.imshow(rgb2gray(z_image), cmap='seismic', vmin=-max_val, vmax=max_val)
            pltcberror = plt.yticks([])
        elif names[img] == "uncertainty":
            max_val = np.max(np.abs(x_image))
            pltcunc = plt.imshow(rgb2gray(np.max(z_image) - z_image), cmap='Greys', vmin=-max_val, vmax=max_val)
        else:
            pltz = plt.imshow(z_image, cmap='Greys')
            pltz = plt.tick_params(labelsize=40)
            pltz = plt.xlabel("x (Pixel)", size=50)
        if img == 0:
            pltz = plt.ylabel("z (Pixel)", size=50)
        if img > 0:
            pltz = plt.tick_params(axis='y', labelleft=False, labelsize=40)
        # pltz = plt.tick_params(labelsize=40)

    if "abs_errormap" in names:
        if "uncertainty" in names:
            cbax = plt.subplot(grid[3:4, -2])
        else:
            cbax = plt.subplot(grid[3:4, -1])
        cbax = Colorbar(ax=cbax, mappable=pltcerror, orientation="horizontal")
        cbax.set_ticks([])
        cbax.set_label('Over-  Under- \n prediction', size=50)
        cbax.ax.tick_params(labelsize=50)
    if "uncertainty" in names:
        cbax = plt.subplot(grid[3:4, -1])
        cbax = Colorbar(ax=cbax, mappable=pltcunc, orientation="horizontal")
        cbax.set_ticks([])
        cbax.set_label('Low      High', size=50)
        cbax.ax.tick_params(labelsize=50)
    print(path_name + z_index  + image_name + ".png")
    plt.savefig(path_name + z_index  + image_name + ".png", dpi=50, bbox_inches='tight')
    plt.show()
    plt.close()


def quantitative_evaluation(path, data, names):
    groundtruth_norm = (data[names.index("groundtruth")] - np.min(data[names.index("groundtruth")])) / (np.max(data[names.index("groundtruth")]) - np.min(data[names.index("groundtruth")]))
    prediction_norm = (data[names.index("prediction")] - np.min(data[names.index("prediction")])) / (np.max(data[names.index("prediction")]) - np.min(data[names.index("prediction")]))
    rel_errormap_norm = np.abs(np.divide(data[names.index("abs_errormap")], groundtruth_norm, out=np.zeros_like(data[names.index("abs_errormap")]),
                                         where=groundtruth_norm != 0))

    groundtruth_binary = binarize(data[names.index("groundtruth")])
    prediction_binary = binarize(data[names.index("groundtruth")])
    groundtruth_norm = normalize(data[names.index("groundtruth")])
    prediction_norm = normalize(data[names.index("groundtruth")])

    from sklearn.metrics import mutual_info_score
    from sklearn.metrics import accuracy_score, adjusted_rand_score, auc, roc_auc_score
    Pearson, Pearson_all = getPearson(data[names.index("prediction")], data[names.index("groundtruth")])

    print("Metriccs:")
    print("The accuracy on the normalized dataset is: ",
          1 - np.mean(np.square(groundtruth_norm - prediction_norm)) / (groundtruth_norm.size))
    print("The median relative error on the normalized dataset is: ", np.median(rel_errormap_norm) * 100, "%")
    print("The mean absolute error on the normalized dataset is: ", np.mean(data[names.index("abs_errormap")]))
    print("The Pearson coefficient is: ", np.median(1-Pearson))
    print("The Jaccard index is: ", jaccard(prediction_binary.flatten(), groundtruth_binary.flatten()))
    print("The AUC is:", roc_auc_score(prediction_binary.flatten(), groundtruth_binary.flatten()))
    # print("The Information score is: ", mutual_info_score(np.concatenate(np.concatenate(prediction_norm)), np.concatenate(np.concatenate(groundtruth_norm))))
    # print("The rand score is:" , adjusted_rand_score(np.concatenate(np.concatenate(groundtruth_norm)), np.concatenate(np.concatenate(prediction_norm))))
    f = open(path + '/Error analysis.txt', 'w')
    f.write('\n' + "The median relative error on the normalized dataset is: " + str(
        np.median(rel_errormap_norm)) + " percent")
    f.write('\n' + "The mean absolute error on the normalized dataset is: " + str(np.abs(np.mean(data[names.index("abs_errormap")]))))
    f.write('\n' + "The Pearson coefficient is: " + str(1-np.abs(Pearson)))
    f.write('\n' + "The Jaccard index is: " + str(jaccard(prediction_binary.flatten(), groundtruth_binary.flatten())))
    f.write('\n' + "The AUC is:" + str(roc_auc_score(prediction_binary.flatten(), groundtruth_binary.flatten())))
    f.close()

def visual_assesment(path, data, names):
    image_names = np.load(path + "/insights/image_names.npy")

    '''Function to execute image generation and saving'''
    os.makedirs(path + "/evaluation/", exist_ok=True)

    # Threshold the uncertainty map to obtain more meaningful images
    if "uncertainty" in names:
        data[names.index("uncertainty")] = threshold(data[names.index("uncertainty")])

    for index, image in enumerate(data):
        if names[index] not in ["abs_errormap"]:
            if names[index] not in ["abs_errormap"]:
                print(index)
                print(np.mean(np.mean(image)))
                print(np.shape(image))
                data[index, ...] = normalize(data[index, ...])

    names_out = copy.deepcopy(names)
    report_dir = "./" + path + "/evaluation/"
    print("Length of data", len(data))
    if len(np.shape(data[0])) == 3 or len(np.shape(data[0])) == 4 and np.shape(data[0])[-1] == 3:
        for index in range(len(data[0])):
            print("index", index)
            print("saving 2D image")
            '''Taking the slices in z-dimension one after another'''
            save_2Dimages(np.array(data)[:, index, ...], names, str(index), report_dir, image_names[index])

    if len(np.shape(data[0])) == 5 or len(np.shape(data[0])) == 5 and np.shape(data[0])[-1] == 4:
        for index in range(len(data[1])):
            print(index)
            print("saving 3D image")
            '''Taking the slices in z-dimension one after another'''
            print("data shape", np.shape(np.array(data)))
            print("save 3d", report_dir)
            save_3Dimages(np.array(data)[:, index, ...], names, str(index), report_dir, image_names[index])


def segmentation_regression_evaluation(path):
    savepath = path + "/insights/"
    max_images = 200000
    prepare_data_for_evaluation(path, max_images)
    # Load data from numpy stacks
    data_names = ["image", "image1", "image2", "image3", "prediction", "groundtruth", "abs_errormap", "uncertainty"]
    datain = []
    names = []
    for name in data_names:
        if os.path.isfile(path + "/insights/" + name + ".npy"):
            import_data = np.array(np.load(path + "/insights/" + name + ".npy").astype('float'))
            print(name, np.shape(import_data))
            datain.append(import_data)
            names.append(str(name))
    data = np.array(datain)
    image_names = np.load(path + "/insights/image_names.npy")
    print(names)
    print(np.shape(data))
    quantitative_evaluation(savepath, data, names)
    visual_assesment(path, data, names)
