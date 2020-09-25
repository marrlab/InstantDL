"""
InstantDL
Utils for data evaluation
Written by Dominik Waibel
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Computer Modern Roman"
from skimage import io
from skimage.io import imread
from skimage.color import gray2rgb, rgb2gray
from skimage.transform import resize
import os
import logging

def normalize(data):
	'''
	:param data: data to be normalized0
	:return: normalize data between 0 and 1
	'''
	mindata = np.min(data)
	maxdata = np.max(data)
	data = (data - mindata) / (maxdata - mindata)
	data = np.nan_to_num(data)
	return data

def import_images(import_dir, files, new_file_ending):
	'''
	:param import_dir: directory from where the images are imported
	:param files: list of filenames
	:param new_file_ending: file ending to be attached to the imagenames
	:return: returns a stack containing the image data and their names in the correct order
	'''
	data = []
	names = []
	for file in files:
		if new_file_ending is not None:
			file = (file + new_file_ending)
		if file.endswith(".npy"):
			imp = np.array(np.load(os.path.join(import_dir, file)))
		else:
			imp = np.array(imread(os.path.join(import_dir, file)))
		if np.shape(imp)[-1] == 1:
			imp = imp[...,-1]
		if np.shape(imp)[-1] is not 3:
			imp = gray2rgb(imp)
		data.append(imp)
		names.append(file)
	data = np.array(data)
	logging.info("Checking dimensions:")
	if np.shape(data)[-1] == 1:
		data = data[...,0]
	logging.info("Shape of data imported is:")
	logging.info(np.shape(data))
	return data, names

def calcerrormap(prediction, groundtruth):
	'''
	:param prediction: stack of images in the prediction dataset
	:param groundtruth: stack of images in the groundtruth dataset
	:return: stack with the absolute and relative differene between the prediction and groundtruth
	'''
	logging.info(np.shape(groundtruth))
	logging.info(np.shape(prediction))
	groundtruth = np.asarray(groundtruth, dtype=float)
	prediction = np.asarray(prediction, dtype=float)
	groundtruth_norm = (groundtruth - np.mean(groundtruth))/(np.std(groundtruth))
	prediction_norm = (prediction - np.mean(prediction))/(np.std(prediction))
	groundtruth_fs = (groundtruth - np.min(groundtruth))/(np.max(groundtruth)- np.min(groundtruth))
	prediction_fs = (prediction - np.min(prediction))/(np.max(prediction)- np.min(prediction))
	abs_errormap_norm = ((groundtruth_fs - prediction_fs))
	rel_errormap_norm = np.abs(np.divide(abs_errormap_norm, groundtruth_norm, out=np.zeros_like(abs_errormap_norm), where=groundtruth_norm!=0))
	rel_error_norm = np.mean(np.concatenate(np.concatenate((rel_errormap_norm))))
	logging.info("The relative Error over the normalized dataset is:",rel_error_norm, " best ist close to zero")
	return abs_errormap_norm, rel_errormap_norm

def prepare_data_for_evaluation(root_dir, max_images):
	'''
	:param root_dir: path to directory
	:param max_images: the maximum number of images to to be evaluated
	:return: executes the quantitative and visual asssesment of the model predition
			saves quantitative results to the insights folder and visual results to the evaluation folder
	'''
	test_dir = root_dir + "/test/"
	results_dir = root_dir + "/results/"
	report_dir = root_dir + "/evaluation/"
	insights_dir = root_dir + "/insights/"
	if os.path.isdir(test_dir + "/image/") == True:
		RCNN = False
	else:
		RCNN = True
	#Set results dir only for RCNN
	results_dir = root_dir + "/results/"
	groundtruth_exists = False
	if os.path.exists(root_dir + "/test/groundtruth/") and os.path.isdir(root_dir + "/test/groundtruth/"):
		groundtruth_exists = True
	if RCNN == True:
		logging.info("Resizing MaskRCNN images to 256, 256 dimensins to make them stackable")
		image_fnames = os.listdir(test_dir)
		image_fnames = image_fnames[0:max_images]
		image = []
		groundtruth = []
		predictions = []
		logging.info("importing", len(image_fnames), "files")
		for name in image_fnames:
			image_folder = (test_dir+name+"/image/"+name+".png")
			imagein = (np.array(rgb2gray(io.imread(image_folder))))
			image.append(resize(imagein, (256,256)))
			groundtruths_imp = []
			for file in os.listdir(test_dir + name + "/mask/"):
				groundtruths_in = np.array(io.imread(test_dir + name + "/mask/" + file))
				groundtruths_in = resize(rgb2gray(groundtruths_in), (256, 256))
				groundtruths_imp.append(groundtruths_in)
			groundtruth.append(np.sum(groundtruths_imp, axis=0))
			prediction_folder = results_dir + name + ".npy"
			prediction = np.load(prediction_folder)
			prediction = np.sum(np.where(prediction == 1, 1, 0), axis = -1)
			prediction[prediction > 1] = 1
			predictions.append(resize(prediction, (256,256)))
		logging.info("pred %s" % np.shape(predictions))
		logging.info("gt %s" % np.shape(groundtruth))
		predictions = np.array(predictions)
		groundtruth = np.array(groundtruth)
		abs_errormap_norm, rel_errormap_norm = calcerrormap(predictions, groundtruth)

	else:
		image_files = os.listdir(test_dir + "/image/")
		if max_images is not None:
			image_files = image_files[0: max_images]
		image, image_fnames = import_images(test_dir + "/image/", image_files, None)
		predictions, _ = import_images(results_dir, image_files, "_predict.tif")
		if os.path.isdir(test_dir + "/groundtruth/"):
			groundtruth, _ = import_images(test_dir + "/groundtruth/", image_files, None)
		image, image_fnames = import_images(test_dir + "/image/", image_files, None)
		if os.path.isdir(test_dir + "image2"):
			image2, _ = import_images(test_dir + "image2", image_files, None)
			image1, _ = import_images(test_dir + "image1", image_files, None)
		elif os.path.isdir(test_dir + "image1"):
			image1, _ = import_images(test_dir + "image1", image_files, None)
	if os.path.isdir(root_dir + "uncertainty"): 
		uncertainty, _ = import_images(root_dir + "uncertainty", image_files, "_predict.tif")
	if os.path.isdir(test_dir + "/groundtruth/"):
		abs_errormap_norm, rel_errormap_norm = calcerrormap(predictions, groundtruth)

	os.makedirs(root_dir + "/insights/", exist_ok=True)
	np.save(root_dir + "/insights/" + "image", image)
	np.save(root_dir + "/insights/" + "prediction", predictions)
	if os.path.isdir(test_dir + "/groundtruth/") and len(groundtruth) > 1:
		np.save(root_dir + "/insights/" + "groundtruth", groundtruth)
		np.save(root_dir + "/insights/" + "abs_errormap", abs_errormap_norm)
		np.save(root_dir + "/insights/" + "rel_errormap", rel_errormap_norm)
	np.save(root_dir + "/insights/" + "image_names", image_fnames)
	if os.path.isdir(root_dir + "uncertainty"):
		np.save(root_dir + "/insights/" + "uncertainty", uncertainty/255.)
	if os.path.isdir(test_dir + "image1"):
		logging.info("Two")
		np.save(root_dir + "/insights/" + "image1", image1)
	if os.path.isdir(test_dir + "image2"):
		logging.info("Three")
		np.save(root_dir + "/insights/" + "image1", image1)
		np.save(root_dir + "/insights/" + "image2", image2)
