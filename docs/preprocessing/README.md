# Preprocessing

In this folder, you can find the information regarding the preparation of the experiments for use with jupyter-notebook.

## CreateFolderstrucutre_and_SplitTrainTestset.ipynb

Executing this notebook can help you set up the correct folder structure for semantic segmentation and regression experiments. It automatically creates folders, randomly splits your training and test data with a split of 20 Percent and copies these to the correct folders.

## PrepareDataforMaskRCNN

Executing this notebook can help you set up the correct folder structure for instance  segmentation experiments. It automatically creates folders and copies the data to these based on the folder setup of the semantic segmentation setup.
For the MaskRCNN each object needs to have an individual groundtruth segmentation, therefore segmentation masks from the semantic segmentation are split into indiviudal objects.
