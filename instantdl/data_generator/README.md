# Data Generator

- The pipeline automatically imports the data using the data generator from the corresponding folders and prepares for training and testing. This includes the following steps:
- Import of data for all common image file formats processable with scikit-image (e.g. .jpg, .tiff, .png) and and .npy files.
- Initialization of model with given weights or random initialization. For classification and instance segmentation it will automatically use imagenet weights if no weights are given.
- Split of data from ‘train’-folder into training and validation set with a 20% split and random shuffle of training data
- Normalization of data to the range between 0 and 1 based on the datasets minimum and maximum pixel value.
- The data will be re-normalized when saved after testing.
- Batch creation and data augmentation on the fly
- Training for the set epoch length using with early stopping of training if the validation accuracy has not improved for the last epochs
- Real time monitoring of training with Tensorboard and in the users IDE or terminal
- Saving of the best models during training
- Automated evaluation of the trained model on the test-data and saving of predicted labels
- For regression, semantic segmentation or classification it will calculate the uncertainty using Monte Carlo Dropout by creating 20 different models and saving the uncertainty estimation to the project directory
- Experiment settings are automatically saved to a logbook in order to simplify experiment monitoring
