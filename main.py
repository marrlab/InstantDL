'''
Author: Dominik Waibel

In this file the funcitions are started to train and test the networks
'''
import argparse
import os
import json
from data_generator.custom_data_generator import *
from segmentation.UNet_models import UNetBuilder
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import time
import tensorflow as tf
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from segmentation.ResNet50 import ResNet50
from segmentation.RCNNSettings import RCNNInferenceConfig, train, detect
import segmentation.RCNNmodel as RCNNmodel
from segmentation.RCNNSettings import RCNNConfig
tf.config.experimental.list_physical_devices('GPU')
from classification.ResNet_models import define_ResNetModel, get_imagenet_weights
import glob
import sys
from keras.optimizers import Adam, SGD
from metrics import accuracy_score
from data_generator.data import write_logbook

def load_json(file_path):
    with open(file_path, 'r') as stream:    
        return json.load(stream)


def start_learning( use_algorithm, 
                    path, 
                    pretrained_weights, 
                    batchsize, 
                    Iterations_Over_Dataset, 
                    data_gen_args, 
                    loss_function, 
                    num_classes, 
                    Image_size, 
                    calculate_uncertainty):


    '''
    Build the models for Regression, Segmentaiton and Classification
    '''
    if use_algorithm is "Regression" or use_algorithm is "Classification" or use_algorithm is "Segmentation":
        '''
            Get the number of input images and their shape
            If the last image dimension,. which should contain the channel information (1 or 3) is not existing e.g. for (512,512) add a 1 as the channel number.
            '''
        if Image_size == None:
            Training_Input_shape, num_channels, Input_image_shape = get_input_image_sizes(path, use_algorithm)
        else:
            Training_Input_shape = Image_size
            num_channels = int(Image_size[-1])
            data_path = path + '/train'
            img_file = os.listdir(data_path + "/image/")[0]
            Input_image_shape = np.array(np.shape(np.array(import_image(data_path + "/image/" + img_file))))

        ''' 
        Check if the 2D or 3D Pipeline is needed
        '''
        if len(Training_Input_shape[:-1]) == 3:
            data_dimensions = 3
        if len(Training_Input_shape[:-1]) == 2:
            data_dimensions = 2

        print("Image dimensions are: ", data_dimensions, "D")

        Folders = ["image", "image1", "image2", "image3", "image4", "image5", "image6", "image7"]
        number_input_images = len([element for element in os.listdir(path + "/train/") if element in Folders])
        network_input_size = np.array(Training_Input_shape)
        network_input_size[-1] = int(Training_Input_shape[-1]) * number_input_images
        network_input_size = tuple(network_input_size)
        print("Number of input folders is: ", number_input_images)
        print("UNet input shape", network_input_size)

        '''
        Import filenames and split them into train and validation set according to the variable -validation_split = 20%
        '''
        data_path = path + '/train'
        train_image_files, val_image_files = training_validation_data_split(data_path)

        steps_per_epoch = int(len(train_image_files)/batchsize)

        epochs = Iterations_Over_Dataset
        print("Making:", steps_per_epoch, "steps per Epoch")

        '''
        Prepare data in Training and Validation set 
        '''
        if use_algorithm == "Regression" or use_algorithm == "Segmentation":
            '''
            Get Output size of U-Net
            '''

            img_file_label_name = os.listdir(data_path + "/groundtruth/")[0]
            print("img_file_label_name", img_file_label_name)
            Training_Input_shape_label = np.shape(np.array(import_image(data_path + "/groundtruth/" + img_file_label_name)))
            num_channels_label = Training_Input_shape_label[-1]
            if all([num_channels_label != 1, num_channels_label != 3]):
                num_channels_label = 1

            if use_algorithm == "Segmentation":
                data_gen_args["binarize_mask"] = True

            TrainingDataGenerator = training_data_generator(Training_Input_shape, batchsize, num_channels, num_channels_label, train_image_files, data_gen_args, data_dimensions, data_path, use_algorithm)
            ValidationDataGenerator = training_data_generator(Training_Input_shape, batchsize, num_channels, num_channels_label, val_image_files, data_gen_args, data_dimensions, data_path, use_algorithm)

            '''
            Build a 2D or 3D U-Net model
            '''

            if pretrained_weights == False:
                pretrained_weights = None
            if data_dimensions == 3:
                print("Using 3D UNet")
                model = UNetBuilder.unet3D(pretrained_weights, network_input_size, num_channels_label,num_classes, loss_function, Dropout_On = True)
            else:
                print("Using 2D UNet")
                model = UNetBuilder.unet2D(pretrained_weights,network_input_size, num_channels_label,num_classes, loss_function, Dropout_On = True)

            '''
            Build a classificaiton model
            '''
        if use_algorithm == "Classification":
            if not isinstance(num_classes, int):
                sys.exit("Number of classes has not been set. You net to set num_classes!")
            TrainingDataGenerator = training_data_generator_classification(Training_Input_shape,num_channels, batchsize, num_classes, train_image_files, data_gen_args, data_path, use_algorithm)
            ValidationDataGenerator = training_data_generator_classification(Training_Input_shape,num_channels, batchsize, num_classes, train_image_files, data_gen_args, data_path, use_algorithm)

            model = ResNet50(network_input_size, Dropout = 0.1,include_top=True, weights=pretrained_weights, input_tensor=None, pooling='max', classes=num_classes)
            if (pretrained_weights):
                model.load_weights(pretrained_weights, by_name=True, skip_mismatch=True)
            else:
                print("No weigths given: Using imagenet weights")
                weights_path = get_imagenet_weights()
                model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            model.compile(loss=loss_function,
                          optimizer='Adam',
                          metrics=['accuracy'])

        model.summary()

        # Train Segmentation and Regression and Classification
        Early_Stopping = EarlyStopping(monitor='val_loss', patience=25, mode='auto', verbose=0)
        datasetname = path.rsplit("/",1)[1]
        checkpoint_filepath = (path + "/logs" + "/pretrained_weights_" + datasetname + ".hdf5") #.{epoch:02d}.hdf5")
        os.makedirs("./" + (path + "/logs"), exist_ok=True)
        model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor=('val_loss'), verbose=1, save_best_only=True)

        tensorboard = TensorBoard(log_dir="logs/" + path + "/" + format(time.time())) #, update_freq='batch')

        if use_algorithm == "Regression" or use_algorithm == "Segmentation":
            callbacks_list = [model_checkpoint, tensorboard, Early_Stopping]
        if use_algorithm == "Classification":
            callbacks_list = [model_checkpoint, tensorboard, Early_Stopping]

        model.fit_generator(TrainingDataGenerator,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=ValidationDataGenerator,
                            validation_steps=2,
                            max_queue_size=50,
                            epochs=epochs,
                            callbacks = callbacks_list,
                            use_multiprocessing=False)
        print('finished Model.fit_generator')


        test_image_files = os.listdir(os.path.join(path + "/test/image"))

        num_test_img = int(len(os.listdir(path + "/test/image")))
        print("Testing on", num_test_img, "test files")
        testGene = testGenerator(Training_Input_shape, path, num_channels, test_image_files, use_algorithm)
        print('finished testGene')
        results = model.predict_generator(testGene, steps=num_test_img, use_multiprocessing=False, verbose=1)
        print("results", np.shape(results))
        print('finished model.predict_generator')
        print("Starting Uncertanty estimation")
        '''Uncertainty prediction
        As suggested by Gal et. al.: https://arxiv.org/abs/1506.02142 
        And as implemented in: https://openreview.net/pdf?id=Sk_P2Q9sG
        '''
        if use_algorithm is "Regression" or use_algorithm is "Segmentation":
            saveResult(path + "/results/", test_image_files, results, Input_image_shape)

        if calculate_uncertainty == True:
            if data_dimensions == 3:
                print("Using 3D UNet")
                if epochs > 0:
                    pretrained_weights = checkpoint_filepath
                model = UNetBuilder.unet3D(pretrained_weights, network_input_size, num_channels_label, num_classes, loss_function, Dropout_On=True)
            else:
                print("Using 2D UNet")
                if epochs > 0:
                    pretrained_weights = checkpoint_filepath
                model = UNetBuilder.unet2D(pretrained_weights,network_input_size, num_channels_label, loss_function, Dropout_On = True)
            resultsMCD = []
            for i in range(0, 20):
                testGene = testGenerator(Training_Input_shape, path, num_channels, test_image_files, use_algorithm)
                resultsMCD.append(model.predict_generator(testGene, steps=num_test_img, use_multiprocessing=False, verbose=1))
            resultsMCD = np.array(resultsMCD)
            aleatoric_uncertainty = np.mean(resultsMCD * (1 - resultsMCD), axis = 0)
            epistemic_uncertainty = np.mean(resultsMCD**2, axis = 0) - np.mean(resultsMCD, axis = 0)**2
            #combined_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
            combined_uncertainty = epistemic_uncertainty
            '''Threshold uncertainty to make the image easier understandable'''
            #combined_uncertainty[combined_uncertainty < np.mean(combined_uncertainty)] = 0
            saveResult(path + "/uncertainty/", test_image_files, combined_uncertainty, Input_image_shape)

        if use_algorithm is "Classification":
            '''Uncertainty prediction
            Using: https://github.com/RobRomijnders/bayes_nn
            for uncertainty estimation with MC Dropout for classification
            '''
            saveResult_classification(path, test_image_files, results)
        if calculate_uncertainty == True:
            if epochs > 0:
                pretrained_weights = checkpoint_filepath
            model = ResNet50(network_input_size, Dropout = 0.5, include_top=True, weights=pretrained_weights, input_tensor=None, pooling='max', classes=num_classes)
            print("Starting Uncertainty estimation")
            resultsMCD = []
            for i in range(0, 20):
                print("Testing Uncertainty Number: ", str(i))
                testGene = testGenerator(Training_Input_shape, path, num_channels, test_image_files, use_algorithm)
                resultsMCD_pred = model.predict_generator(testGene, steps=num_test_img, use_multiprocessing=False, verbose=1)
                resultsMCD.append(resultsMCD_pred)
            resultsMCD = np.array(resultsMCD)
            argmax_MC_Pred = (np.argmax(resultsMCD, axis=-1))
            average_MC_Pred = []
            for i in range(len(argmax_MC_Pred[1])):
                bincount = np.bincount(argmax_MC_Pred[:,i])
                average_MC_Pred.append(np.argmax(bincount))
            average_MC_Pred = np.array(average_MC_Pred)
            combined_certainty = np.mean(-1 * np.sum(resultsMCD * np.log(resultsMCD + 10e-6), axis=0), axis = 1)
            combined_certainty /= np.log(20) # normalize to values between 0 and 1
            saveResult_classification_uncertainty(path, test_image_files, results, average_MC_Pred, combined_certainty)

    if use_algorithm is "ObjectDetection":

        UseResnet = 50

        RESULTS_DIR = os.path.join(path, "results/")

        image_files = os.listdir(os.path.join(path + "/train"))

        '''Set values for config file'''
        RCNNConfig.NAME = "Image"
        RCNNConfig.classdefs = ("Image", num_classes, "Classes")
        RCNNConfig.BACKBONE = str("resnet" + str(UseResnet))
        print("RCNNConfig.BACKBONE", RCNNConfig.BACKBONE)
        RCNNConfig.STEPS_PER_EPOCH = int(len(image_files))
        RCNNConfig.VALIDATION_STEPS = 1
        RCNNConfig.NUM_CLASSES = 1 + num_classes # Background + Classes
        RCNNConfig.data_gen_args = data_gen_args

        lenval = len(image_files) * 0.2
        validation_spilt_id = np.array(list(range(0, len(image_files), int(len(image_files) / lenval))))
        print(validation_spilt_id)
        VAL_IMAGE_IDS = []
        for i in range(0, len(image_files)):
            if i in validation_spilt_id:
                VAL_IMAGE_IDS.append(image_files[i])

        trainsubset = "train"
        testsubset = "test"

        if pretrained_weights == None:
            print("Weights == False")
            weights = "last"
        else:
            weights = pretrained_weights
        dataset = path
        logs = os.path.join(path, "logs")

        # Configurations
        config = RCNNConfig()
        config.display()

        # Create model
        model = RCNNmodel.MaskRCNN(mode="training", config=config, model_dir=logs)

        # Select weights file to load
        if os.path.isfile(str(pretrained_weights)):
            if pretrained_weights != None:
            #if os.path.exists(pretrained_weights):
                print("Using pretrained weights from path")
                weights_path = weights
            elif "h5" in pretrained_weights or pretrained_weights == "last":
                if os.path.exists(model.find_last()):
                    print("Using last weights")
                    weights_path = model.find_last()
        else:
            print("No weigths given: Using imagenet weights")
            weights_path = model.get_imagenet_weights()

        # Load weights
        print("Loading weights ", weights_path)
        if weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)
        tensorboard = TensorBoard(log_dir="logs/" + path + "/" + format(time.time()))  # , update_freq='batch')
        custom_callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto'), tensorboard]
        if Iterations_Over_Dataset > 1:
            print("Start train")
            train(model, dataset, trainsubset, VAL_IMAGE_IDS, Iterations_Over_Dataset+1, custom_callbacks)


        print("Testing with weights:", weights_path)
        model.load_weights(weights_path, by_name=True)
        config = RCNNInferenceConfig()
        model = RCNNmodel.MaskRCNN(mode="inference", config=config, model_dir=logs)
        weights_path = model.find_last()
        model.load_weights(weights_path, by_name=True)
        test_Val_IMAGE_IDS = []
        detect(model, dataset, testsubset, RESULTS_DIR, test_Val_IMAGE_IDS)

    model = None
    K.clear_session()


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser( \
                            description='Starting the deep learning code')
    parser.add_argument('-c',\
                        '--config', \
                        help='config json file address', \ 
                        type=str)

    args = vars(parser.parse_args())
    
    configs = load_json(args['config'])

    for k in configs:
        print("%s : %s \n" % (k,configs[k]))
    
    use_algorithm = configs["use_algorithm"]
    path = configs["path"]  
    pretrained_weights = configs["pretrained_weights"] 
    batchsize = configs["batchsize"] 
    Iterations_Over_Dataset = configs["Iterations_Over_Dataset"]  
    data_gen_args = configs["data_gen_args"] 
    loss_function = configs["loss_function"]  
    num_classes = configs["num_classes"]  
    Image_size = configs["Image_size"] 
    calculate_uncertainty = configs["calculate_uncertainty"]

    '''
    Sanity checks in order to ensure all settings in config
    have been set so the programm is able to run
    '''
    assert use_algorithm in ['Segmentation', 
                            'Regression', 
                            'ObjectDetection', 
                            'Classification']
        
    if not isinstance(batchsize, int):
        warnings.warn("Batchsize has not been set. Setting batchsize = 1")
        batchsize = 1
    if not isinstance(Iterations_Over_Dataset, int):
        warnings.warn("Epochs has not been set. Setting epochs = 500 and using early stopping")
        Iterations_Over_Dataset = 500

    if use_pretrained_weights == True:
        pretrained_weights = (path + "/logs/pretrained_weights_CPC25.hdf5")
    else:
        pretrained_weights = None

 
    start_learning( use_algorithm, 
                    path, 
                    pretrained_weights, 
                    batchsize, 
                    Iterations_Over_Dataset, 
                    data_gen_args, 
                    loss_function, 
                    num_classes, 
                    Image_size, 
                    calculate_uncertainty)         