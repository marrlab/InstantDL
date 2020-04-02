'''
InstantDL
Written by Dominik Waibel and Ali Boushehri

In this file the functions are started to train and test the networks
'''

from utils import *

def start_learning( use_algorithm,
                    path, 
                    pretrained_weights, 
                    batchsize, 
                    Iterations_Over_Dataset, 
                    data_gen_args, 
                    loss_function, 
                    num_classes, 
                    Image_size, 
                    calculate_uncertainty,
                    evaluation):

    print("Start learning")
    print(use_algorithm)
    if use_algorithm == "Regression" or use_algorithm == "Classification" or use_algorithm == "SemanticSegmentation":

        '''
        Build the models for Regression, Segmentaiton and Classification
        '''
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
        if use_algorithm == "Regression" or use_algorithm == "SemanticSegmentation":
            '''
            Get Output size of U-Net
            '''

            img_file_label_name = os.listdir(data_path + "/groundtruth/")[0]
            print("img_file_label_name", img_file_label_name)
            Training_Input_shape_label = np.shape(np.array(import_image(data_path + "/groundtruth/" + img_file_label_name)))
            num_channels_label = Training_Input_shape_label[-1]
            if all([num_channels_label != 1, num_channels_label != 3]):
                num_channels_label = 1

            if use_algorithm == "SemanticSegmentation":
                data_gen_args["binarize_mask"] = True

            TrainingDataGenerator = training_data_generator(Training_Input_shape, batchsize, num_channels, num_channels_label, train_image_files, data_gen_args, data_dimensions, data_path, use_algorithm)
            ValidationDataGenerator = training_data_generator(Training_Input_shape, batchsize, num_channels, num_channels_label, val_image_files, data_gen_args, data_dimensions, data_path, use_algorithm)

            '''
            Build a 2D or 3D U-Net model and initialize it with pretrained or random weights
            '''
            if pretrained_weights == False:
                pretrained_weights = None
            if data_dimensions == 3:
                print("Using 3D UNet")
                model = UNetBuilder.unet3D(pretrained_weights, network_input_size, num_channels_label,num_classes, loss_function, Dropout_On = True)
            else:
                print("Using 2D UNet")
                model = UNetBuilder.unet2D(pretrained_weights,network_input_size, num_channels_label,num_classes, loss_function, Dropout_On = True)


        if use_algorithm == "Classification":
            '''
            Build the classificaiton model with a ResNet50 and initilize with pretrained, imagenet or random weights
            Initialize data generator 
            '''
            if not isinstance(num_classes, int):
                sys.exit("Number of classes has not been set. You net to set num_classes!")
            TrainingDataGenerator = training_data_generator_classification(Training_Input_shape,batchsize, num_channels, num_classes, train_image_files, data_gen_args, data_path, use_algorithm)
            ValidationDataGenerator = training_data_generator_classification(Training_Input_shape,batchsize, num_channels, num_classes, train_image_files, data_gen_args, data_path, use_algorithm)

            model = ResNet50(network_input_size, Dropout = 0.1,include_top=True, weights=pretrained_weights, input_tensor=None, pooling='max', classes=num_classes)
            if (pretrained_weights):
                model.load_weights(pretrained_weights, by_name=True, skip_mismatch=True)
            else:
                print("No weigths given")
                #weights_path = get_imagenet_weights()
                #model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            model.compile(loss=loss_function,
                          optimizer='Adam',
                          metrics=['accuracy'])

        model.summary()

        '''
        Set Model callbacks such as: 
        - Early stopping (after the validation loss has not improved for 25 epochs
        - Checkpoints: Save model after each epoch if the validation loss has improved 
        - Tensorboard: Monitor training live with tensorboard. Start tensorboard in terminal with: tensorboard --logdir=/path_to/logs 
        '''
        Early_Stopping = EarlyStopping(monitor='val_loss', patience=25, mode='auto', verbose=0)
        datasetname = path.rsplit("/",1)[1]
        checkpoint_filepath = (path + "/logs" + "/pretrained_weights_" + datasetname + ".hdf5") #.{epoch:02d}.hdf5")
        os.makedirs("./" + (path + "/logs"), exist_ok=True)
        model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor=('val_loss'), verbose=1, save_best_only=True)

        tensorboard = TensorBoard(log_dir="logs/" + path + "/" + format(time.time())) #, update_freq='batch')

        if use_algorithm == "Regression" or use_algorithm == "SemanticSegmentation":
            callbacks_list = [model_checkpoint, tensorboard, Early_Stopping]
        if use_algorithm == "Classification":
            callbacks_list = [model_checkpoint, tensorboard, Early_Stopping]

        '''
        Train the model given the initialized model and the data from the data generator
        '''
        model.fit_generator(TrainingDataGenerator,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=ValidationDataGenerator,
                            validation_steps=2,
                            max_queue_size=50,
                            epochs=epochs,
                            callbacks = callbacks_list,
                            use_multiprocessing=True)
        print('finished Model.fit_generator')

        '''
        Get the names of the test images for model evaluation
        '''
        test_image_files = os.listdir(os.path.join(path + "/test/image"))
        num_test_img = int(len(os.listdir(path + "/test/image")))
        print("Testing on", num_test_img, "test files")

        '''
        Initialize the testset generator
        '''
        testGene = testGenerator(Training_Input_shape, path, num_channels, test_image_files, use_algorithm)
        print('finished testGene')
        results = model.predict_generator(testGene, steps=num_test_img, use_multiprocessing=False, verbose=1)
        print("results", np.shape(results))
        print('finished model.predict_generator')

        if use_algorithm == "Regression" or use_algorithm == "SemanticSegmentation":
            '''
            Save the models prediction on the testset by printing the predictions as images to the results folder in the project path
            '''
            saveResult(path + "/results/", test_image_files, results, Input_image_shape)
            if calculate_uncertainty == False:
                if evaluation == True:
                    segmentation_regression_evaluation(path)
        elif use_algorithm == "Classification":
            '''
            Save the models prediction on the testset by saving a .csv file containing filenames and predicted classes to the results folder in the project path
            '''
            saveResult_classification(path, test_image_files, results)
            if evaluation == True:
                classification_evaluation(path)

        if calculate_uncertainty == True:
            if use_algorithm is "Regression" or use_algorithm is "SemanticSegmentation":
                '''
                 Start uncertainty prediction if selected for regression or semantic segmentation
                 As suggested by Gal et. al.: https://arxiv.org/abs/1506.02142 
                 And as implemented in: https://openreview.net/pdf?id=Sk_P2Q9sG
                 '''
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
                if evaluation == True:
                   segmentation_regression_evaluation(path)

        if calculate_uncertainty == True:
            if use_algorithm is "Classification":
                '''
                Uncertainty prediction for classification 
                Using: https://github.com/RobRomijnders/bayes_nn
                for uncertainty estimation with MC Dropout for classification
                '''
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
                if evaluation == True:
                    classification_evaluation(path)
    if use_algorithm == "InstanceSegmentation":
        '''
        Initialize a model for instance segmentation 
        '''
        UseResnet = 50

        RESULTS_DIR = os.path.join(path, "results/")

        image_files = os.listdir(os.path.join(path + "/train"))

        '''
        Set values for config file
        '''
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
        '''
        Initialize a model for instance segmentation with pretrained weights or imagenet weights
        '''
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
            #if os.path.exists(pretrained_weights):
            print("Using pretrained weights from path")
            weights_path = weights
        elif pretrained_weights == "last":
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
        if evaluation == True:
            segmentation_regression_evaluation(path)
    model = None
    K.clear_session()


if __name__ == "__main__":

    parser = argparse.ArgumentParser( \
                            description='Starting the deep learning code')
    parser.add_argument('-c',\
                        '--config', \
                        default="config.json", \
                        help='config json file address', \
                        type=str)

    args = vars(parser.parse_args())

    configs = load_json(args['config'])

    for k in configs:
        print("%s : %s \n" % (k,configs[k]))

    use_algorithm = configs["use_algorithm"]
    path = configs["path"]
    pretrained_weights_path = configs["pretrained_weights_path"]
    batchsize = configs["batchsize"]
    Iterations_Over_Dataset = configs["Iterations_Over_Dataset"]
    data_gen_args = configs["data_gen_args"]
    loss_function = configs["loss_function"]
    num_classes = configs["num_classes"]
    Image_size = configs["Image_size"]
    calculate_uncertainty = configs["calculate_uncertainty"]
    evaluation = configs["evaluation"]
    '''
    Sanity checks in order to ensure all settings in config
    have been set so the programm is able to run
    '''
    assert use_algorithm in ['SemanticSegmentation',
                            'Regression',
                            'InstanceSegmentation',
                            'Classification']

    if not isinstance(batchsize, int):
        warnings.warn("Batchsize has not been set. Setting batchsize = 1")
        batchsize = 1
    if not isinstance(Iterations_Over_Dataset, int):
        warnings.warn("Epochs has not been set. Setting epochs = 500 and using early stopping")
        Iterations_Over_Dataset = 500

    if os.path.isfile((pretrained_weights_path)):
        pretrained_weights = (pretrained_weights_path)
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
                    calculate_uncertainty,
                    evaluation)