'''
InstantDL
Written by Dominik Waibel and Ali Boushehri

In this file the functions are started to train and test the networks
'''

from instantdl.utils import *

class InstanceSegmentation(object):
    def __init__(   self,
                    use_algorithm,
                    path,
                    pretrained_weights = None,
                    batchsize = 2,
                    iterations_over_dataset = 100,
                    data_gen_args = None,
                    loss_function = "mse",
                    num_classes = 1,
                    image_size = None,
                    calculate_uncertainty = False,
                    evaluation = True):

        self.use_algorithm = "InstanceSegmentation"
        self.path = path
        self.pretrained_weights = pretrained_weights
        self.batchsize = batchsize
        self.iterations_over_dataset = iterations_over_dataset
        self.loss_function = loss_function
        self.num_classes = num_classes
        self.image_size = image_size
        self.calculate_uncertainty = calculate_uncertainty
        self.evaluation = evaluation
        
        if data_gen_args is None:
            self.data_gen_args = dict()
        else:
            self.data_gen_args = data_gen_args
    


    def run(self):    
        '''
        Initialize a model for instance segmentation
        '''
        UseResnet = 50

        RESULTS_DIR = os.path.join(self.path, "results/")

        image_files = os.listdir(os.path.join(self.path + "/train"))

        '''
        Set values for config file
        '''
        RCNNConfig.NAME = "Image"
        RCNNConfig.classdefs = ("Image", self.num_classes, "Classes")
        RCNNConfig.BACKBONE = str("resnet" + str(UseResnet))
        print("RCNNConfig.BACKBONE", RCNNConfig.BACKBONE)
        RCNNConfig.STEPS_PER_EPOCH = int(len(image_files))
        RCNNConfig.VALIDATION_STEPS = 10
        RCNNConfig.NUM_CLASSES = 1 + self.num_classes # Background + Classes
        RCNNConfig.data_gen_args = self.data_gen_args

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
        if self.pretrained_weights == None:
            print("Weights == False")
            weights = "last"
        else:
            weights = self.pretrained_weights
        dataset = self.path
        logs = os.path.join(self.path, "logs")

        # Configurations
        config = RCNNConfig()
        config.display()

        # Create model
        model = RCNNmodel.MaskRCNN(mode="training", config=config, model_dir=logs)

        # Select weights file to load
        if os.path.isfile(str(self.pretrained_weights)):
            #if os.path.exists(self.pretrained_weights):
            print("Using pretrained weights from path")
            weights_path = weights
        elif self.pretrained_weights == "last":
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
        tensorboard = TensorBoard(log_dir="logs/" + self.path + "/" + format(time.time()))  # , update_freq='batch')
        custom_callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto'), tensorboard]
        if self.iterations_over_dataset > 0:
            print("Start train")
            train(model, dataset, trainsubset, VAL_IMAGE_IDS, self.iterations_over_dataset, custom_callbacks)

        print("Iterations are set to:", self.iterations_over_dataset)
        print("Testing with weights:", weights_path)
        config = RCNNInferenceConfig()
        model = RCNNmodel.MaskRCNN(mode="inference", config=config, model_dir=logs)
        if self.iterations_over_dataset > 0:
            weights_path = model.find_last()
        model.load_weights(weights_path, by_name=True)
        test_Val_IMAGE_IDS = []
        detect(model, dataset, testsubset, RESULTS_DIR, test_Val_IMAGE_IDS)
        if self.evaluation == True:
            segmentation_regression_evaluation(self.path)

        model = None
