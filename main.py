'''
InstantDL
Written by Dominik Waibel and Ali Boushehri

In this file the functions are started to train and test the networks
'''

from utils import *
from classification.classification import Classification
from segmentation.Regression import Regression

def GetPipeLine(use_algorithm,
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
    
    if use_algorithm == "Classification":
        pipeline = Classification(use_algorithm,
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
        return pipeline
    elif use_algorithm in ["Regression", "SemanticSegmentation"]:
        pipeline = Regression(use_algorithm,
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
    elif use_algorithm == "InstanceSegmentation":
        pipeline = InstanceSegmentation(use_algorithm,
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
    else: 
        print("pipeline is still not ready")
    

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
    
    
    pipeline = GetPipeLine( use_algorithm,
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
                    
    pipeline.run()
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