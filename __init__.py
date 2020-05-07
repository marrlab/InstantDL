
from utils import *
from classification.classification import Classification
from segmentation.Regression import Regression
from segmentation.InstanceSegmentation import InstanceSegmentation

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
        return pipeline
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
        return pipeline
    else: 
        raise KeyError("The use_algorithm should be set correctly")