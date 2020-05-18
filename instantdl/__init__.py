
from instantdl.utils import *
from instantdl.classification.classification import Classification
from instantdl.segmentation.Regression import Regression
from instantdl.segmentation.InstanceSegmentation import InstanceSegmentation
from instantdl.segmentation.SemanticSegmentation import SemanticSegmentation

def GetPipeLine(use_algorithm,
                    path, 
                    pretrained_weights, 
                    batchsize, 
                    iterations_over_dataset, 
                    data_gen_args, 
                    loss_function, 
                    num_classes, 
                    image_size, 
                    calculate_uncertainty,
                    evaluation):
    
    if use_algorithm == "Classification":
        pipeline = Classification(use_algorithm,
                        path, 
                        pretrained_weights, 
                        batchsize, 
                        iterations_over_dataset, 
                        data_gen_args, 
                        loss_function, 
                        num_classes, 
                        image_size, 
                        calculate_uncertainty,
                        evaluation)
        return pipeline
    elif use_algorithm == "Regression":
        pipeline = Regression(use_algorithm,
                        path, 
                        pretrained_weights, 
                        batchsize, 
                        iterations_over_dataset, 
                        data_gen_args, 
                        loss_function, 
                        num_classes, 
                        image_size, 
                        calculate_uncertainty,
                        evaluation)
        return pipeline
    elif use_algorithm == "SemanticSegmentation" :
        pipeline = SemanticSegmentation(use_algorithm,
                        path, 
                        pretrained_weights, 
                        batchsize, 
                        iterations_over_dataset, 
                        data_gen_args, 
                        loss_function, 
                        num_classes, 
                        image_size, 
                        calculate_uncertainty,
                        evaluation)
        return pipeline
    elif use_algorithm == "InstanceSegmentation":
        pipeline = InstanceSegmentation(use_algorithm,
                        path, 
                        pretrained_weights, 
                        batchsize, 
                        iterations_over_dataset, 
                        data_gen_args, 
                        loss_function, 
                        num_classes, 
                        image_size, 
                        calculate_uncertainty,
                        evaluation)
        return pipeline
    else: 
        raise KeyError("The use_algorithm should be set correctly")