from instantdl.utils import *
from instantdl.classification.classification import Classification
from instantdl.segmentation.Regression import Regression
from instantdl.segmentation.InstanceSegmentation import InstanceSegmentation
from instantdl.segmentation.SemanticSegmentation import SemanticSegmentation

def GetPipeLine(configs):
    if "seeds" in configs:
        if configs["seeds"] == True:
            import numpy as np
            np.random.seed(123)
            import random as python_random
            python_random.seed(123)
            import tensorflow as tf
            tf.random.set_random_seed(123)
            sess = tf.Session(graph=tf.get_default_graph())
            K.set_session(sess)
    if configs["use_algorithm"] == "Classification":
        pipeline = Classification(**configs)
        return pipeline
    elif configs["use_algorithm"] == "Regression":
        pipeline = Regression(**configs)
        return pipeline
    elif configs["use_algorithm"] == "SemanticSegmentation" :
        pipeline = SemanticSegmentation(**configs)
        return pipeline
    elif configs["use_algorithm"] == "InstanceSegmentation":
        pipeline = InstanceSegmentation(**configs)
        return pipeline
    else: 
        raise KeyError("The use_algorithm should be set correctly")