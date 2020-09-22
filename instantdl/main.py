'''
InstantDL
Written by Dominik Waibel and Ali Boushehri

In this file the functions are started to train and test the networks
'''

import os
import argparse
from instantdl.utils import load_json
from instantdl import GetPipeLine
import logging
from keras import backend as K

def start_learning( use_algorithm,
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

    logging.info("Start learning")
    logging.info(use_algorithm)
    
    
    pipeline = GetPipeLine(use_algorithm,
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
        logging.info("%s : %s \n" % (k,configs[k]))

   
    '''
    Sanity checks in order to ensure all settings in config
    have been set so the programm is able to run
    '''
    assert configs["use_algorithm"] in ['SemanticSegmentation',
                                        'Regression',
                                        'InstanceSegmentation',
                                        'Classification']

    if not isinstance(configs["batchsize"], int):
        logging.warning("Batchsize has not been set. Setting batchsize = 1")
        batchsize = 1
    if not isinstance(configs["iterations_over_dataset"], int):
        logging.warning("Epochs has not been set. Setting epochs = 500 and using early stopping")
        iterations_over_dataset = 500

    if "pretrained_weights" in configs:
        if not os.path.isfile((configs["pretrained_weights"])):
            configs["pretrained_weights"] = None

    start_learning( **configs)