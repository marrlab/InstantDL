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

def start_learning(configs):

    logging.info("Start learning")
    logging.info(configs["use_algorithm"])

    pipeline = GetPipeLine(configs)

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

    if "batchsize" in configs:
        if not isinstance(configs["batchsize"], int):
            logging.warning("Batchsize has not been set. Setting batchsize = 1")
            batchsize = 2
    else:
        logging.warning("Batchsize has not been set. Setting batchsize = 1")
        configs["batchsize"] = 2

    if "iterations_over_dataset" in configs:
        if not isinstance(configs["iterations_over_dataset"], int):
            logging.warning("Epochs has not been set. Setting epochs = 500 and using early stopping")
            iterations_over_dataset = 100

    if "pretrained_weights" in configs:
        if not isinstance(configs["pretrained_weights"], str):
            if not os.path.isfile((configs["pretrained_weights"])):
                configs["pretrained_weights"] = None
    else:
        configs["pretrained_weights"] = None

    start_learning(configs)