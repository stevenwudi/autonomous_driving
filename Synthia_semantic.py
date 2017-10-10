#!/usr/bin/env python
import argparse
import os
import sys
from getpass import getuser
from matplotlib import pyplot as plt
import time
from datetime import datetime
#matplotlib.use('Agg')  # Faster plot

# Import tools
from code_base.config.configuration import Configuration
from code_base.tools.logger import Logger
from code_base.tools.PyTorch_data_generator import Dataset_Generators
#from code_base.callbacks.callbacks_factory import Callbacks_Factory
from code_base.utils import HMS, configurationPATH, show_DG
#from code_base.models.model_factory import Model_Factory
from code_base.models.PyTorch_fcn import FeatureResNet, SegResNet
from torchvision import models
from torch import nn

import argparse

import os
import sys
# Di Wu add the following really ugly code so that python can find the path
sys.path.append(os.getcwd())
import time
from datetime import datetime

from code_base.tools.PyTorch_data_generator import Dataset_Generators_Cityscape
from code_base.models.PyTorch_model_factory import Model_Factory
from code_base.config.configuration import Configuration
from code_base.utils import HMS, configurationPATH



def process(cf):
    # Enable log file
    sys.stdout = Logger(cf.log_file)
    print(' ---> Init experiment: ' + cf.exp_name + ' <---')

    # Create the data generators
    DG = Dataset_Generators(cf)
    show_DG(DG)  # this script will draw an image

    # Build model
    print('\n > Building model...')
    pretrained_net = FeatureResNet()
    pretrained_net.load_state_dict(models.resnet34(pretrained=True).state_dict())
    net = SegResNet(cf.n_classes, pretrained_net).cuda()
    crit = nn.BCELoss().cuda()


    # Create the callbacks
    print('\n > Creating callbacks...')
    cb = Callbacks_Factory().make(cf, valid_gen)

    if cf.train_model:
        # Train the model
        model.train(train_gen, valid_gen, cb)

    if cf.test_model:
        # Compute validation metrics
        model.test(valid_gen)
        # Compute test metrics
        model.test(test_gen)

    if cf.pred_model:
        # Compute validation metrics
        model.predict(valid_gen, tag='pred')
        # Compute test metrics
        model.predict(test_gen, tag='pred')

    # Finish
    print(' ---> Finish experiment: ' + cf.exp_name + ' <---')


def main():
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str, default='/home/stevenwudi/PycharmProjects/autonomous_driving/code_base/config/synthia_segmentation.py', help='Configuration file')
    parser.add_argument('-e', '--exp_name', type=str, default='synthia_segmentation', help='Name of the experiment')
    parser.add_argument('-s', '--shared_path', type=str, default='/home/public', help='Path to shared data folder')
    parser.add_argument('-l', '--local_path', type=str, default='/home/stevenwudi/PycharmProjects/autonomous_driving', help='Path to local data folder')
    arguments = parser.parse_args()
    assert arguments.config_path is not None, 'Please provide a path using -c config/pathname in the command line'
    assert arguments.exp_name is not None, 'Please provide a name for the experiment using -e name in the command line'
    # Start Time
    print('\n > Start Time:')
    print('   ' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    start_time = time.time()
    # Define the user paths
    shared_path = arguments.shared_path
    local_path = arguments.local_path
    dataset_path = os.path.join(local_path, 'Datasets')
    shared_dataset_path = os.path.join(shared_path)
    experiments_path = os.path.join(local_path, 'Experiments')
    shared_experiments_path = os.path.join(shared_path, 'Experiments')
    # Load configuration files
    configuration = Configuration(arguments.config_path, arguments.exp_name,
                                  dataset_path, shared_dataset_path,
                                  experiments_path, shared_experiments_path)

    cf = configuration.load()
    configurationPATH(cf, dataset_path)

    # Train /test/predict with the network, depending on the configuration
    process(cf)

    # End Time
    end_time = time.time()
    print('\n > End Time:')
    print('   ' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    print('\n   ET: ' + HMS(end_time - start_time))


if __name__ == "__main__":
    main()
