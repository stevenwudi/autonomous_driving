#!/usr/bin/env python
import argparse
import os
import sys
# Di Wu add the following really ugly code so that python can find the path
sys.path.append(os.getcwd())
import time
from datetime import datetime

from code_base.tools.PyTorch_data_generator import Dataset_Generators_Cityscape
from code_base.models.PyTorch_model_factory import Model_Factory
from code_base.tools.PyTorch_data_generator import Dataset_Generators_Synthia
from code_base.config.configuration import Configuration
from code_base.utils import show_DG, HMS, configurationPATH


def process(cf):
    # Create the data generators
    DG = Dataset_Generators_Synthia(cf)
    show_DG(DG)  # this script will draw an image

    # Build model
    print('\n > Building model...')

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
