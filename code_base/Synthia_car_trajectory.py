import argparse
import os
import sys
# Di Wu add the following really ugly code so that python can find the path
sys.path.append(os.getcwd())
import time
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')

from code_base.config.configuration import Configuration
from code_base.tools.utils import HMS, configurationPATH
from code_base.tools.gt_acquisition import gt_collection_examintion
from code_base.models.PyTorch_model_factory import Model_Factory_LSTM
from code_base.tools.PyTorch_model_training import prepare_data


def process(cf):

    # Create the data generators
    if cf.collect_data:
        print(' ---> Collecting data: ' + cf.sequence_name + ' <---')
        gt_collection_examintion(cf)

    # Build model
    print('\n > Building model...')
    model = Model_Factory_LSTM(cf)
    train_input, train_target, valid_input, valid_target, test_input, test_target, data_mean, data_std = prepare_data(cf)

    if cf.train_model:
        print(' ---> Training data: ' + cf.sequence_name + ' <---')
        for epoch in range(1, cf.n_epochs + 1):
            model.train(train_input, train_target, cf)
            if cf.valid_model:
                model.test(valid_input, valid_target, data_std, data_mean, cf, epoch)
    if cf.test_model:
        model.test(test_input, test_target, data_std, data_mean, cf)
    print(' ---> Finish experiment: ' + cf.exp_name + ' <---')


def main():
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str, default='/home/wzn/PycharmProjects/autonomous_driving/code_base/config/synthia_car_trajectory.py', help='Configuration file')
    arguments = parser.parse_args()
    assert arguments.config_path is not None, 'Please provide a path using -c config/pathname in the command line'
    print('\n > Start Time:')
    print('   ' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    start_time = time.time()
    # Define the user paths

    # Load configuration files
    configuration = Configuration(arguments.config_path)
    cf = configuration.load()
    configurationPATH(cf)

    # Train /test/predict with the network, depending on the configuration
    process(cf)

    # End Time
    end_time = time.time()
    print('\n > End Time:')
    print('   ' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    print('\n   ET: ' + HMS(end_time - start_time))


if __name__ == "__main__":
    main()
