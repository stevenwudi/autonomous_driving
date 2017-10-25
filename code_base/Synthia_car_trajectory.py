import argparse
import os
import sys

# Di Wu add the following really ugly code so that python can find the path
sys.path.append(os.getcwd())
import time
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')

from code_base.tools.PyTorch_data_generator_car_trajectory import Dataset_Generators_Synthia_Car_trajectory
from code_base.config.configuration import Configuration
from code_base.utils import HMS, configurationPATH
from code_base.tools.gt_acquisition import get_ground_truth_sequence_car_trajectory, draw_selected_gt_car_trajectory


def process(cf):
    # Create the data generators
    cf.batch_size_train = 1  # because we want to read every single image sequentially
    dataset_path_list = cf.dataset_path
    processed_list = os.listdir(cf.savepath)
    for dataset_path in dataset_path_list:
        sequence_name = dataset_path.split('/')[-1]
        cf.dataset_path = dataset_path
        DG = Dataset_Generators_Synthia_Car_trajectory(cf)
        #if sequence_name+'.json' not in processed_list:
        if False:
            get_ground_truth_sequence_car_trajectory(DG, cf, sequence_name)
        elif sequence_name == 'SYNTHIA-SEQS-06-SUMMER' and True:
            draw_selected_gt_car_trajectory(DG, cf, sequence_name)
    # Build model
    print('\n > Building model...')

    # Finish
    print(' ---> Finish experiment: ' + cf.exp_name + ' <---')


def main():
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str, default='/home/stevenwudi/PycharmProjects/autonomous_driving/code_base/config/synthia_car_trajectory.py', help='Configuration file')
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
