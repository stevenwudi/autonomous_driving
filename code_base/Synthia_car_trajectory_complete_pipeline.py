import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
# Di Wu add the following really ugly code so that python can find the path
sys.path.append(os.getcwd())
import time
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')

from code_base.config.configuration import Configuration
from code_base.tools.utils import HMS, configurationPATH
from code_base.tools.gt_acquisition import car_detection, car_tracking


def process(cf):

    if cf.car_detection:
        print(' ---> Detecting cars : ' + cf.sequence_name + ' <---')
        car_detection(cf)

    if cf.car_tracking:
        print(' ---> Tracking cars : ' + cf.sequence_name + ' <---')
        print('Note that This part requires car detection to run first for very frame')
        car_tracking(cf)

    print(' <--- Experiment finished! --->')


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
