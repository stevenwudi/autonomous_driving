import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
# Di Wu add the following really ugly code so that python can find the path
sys.path.append(os.getcwd())
import time
from datetime import datetime

from code_base.tools.PyTorch_data_generator import Dataset_Generators_Cityscape
from code_base.models.PyTorch_model_factory import Model_Factory
from code_base.config.configuration import Configuration
from code_base.utils import HMS, configurationPATH


# Train the network
def process(cf):

    print(' ---> Init experiment: ' + cf.exp_name + ' <---')
    # Create the data generators
    DG = Dataset_Generators_Cityscape(cf)
    #show_DG(DG, 'train')  # this script will draw an image

    # Build model
    print('\n > Building model...')
    model = Model_Factory(cf)

    # model.test(DG.val_loader, 0)
    if cf.train_model:
        for epoch in range(1, cf.n_epochs + 1):
            model.train(DG.train_loader, epoch)
            if epoch % 30 == 0:
                if cf.test_model:
                    model.test(DG.val_loader, epoch)

    # Finish
    print(' ---> Finish experiment: ' + cf.exp_name + ' <---')


# Main function
def main():
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Semantic segmentation')
    parser.add_argument('-c', '--config_path', type=str,
                        default='/home/ty/code/autonomous_driving/code_base/config/cityscapes_segmentation.py', help='Configuration file')
    parser.add_argument('-e', '--exp_name', type=str,
                        default='cityscape_segmentation', help='Name of the experiment')
    parser.add_argument('-s', '--shared_path', type=str,
                        default='/home/public', help='Path to shared data folder')
    parser.add_argument('-l', '--local_path', type=str,
                        default='/home/ty/code/autonomous_driving', help='Path to local data folder')

    arguments = parser.parse_args()
    assert arguments.config_path is not None, 'Please provide a configuration path using -c config/pathname'
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
    configurationPATH(cf)

    # Train /test/predict with the network, depending on the configuration
    process(cf)

    # End Time
    end_time = time.time()
    print('\n > End Time:')
    print('   ' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    print('\n   ET: ' + HMS(end_time - start_time))  # -> H:M:S


# Entry point of the script
if __name__ == "__main__":
    main()
