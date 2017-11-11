import argparse
import os
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
# Di Wu add the following really ugly code so that python can find the path
sys.path.append(os.getcwd())
import time
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from code_base.config.configuration import Configuration
from code_base.tools.utils import HMS, configurationPATH
from code_base.tools.gt_acquisition import gt_collection_examintion
from code_base.models.PyTorch_model_factory import Model_Factory_LSTM
from code_base.tools.PyTorch_model_training import prepare_data, prepare_data_image_list


def process(cf):

    # Create the data generators
    if cf.collect_data:
        print(' ---> Collecting data: ' + cf.sequence_name + ' <---')
        gt_collection_examintion(cf)

    # Build model
    train_input, train_target, valid_input, valid_target, test_input, test_target, data_mean, data_std = prepare_data_image_list(cf)

    print('\n > Building model...')
    model = Model_Factory_LSTM(cf)

    if cf.train_model:
        train_losses = []
        valid_losses = []
        print(' ---> Training data: ' + cf.sequence_name + ' <---')
        for epoch in range(1, cf.n_epochs + 1):
            train_losses += [model.train(train_images, train_input, train_target, cf)]
            if cf.valid_model:
                valid_losses += [model.test(valid_images, valid_input, valid_target, data_std, data_mean, cf, epoch)]
        print('---> Train losses:')
        print(train_losses)
        print('---> Valid losses:')
        print(valid_losses)
        # losses figure
        plt.figure()
        plt.title('Losses')
        plt.xlabel('steps')
        plt.ylabel('losses')
        p1 = plt.plot(train_losses, color='b')
        p2 = plt.plot(valid_losses, color='r')
        plt.legend((p1[0], p2[0]), ('trainLosses', 'validLosses'))
        figure_path = os.path.join(model.exp_dir, 'loss_figure.jpg')
        plt.savefig(figure_path)

    if cf.test_model:
        test_loss = model.test(test_images, test_input, test_target, data_std, data_mean, cf)
        print('---> Test losses:')
        print(test_loss)
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
