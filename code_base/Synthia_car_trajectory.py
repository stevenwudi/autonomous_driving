import argparse
import os
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
from code_base.tools.PyTorch_data_generator import DataGenerator_Synthia_car_trajectory


def process(cf):

    # Create the data generators
    if cf.collect_data:
        print(' ---> Collecting data: ' + cf.sequence_name + ' <---')
        gt_collection_examintion(cf)

    print('Create dataloader')
    DG = DataGenerator_Synthia_car_trajectory(cf)

    print('\n > Building model...')
    model = Model_Factory_LSTM(cf)
    #model.test(cf, DG.valid_loader, DG.data_mean, DG.data_std, 0)
    model.test(cf, DG.test_loader, DG.data_mean, DG.data_std, epoch=None)
    if cf.train_model:
        train_losses = []
        valid_losses = []
        print(' ---> Training data: ' + cf.sequence_name + ' <---')
        for epoch in range(1, cf.n_epochs + 1):
            train_losses += [model.train(cf, DG.train_loader, epoch, train_losses)]
            if cf.valid_model:
                valid_losses += [model.test(cf, DG.valid_loader, DG.data_mean, DG.data_std, epoch)]
                if epoch > 0 and epoch%cf.figure_epoch == 0:
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
                    figure_name = 'loss_figure_' + str(epoch) + '.jpg'
                    figure_path = os.path.join(model.exp_dir, figure_name)
                    plt.savefig(figure_path)

    if cf.test_model:
        test_loss = model.test(cf, DG.test_loader, DG.data_mean, DG.data_std, epoch=None)
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
