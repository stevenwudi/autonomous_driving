import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
# Di Wu add the following really ugly code so that python can find the path
sys.path.append(os.getcwd())
import time
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')
from code_base.tools.PyTorch_data_generator import Dataset_Generators_Synthia
from code_base.models.PyTorch_model_factory import Model_Factory_semantic_seg
from code_base.config.configuration import Configuration
from code_base.tools.utils import HMS, configurationPATH
from code_base.tools.gt_acquisition import video_sequence_prediction
from code_base.tools.PyTorch_data_generator_car_trajectory import Dataset_Generators_Synthia_Car_trajectory_segmantic_video


# Train the network
def process(cf):

    print('---> Building model...')
    model = Model_Factory_semantic_seg(cf)

    # Create the data generators
    if not cf.video_sequence_prediction:
        if not cf.video_sequence_train:
            DG = Dataset_Generators_Synthia(cf)
            print('---> Testing and training the model')
            #model.test(DG.val_loader, epoch=0, cf=cf)
            if cf.train_model:
                for epoch in range(1, cf.n_epochs + 1):
                    model.train(DG.train_loader, epoch)
                    if epoch % cf.test_epoch == 0:
                        if cf.test_model:
                            model.test(DG.val_loader, epoch, cf)
        else:
            DG = Dataset_Generators_Synthia_Car_trajectory_segmantic_video(cf)
            # show_DG(DG, 'train')  # this script will draw an image

            print('---> Testing and training the model')
            #model.test(DG.dataloader['valid'], epoch=0, cf=cf)
            if cf.train_model:
                for epoch in range(1, cf.n_epochs + 1):
                    model.train(DG.dataloader['train'], epoch)
                    if epoch % cf.test_epoch == 0:
                        if cf.test_model:
                            model.test(DG.dataloader['valid'], epoch, cf)

    # Finish
    print('---> Test on continuous video sequences: ' + cf.exp_name + ' <---')
    if cf.video_sequence_prediction:
        video_sequence_prediction(cf, model)

    print(' ---> Finish experiment: ' + cf.exp_name + ' <---')


# Main function
def main():
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Semantic segmentation')
    parser.add_argument('-c', '--config_path', type=str, default='/home/stevenwudi/PycharmProjects/autonomous_driving/code_base/config/synthia_segmentation.py')
    arguments = parser.parse_args()
    assert arguments.config_path is not None, 'Please provide a configuration path using -c config/pathname'

    # Start Time
    print('\n > Start Time:')
    print('   ' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    start_time = time.time()

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
    print('\n   ET: ' + HMS(end_time - start_time))  # -> H:M:S


# Entry point of the script
if __name__ == "__main__":
    main()
