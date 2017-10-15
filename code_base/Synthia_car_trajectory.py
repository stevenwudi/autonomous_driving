import argparse
import os
import sys
import numpy as np
import cv2
import imutils
from scipy import ndimage
# Di Wu add the following really ugly code so that python can find the path
sys.path.append(os.getcwd())
import time
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
#from code_base.models.PyTorch_model_factory import Model_Factory
from code_base.tools.PyTorch_data_generator_car_trajectory import Dataset_Generators_Synthia_Car_trajectory
from code_base.config.configuration import Configuration
from code_base.utils import show_DG_car_trajectory, HMS, configurationPATH


def process(cf):
    # Create the data generators
    DG = Dataset_Generators_Synthia_Car_trajectory(cf)
    instances, classes = show_DG_car_trajectory(DG)  # this script will draw an image

    # TODO: extract instance for LSTM training
    # now we count the number of pixels for each instances
    mask = instances.astype('uint8')
    classes = classes.astype('uint8')
    #plt.figure(1);plt.imshow(mask);plt.figure(2);plt.imshow(classes);plt.show()
    unique_labels, counts_label = np.unique(mask, return_counts=True)
    instance_percentage = counts_label / np.prod(cf.resize_train)
    # we only count if the sum of the pixels are larger than 1e-3
    instance_large = unique_labels[instance_percentage > cf.threshold_car_pixel]

    # Create figure and axes
    fig, tracking_figure_axes = plt.subplots(1)
    tracking_figure_axes.imshow(mask)

    for i in instance_large[1:]:
        instance_image = mask == i
        # https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.measurements.center_of_mass.html
        # now we start cleaning the instance, the first instance is the background and we will ignore it
        thresh = instance_image.astype('uint8')
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        # loop over the contours
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            plt.scatter(cX, cY, c='r', s=40)

            contour = np.squeeze(np.asarray(c))
            top, left = contour[:, 0].min(), contour[:, 1].min()
            bottom, right = contour[:, 0].max(), contour[:, 1].max()
            height = right - left
            width = bottom - top
            tracking_rect = Rectangle(
                xy=(cX-int(width/2), cY-int(height/2)),
                width=width,
                height=height,
                facecolor='none',
                edgecolor='r',
            )
        tracking_figure_axes.add_patch(tracking_rect)
        pixel_rate = instance_percentage[unique_labels==i]
        tracking_figure_axes.annotate(['{:.4f}'.format(i) for i in pixel_rate], xy=(cX-int(width/2), cY-int(height/2)), color='red')


    # we plot the annotate
    for i in unique_labels:
        if not i in instance_large:
            instance_image = mask == i
            # https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.measurements.center_of_mass.html
            # now we start cleaning the instance, the first instance is the background and we will ignore it
            thresh = instance_image.astype('uint8')
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            # loop over the contours
            for c in cnts:
                # compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                plt.scatter(cX, cY, c='g', s=40)

                contour = np.squeeze(np.asarray(c))
                top, left = contour[:, 0].min(), contour[:, 1].min()
                bottom, right = contour[:, 0].max(), contour[:, 1].max()
                height = right - left
                width = bottom - top
                tracking_rect = Rectangle(
                    xy=(cX - int(width / 2), cY - int(height / 2)),
                    width=width,
                    height=height,
                    facecolor='none',
                    edgecolor='g',
                )
            tracking_figure_axes.add_patch(tracking_rect)
            pixel_rate = instance_percentage[unique_labels == i]
            tracking_figure_axes.annotate(['{:.4f}'.format(i) for i in pixel_rate],
                                          xy=(cX - int(width / 2), cY - 20 - int(height / 2)), color='green')

    plt.show()



    #
    # Build model
    print('\n > Building model...')

    # Finish
    print(' ---> Finish experiment: ' + cf.exp_name + ' <---')


def main():
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str, default='/home/stevenwudi/PycharmProjects/autonomous_driving/code_base/config/synthia_car_trajectory.py', help='Configuration file')
    parser.add_argument('-s', '--shared_path', type=str, default='/home/public', help='Path to shared data folder')
    parser.add_argument('-l', '--local_path', type=str, default='/home/stevenwudi/PycharmProjects/autonomous_driving', help='Path to local data folder')
    parser.add_argument('-f', '--sequence_name', type=str, default='SYNTHIA-SEQS-01-SPRING')
    arguments = parser.parse_args()
    assert arguments.config_path is not None, 'Please provide a path using -c config/pathname in the command line'    # Start Time
    print('\n > Start Time:')
    print('   ' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    start_time = time.time()
    # Define the user paths
    shared_path = arguments.shared_path
    local_path = arguments.local_path
    sequence_name = arguments.sequence_name
    dataset_path = os.path.join(local_path, 'Datasets')
    shared_dataset_path = os.path.join(shared_path)
    experiments_path = os.path.join(local_path, 'Experiments')
    shared_experiments_path = os.path.join(shared_path, 'Experiments')
    # Load configuration files
    configuration = Configuration(arguments.config_path, None,
                                  dataset_path, shared_dataset_path,
                                  experiments_path, shared_experiments_path,
                                  sequence_name)
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
