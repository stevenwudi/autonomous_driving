from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib
import numpy as np
import cv2
import os
import imutils
import json


def convert_key_to_string(car_tracking_dict):
    # json file key only allow string...
    for key in car_tracking_dict.keys():
        if type(key) is not str:
            car_tracking_dict[str(key)] = car_tracking_dict[key]
            del car_tracking_dict[key]

    return car_tracking_dict


def convert_list_to_json_serializable(car_tracking_dict):
    # because stupid reason the saved list data type can not be dumped directly to json
    for key in car_tracking_dict.keys():
        for i, track_list in enumerate(car_tracking_dict[key]['tracking_rect']):
            converted_list = []
            for l in track_list:
                converted_list.append([np.int(i) for i in l])
            car_tracking_dict[str(key)]['tracking_rect'][i] = converted_list

    return car_tracking_dict


def get_ground_truth_box_centre_of_mass(cf, instances, classes):
    # TODO: extract instance for LSTM training
    # now we count the number of pixels for each instances
    mask = instances.astype('uint8')
    classes = classes.astype('uint8')
    #plt.figure(1);plt.imshow(mask);plt.figure(2);plt.imshow(classes);plt.show()
    unique_labels, counts_label = np.unique(mask, return_counts=True)
    instance_percentage = counts_label / np.prod(cf.resize_train)
    # we only count if the sum of the pixels are larger than 1e-3
    instance_large = unique_labels[instance_percentage > cf.threshold_car_POR_start]

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
        if classes[cY, cX] == cf.classes['car']:
            tracking_figure_axes.annotate(['car' for i in pixel_rate],
                                          xy=(cX - int(width / 2), cY - int(height / 2)-10), color='red')

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


def get_ground_truth_box(cf, instances, classes, car_tracking_dict, i_frame, img_name, draw_flag=False):
    # now we count the number of pixels for each instances
    unique_labels, counts_label = np.unique(instances, return_counts=True)
    instance_percentage = counts_label / np.prod(cf.resize_train)
    # we only count if the sum of the pixels are larger than 1e-3
    instance_large = unique_labels[instance_percentage > cf.threshold_car_POR_start]
    instance_median = unique_labels[instance_percentage > cf.threshold_car_POR_end]
    # Create figure and axes
    if draw_flag:
        fig, tracking_figure_axes = plt.subplots(1)
        tracking_figure_axes.imshow(instances)

    for instance_number_in_seq in instance_median[1:]:
        instance_image = instances == instance_number_in_seq
        thresh = instance_image.astype('uint8')
        nonzeroX, nonzeroY = np.nonzero(thresh)
        bottom, right = nonzeroX.max(), nonzeroY.max()
        top, left = nonzeroX.min(), nonzeroY.min()
        width = right - left
        height = bottom - top
        centreX, centreY = int((top+bottom)/2), int((left+right)/2)
        if classes[centreX, centreY] == cf.classes['car']:
            if draw_flag:
                fig, tracking_figure_axes = plt.subplots(1)
                tracking_figure_axes.imshow(instances)
                tracking_rect = Rectangle(
                    xy=(centreY - int(width / 2), centreX - int(height / 2)),
                    width=width,
                    height=height,
                    facecolor='none',
                    edgecolor='r',
                )
                tracking_figure_axes.add_patch(tracking_rect)
                pixel_occupant_rate = instance_percentage[unique_labels == instance_number_in_seq]
                tracking_figure_axes.annotate(['{:.4f}'.format(i) for i in pixel_occupant_rate],
                                              xy=(centreY, centreX), color='red')
                tracking_figure_axes.annotate('car', xy=(centreY, centreX - 10), color='red')
                plt.show()
            # if it is an already tracked car:
            if instance_number_in_seq in car_tracking_dict:
                # print('continue tracking the car')
                if int(img_name[:-4]) == (int(car_tracking_dict[instance_number_in_seq]['img_list'][-1][-1][:-4]) + 1):
                    car_tracking_dict[instance_number_in_seq]['tracking_rect'][-1].append([centreX, centreY, height, width])
                    car_tracking_dict[instance_number_in_seq]['img_list'][-1].append(img_name)
                else:
                    #print("Tracked car not continuous! Investigate! image name: %s, car instance: %d" % (img_name, instance_number_in_seq))
                    if instance_number_in_seq in instance_large:
                        print("Tracked car not continuous! Start another track for this car")
                        print("image name: %s, car instance: %d" % (img_name, instance_number_in_seq))
                        car_tracking_dict[instance_number_in_seq]['start_frame'].append(i_frame)
                        car_tracking_dict[instance_number_in_seq]['img_list'].append([])
                        car_tracking_dict[instance_number_in_seq]['img_list'][-1].append(img_name)
                        car_tracking_dict[instance_number_in_seq]['tracking_rect'].append([])
                        car_tracking_dict[instance_number_in_seq]['tracking_rect'][-1].append([centreX, centreY, height, width])

            # else we initiate a instance_number_in_seq as car ID to start tracking
            else:
                if instance_number_in_seq in instance_large:
                    print('start tracking a new car %d' % instance_number_in_seq)
                    car_tracking_dict[instance_number_in_seq] = {}
                    car_tracking_dict[instance_number_in_seq]['dir'] = os.path.join(cf.dataset_path, cf.data_type, cf.data_stereo, cf.data_camera)
                    car_tracking_dict[instance_number_in_seq]['start_frame'] = []
                    car_tracking_dict[instance_number_in_seq]['start_frame'].append(i_frame)
                    car_tracking_dict[instance_number_in_seq]['img_list'] = [[]]
                    car_tracking_dict[instance_number_in_seq]['img_list'][0].append(img_name)
                    car_tracking_dict[instance_number_in_seq]['tracking_rect'] = [[]]
                    car_tracking_dict[instance_number_in_seq]['tracking_rect'][0].append([centreX, centreY, height, width])

    return car_tracking_dict


def get_ground_truth_sequence_car_trajectory(DG, cf, show_set='train'):
    """
     # Let's instantiate this class the iterate through the data samples.
    # For the paper "End-to-end Learning of Driving Models from Large-scale Video Datasets" cvpr2017 (oral)
    they use 3 seconds(9 frames) to predict  1 frame (3Hz)
    Here, Synthia dataset is 5 Hz.
    According to http://copradar.com/redlight/factors/, we need to predict next 1.5 second
    Hence it's 1.5*5 ~ 8 frames.
    We will use 3*5=15 to predict next 8 frames.
    :param DG:
    :param show_set:
    :return:
    """

    train_set = DG.dataloader[show_set]
    car_tracking_dict = {}
    for i_batch, sample_batched in enumerate(train_set):
        if i_batch == 0:
            print(sample_batched['image'].size(),
                  sample_batched['classes'].size(),
                  sample_batched['instances'].size())
        if i_batch % 100 == 0:
            print("Processing batch: %d" % i_batch)

        instances_torch = sample_batched['instances'][0]
        classes_torch = sample_batched['classes'][0]
        instances = instances_torch.numpy().astype('uint8')
        classes = classes_torch.numpy().astype('uint8')
        img_name = sample_batched['img_name'][0]  # because for GT construction, the batch size is always 1
        car_tracking_dict = get_ground_truth_box(cf, instances, classes, car_tracking_dict, i_batch, img_name)

    json_file_path = os.path.join(cf.savepath, cf.sequence_name+'.json')

    # Di Wu's comment: json key must be string, duh....
    car_tracking_dict = convert_key_to_string(car_tracking_dict)
    car_tracking_dict = convert_list_to_json_serializable(car_tracking_dict)
    with open(json_file_path, 'w') as fp:
        json.dump(car_tracking_dict, fp, indent=4)

    return car_tracking_dict


def draw_selected_gt_car_trajectory(DG, cf, show_set='train'):
    """
    This script is used to visualise the collected GT
    :param DG:
    :param cf:
    :return:
    """
    json_file_path = os.path.join(cf.savepath, cf.sequence_name + '.json')
    with open(json_file_path) as json_data:
        car_tracking_dict = json.load(json_data)

    print("car instance ids are: " + ', '.join(['%s' % key for key in car_tracking_dict.keys()]))
    print('Total number of cars is %d.' % len(car_tracking_dict.keys()))
    cm_range = np.array([int(x) for x in car_tracking_dict.keys()])
    norm = matplotlib.colors.Normalize(vmin=cm_range.min(), vmax=cm_range.max())
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig, tracking_figure_axes = plt.subplots()

    train_set = DG.dataloader[show_set]
    for i_batch, sample_batched in enumerate(train_set):
        instances_torch = sample_batched['instances'][0]
        classes_torch = sample_batched['classes'][0]
        instances = instances_torch.numpy().astype('uint8')
        classes = classes_torch.numpy().astype('uint8')
        img_name = sample_batched['img_name'][0]  # because for GT construction, the batch size is always 1

        draw_ground_truth_box(tracking_figure_axes, m, cf, instances, classes, car_tracking_dict, i_batch, img_name)


def draw_ground_truth_box(tracking_figure_axes, m, cf, instances, classes, car_tracking_dict, i_batch, img_name):
    tracking_figure_axes.clear()
    tracking_figure_axes.imshow(instances)
    unique_labels, counts_label = np.unique(instances, return_counts=True)
    instance_percentage = counts_label / np.prod(cf.resize_train)

    for instance_number_in_seq in car_tracking_dict.keys():
        car_color = m.to_rgba(int(instance_number_in_seq))
        for track_num, img_lists_all in enumerate(car_tracking_dict[instance_number_in_seq]['img_list']):
            if img_name in img_lists_all:
                im_num = img_lists_all.index(img_name)
                [centreX, centreY, height, width] = car_tracking_dict[instance_number_in_seq]['tracking_rect'][track_num][im_num]
                tracking_rect = Rectangle(
                    xy=(centreY - int(width / 2), centreX - int(height / 2)),
                    width=width,
                    height=height,
                    facecolor='none',
                    edgecolor=car_color,
                )
                tracking_figure_axes.add_patch(tracking_rect)
                pixel_occupant_rate = instance_percentage[unique_labels == int(instance_number_in_seq)]
                tracking_figure_axes.annotate('{:.4f}'.format(pixel_occupant_rate[0]),
                                              xy=(centreY - int(width / 2), centreX), color=car_color)
                car_id = instance_number_in_seq + '_' + str(0)
                tracking_figure_axes.annotate(car_id, xy=(centreY - int(width / 2), centreX - int(height / 2) - 5), color=car_color)
    plt.title(img_name)
    plt.waitforbuttonpress(0.01)

