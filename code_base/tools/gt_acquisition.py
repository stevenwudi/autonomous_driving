from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib
import numpy as np
import cv2
import os
import imutils
import json
import h5py
from code_base.tools.PyTorch_data_generator_car_trajectory import Dataset_Generators_Synthia_Car_trajectory


def convert_key_to_string(car_tracking_dict):
    # json file key only allow string...
    # for key in car_tracking_dict.keys():
    #     if type(key) is not str:
    #         car_tracking_dict[str(key)] = car_tracking_dict[key]
    #         del car_tracking_dict[key]

    for key in car_tracking_dict.keys():
        if type(key) is not str:
            try:
                car_tracking_dict[str(key)] = car_tracking_dict[key]
            except:
                try:
                    car_tracking_dict[repr(key)] = car_tracking_dict[key]
                except:
                    pass
            del car_tracking_dict[key]

    return car_tracking_dict


def convert_list_to_json_serializable(car_tracking_dict):
    # because stupid reason the saved list data type can not be dumped directly to json
    for key in car_tracking_dict.keys():
        # if type(key) is not str:
        #     car_tracking_dict[str(key)] = car_tracking_dict[key]
        #     del car_tracking_dict[key]
        for i, track_list in enumerate(car_tracking_dict[key]['tracking_rect']):
            converted_list = []
            for l in track_list:
                converted_list.append([np.int(i) for i in l])
            car_tracking_dict[key]['tracking_rect'][i] = converted_list

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


def get_ground_truth_box(cf, instances, classes, depth, car_tracking_dict, i_frame, img_name, draw_flag=False):
    # now we count the number of pixels for each instances
    unique_labels, counts_label = np.unique(instances, return_counts=True)
    instance_percentage = counts_label / np.prod(cf.resize_train)

    if cf.threshold_car_depth:
        # we calculate the instance depth here. We use the minimum depth
        instance_large = []
        instance_median = []
        for instance_number_in_seq in unique_labels:
            instance_image = instances == instance_number_in_seq
            depth_min = depth[instance_image].min()
            if depth_min < cf.threshold_car_depth_start:
                instance_large.append(instance_number_in_seq)
                instance_median.append(instance_number_in_seq)
            elif depth_min < cf.threshold_car_depth_end:
                instance_median.append(instance_number_in_seq)

        # we only count if the sum of the pixels are larger than 1e-3
        instance_large = np.asarray(instance_large)
        instance_median = np.asarray(instance_median)
    else:
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
        depth_min = depth[instance_image].min()
        depth_max = depth[instance_image].max()
        depth_centre = depth[centreX, centreY]
        depth_mean = depth[instance_image].mean()
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
                tracking_figure_axes.annotate('{:.1f}'.format(pixel_occupant_rate[0]*1000), xy=(centreY, centreX), color='red')
                tracking_figure_axes.annotate('{:d}'.format(int(depth_instance)), xy=(centreY+10, centreX), color='red')
                tracking_figure_axes.annotate('car', xy=(centreY, centreX - 10), color='red')
                plt.show()
            # if it is an already tracked car:
            if cf.threshold_car_depth:
                if instance_number_in_seq in car_tracking_dict:
                    # print('continue tracking the car')
                    if int(img_name[:-4]) == (int(car_tracking_dict[instance_number_in_seq]['img_list'][-1][-1][:-4])+1):
                        car_tracking_dict[instance_number_in_seq]['tracking_rect'][-1].append(
                            [centreX, centreY, height, width, depth_min, depth_max, depth_centre, depth_mean])
                        car_tracking_dict[instance_number_in_seq]['img_list'][-1].append(img_name)
                    else:
                        if instance_number_in_seq in instance_large:
                            print("Tracked car not continuous! Start another track for this car")
                            print("image name: %s, car instance: %d" % (img_name, instance_number_in_seq))
                            car_tracking_dict[instance_number_in_seq]['start_frame'].append(i_frame)
                            car_tracking_dict[instance_number_in_seq]['img_list'].append([])
                            car_tracking_dict[instance_number_in_seq]['img_list'][-1].append(img_name)
                            car_tracking_dict[instance_number_in_seq]['tracking_rect'].append([])
                            car_tracking_dict[instance_number_in_seq]['tracking_rect'][-1].append(
                                [centreX, centreY, height, width, depth_min, depth_max, depth_centre, depth_mean])

                # else we initiate a instance_number_in_seq as car ID to start tracking
                else:
                    if instance_number_in_seq in instance_large:
                        print('start tracking a new car %d' % instance_number_in_seq)
                        car_tracking_dict[instance_number_in_seq] = {}
                        car_tracking_dict[instance_number_in_seq]['dir'] = os.path.join(cf.dataset_path, cf.data_type,
                                                                                        cf.data_stereo, cf.data_camera)
                        car_tracking_dict[instance_number_in_seq]['start_frame'] = []
                        car_tracking_dict[instance_number_in_seq]['start_frame'].append(i_frame)
                        car_tracking_dict[instance_number_in_seq]['img_list'] = [[]]
                        car_tracking_dict[instance_number_in_seq]['img_list'][0].append(img_name)
                        car_tracking_dict[instance_number_in_seq]['tracking_rect'] = [[]]
                        car_tracking_dict[instance_number_in_seq]['tracking_rect'][0].append(
                            [centreX, centreY, height, width, depth_min, depth_max, depth_centre, depth_mean])

            else:
                if instance_number_in_seq in car_tracking_dict:
                    # print('continue tracking the car')
                    if int(img_name[:-4]) == (int(car_tracking_dict[instance_number_in_seq]['img_list'][-1][-1][:-4]) + 1):
                        car_tracking_dict[instance_number_in_seq]['tracking_rect'][-1].append([centreX, centreY, height, width, depth_min, depth_max, depth_centre, depth_mean])
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
                            car_tracking_dict[instance_number_in_seq]['tracking_rect'][-1].append([centreX, centreY, height, width, depth_min, depth_max,depth_centre, depth_mean])

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
                        car_tracking_dict[instance_number_in_seq]['tracking_rect'][0].append([centreX, centreY, height, width, depth_min, depth_max, depth_centre, depth_mean])

    return car_tracking_dict


def get_ground_truth_sequence_car_trajectory(DG, cf, sequence_name, show_set='train'):
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
    print("Processing sequence: "+cf.dataset_path)
    train_set = DG.dataloader[show_set]
    car_tracking_dict = {}
    for i_batch, sample_batched in enumerate(train_set):
        if i_batch == 0:
            print('image, classes, instances and depth sizes are:')
            print(sample_batched['image'].size(),
                  sample_batched['classes'].size(),
                  sample_batched['instances'].size(),
                  sample_batched['depth'].size())
        if i_batch % 100 == 0:
            print("Processing batch: %d" % i_batch)

        instances_torch = sample_batched['instances'][0]
        classes_torch = sample_batched['classes'][0]
        depth_torch = sample_batched['depth'][0]
        instances = instances_torch.numpy().astype('uint8')
        classes = classes_torch.numpy().astype('uint8')
        depth = depth_torch.numpy()
        img_name = sample_batched['img_name'][0]  # because for GT construction, the batch size is always 1
        car_tracking_dict = get_ground_truth_box(cf, instances, classes, depth, car_tracking_dict, i_batch, img_name)

    json_file_path = os.path.join(cf.savepath, sequence_name+'.json')

    # Di Wu's comment: json key must be string, duh....
    car_tracking_dict = convert_list_to_json_serializable(car_tracking_dict)
    car_tracking_dict = convert_key_to_string(car_tracking_dict)
    # Di Wu has no idea why it need to execute twice here, if not, cannot dump into json file
    car_tracking_dict = convert_key_to_string(car_tracking_dict)

    with open(json_file_path, 'w') as fp:
        json.dump(car_tracking_dict, fp, indent=4)

    return car_tracking_dict


def draw_selected_gt_car_trajectory(DG, cf, sequence_name, show_set='train', draw_image='image'):
    """
    This script is used to visualise the collected GT
    :param DG:
    :param cf:
    :return:
    """
    print("Processing sequence: " + cf.dataset_path)
    json_file_path = os.path.join(cf.savepath, sequence_name + '.json')
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
        image_torch = sample_batched['image'][0]
        instances_torch = sample_batched['instances'][0]
        classes_torch = sample_batched['classes'][0]
        image = image_torch.numpy().astype('uint8')
        instances = instances_torch.numpy().astype('uint8')
        classes = classes_torch.numpy().astype('uint8')
        img_name = sample_batched['img_name'][0]  # because for GT construction, the batch size is always 1

        draw_ground_truth_box(tracking_figure_axes, m, cf, image, instances, classes, car_tracking_dict, i_batch, img_name, draw_image)


def draw_ground_truth_box(tracking_figure_axes, m, cf, image, instances, classes, car_tracking_dict, i_batch, img_name,
                          draw_image):
    tracking_figure_axes.clear()
    if draw_image == 'image':
        tracking_figure_axes.imshow(image.transpose(1, 2, 0))
    elif draw_image == 'class':
        tracking_figure_axes.imshow(classes)
    else:
        tracking_figure_axes.imshow(instances)
    unique_labels, counts_label = np.unique(instances, return_counts=True)
    instance_percentage = counts_label / np.prod(cf.resize_train)

    for instance_number_in_seq in car_tracking_dict.keys():
        car_color = m.to_rgba(int(instance_number_in_seq))
        for track_num, img_lists_all in enumerate(car_tracking_dict[instance_number_in_seq]['img_list']):
            if img_name in img_lists_all:
                im_num = img_lists_all.index(img_name)
                [centreX, centreY, height, width, d_min, d_centre, d_mean] = car_tracking_dict[instance_number_in_seq]['tracking_rect'][track_num][im_num]
                tracking_rect = Rectangle(
                    xy=(centreY - int(width / 2), centreX - int(height / 2)),
                    width=width,
                    height=height,
                    facecolor='none',
                    edgecolor=car_color,
                )
                tracking_figure_axes.add_patch(tracking_rect)
                pixel_occupant_rate = instance_percentage[unique_labels == int(instance_number_in_seq)]
                tracking_figure_axes.annotate('{:.1f}'.format(pixel_occupant_rate[0]*1000),
                                              xy=(centreY - int(width / 2), centreX+10), color='red')
                tracking_figure_axes.annotate('{:.1f}'.format(d_min),
                                              xy=(centreY - int(width / 2), centreX-10), color='red')
                car_id = instance_number_in_seq + '_' + str(0)
                tracking_figure_axes.annotate(car_id, xy=(centreY - int(width / 2), centreX - int(height / 2) - 5), color=car_color)
    plt.title(img_name)
    plt.waitforbuttonpress(0.01)


def formatting_ground_truth_sequence_car_trajectory(cf, sequence_name, time_step=1, total_frame=23):
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
    print("Processing sequence: "+cf.dataset_path)

    json_file_path = os.path.join(cf.savepath, sequence_name+'.json')
    with open(json_file_path) as fp:
        car_tracking_dict = json.load(fp)

    features = []
    for instance_number_in_seq in car_tracking_dict.keys():
        for track_num in range(len(car_tracking_dict[instance_number_in_seq]['tracking_rect'])):
            num_tracked_frames = len(car_tracking_dict[instance_number_in_seq]['tracking_rect'][track_num])
            # we only consider tracked cars with more than total_frames (23 frames)
            if num_tracked_frames >= total_frame:
                for im_num in range(0, len(car_tracking_dict[instance_number_in_seq]['tracking_rect'][track_num])-total_frame+1, time_step):
                    features_temp = []
                    for frame_num in range(im_num, im_num+total_frame, time_step):
                        [centreX, centreY, height, width, d_min, d_max, d_centre, d_mean] = \
                        car_tracking_dict[instance_number_in_seq]['tracking_rect'][track_num][frame_num]
                        features_temp.append([centreX, centreY, height, width, d_min, d_max])
                    features.append(np.asarray(features_temp))

    features = np.asarray(features)
    num_total_seq = len(features)
    train_len = int(num_total_seq * 0.8)
    valid_len = int(num_total_seq * 0.1)
    train_features = features[:train_len]
    val_features = features[train_len+1: train_len+valid_len]
    test_features = features[train_len+valid_len:]
    print('Train length is %d, valid length is %d, test length is %d.' % (len(train_features), len(val_features), len(test_features)))
    return train_features, val_features, test_features


def gt_collection_examintion(cf):
    # Create the data generators
    cf.batch_size_train = 1  # because we want to read every single image sequentially
    dataset_path_list = cf.dataset_path
    processed_list = os.listdir(cf.savepath)
    if cf.formatting_ground_truth_sequence_car_trajectory:
        train_features_all, val_features_all, test_features_all = [], [], []
    for dataset_path in dataset_path_list:
        sequence_name = dataset_path.split('/')[-1]
        cf.dataset_path = dataset_path
        DG = Dataset_Generators_Synthia_Car_trajectory(cf)
        if cf.get_ground_truth_sequence_car_trajectory:
            if sequence_name+'.json' not in processed_list:
                get_ground_truth_sequence_car_trajectory(DG, cf, sequence_name)
        elif cf.formatting_ground_truth_sequence_car_trajectory:
            train_features, val_features, test_features = formatting_ground_truth_sequence_car_trajectory(cf,
                                                                                                         sequence_name)
            train_features_all.append(train_features)
            val_features_all.append(val_features)
            test_features_all.append(test_features)
        elif sequence_name == cf.draw_seq:
            draw_selected_gt_car_trajectory(DG, cf, sequence_name, draw_image='image')
    # Build model

    print('\n > Saving training data...')
    if cf.formatting_ground_truth_sequence_car_trajectory:
        train_data = np.vstack(train_features_all)
        valid_data = np.vstack(val_features_all)
        test_data = np.vstack(test_features_all)
        print('Total train length is %d, valid length is %d, test length is %d.' % (
            len(train_data), len(valid_data), len(test_data)))
        save_dir = os.path.join(cf.shared_path, cf.problem_type)
        if not os.path.exists(cf.shared_path):
            os.mkdir(cf.shared_path)
            os.mkdir(save_dir)

        f = h5py.File(os.path.join(save_dir, cf.sequence_name + ".hdf5"), "w")
        f.create_dataset('train_data', data=train_data)
        f.create_dataset('valid_data', data=valid_data)
        f.create_dataset('test_data', data=test_data)
        f.close()
    # Finish
    print(' ---> Finish experiment: ' + cf.exp_name + ' <---')