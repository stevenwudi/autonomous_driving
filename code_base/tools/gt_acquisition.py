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
from code_base.tools.PyTorch_data_generator_car_trajectory import Dataset_Generators_Synthia_Car_trajectory, Dataset_Generators_Synthia_Car_trajectory_NEW
import pickle
from scipy.misc import imresize
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen
from skimage.filters import threshold_li
import collections


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
                        feature_img_name = sequence_name + '/' + car_tracking_dict[instance_number_in_seq]['img_list'][track_num][frame_num]
                        [centreX, centreY, height, width, d_min, d_max, d_centre, d_mean] = \
                        car_tracking_dict[instance_number_in_seq]['tracking_rect'][track_num][frame_num]
                        features_temp.append([centreX, centreY, height, width, d_min, d_max, feature_img_name])
                    #features.append(np.asarray(features_temp))
                    features.append(features_temp)

    #features = np.asarray(features)
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
        if cf.formatting_ground_truth_sequence_car_trajectory:
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
            import pickle
            with open(os.path.join(save_dir, cf.sequence_name + "_train.npy"), 'wb') as fp:
                pickle.dump(train_data, fp)
            with open(os.path.join(save_dir, cf.sequence_name + "_valid.npy"), 'wb') as fp:
                pickle.dump(valid_data, fp)
            with open(os.path.join(save_dir, cf.sequence_name + "_test.npy"), 'wb') as fp:
                pickle.dump(test_data, fp)
        # f = h5py.File(os.path.join(save_dir, cf.sequence_name + ".hdf5"), "w")
        # f.create_dataset('train_data', data=[x.encode('utf8') for x in train_data])
        # f.create_dataset('valid_data', data=valid_data)
        # f.create_dataset('test_data', data=test_data)
        # f.close()
    # Finish
    # Total train length is 10291, valid length is 1275, test length is 1294.
    print(' ---> Finish experiment: ' + cf.exp_name + ' <---')


def video_sequence_prediction(cf, model):
    # Create the data generators
    cf.batch_size_train = 1  # because we want to read every single image sequentially
    dataset_path_list = cf.dataset_path
    processed_list = os.listdir(cf.seg_img_path)
    for dataset_path in dataset_path_list:
        sequence_name = dataset_path.split('/')[-1]
        cf.dataset_path = dataset_path
        DG = Dataset_Generators_Synthia_Car_trajectory_NEW(cf)
        train_set = DG.dataloader['train']
        model.test_frame(train_set, cf, sequence_name)

        if sequence_name == cf.draw_seq:
            draw_selected_gt_car_trajectory(DG, cf, sequence_name, draw_image='image')

    # Total train length is 10291, valid length is 1275, test length is 1294.
    print(' ---> Finish experiment: ' + cf.exp_name + ' <---')


def car_detection(cf):
    # Slow import from below!
    from code_base.models.Keras_SSD import SSD512v2, BBoxUtility

    # Create the data generators
    cf.batch_size_train = 1  # because we want to read every single image sequentially
    dataset_path_list = cf.dataset_path
    processed_list = os.listdir(cf.savepath)
    for dataset_path in dataset_path_list:
        sequence_name = dataset_path.split('/')[-1]
        cf.dataset_path = dataset_path
        DG = Dataset_Generators_Synthia_Car_trajectory(cf)
        if cf.get_sequence_car_detection:
            # now we count the number of pixels for each instances
            priors = pickle.load(open(cf.ssd_prior_boxes, 'rb'), encoding='latin1')
            bbox_util = BBoxUtility(cf.ssd_number_classes, priors, nms_thresh=0.45)
            ### load model ###
            model = SSD512v2(cf.ssd_input_shape, num_classes=cf.ssd_number_classes)
            model.load_weights(cf.ssd_model_checkpoint, by_name=True)

            if sequence_name+'.json' not in processed_list:
                get_sequence_car_detection(DG, cf, sequence_name, model, bbox_util)
    print(' ---> Finish experiment: ' + cf.exp_name + ' <---')


def get_sequence_car_detection(DG, cf, sequence_name, model, bbox_util, show_set='train'):
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
    import collections
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
        image_torch = sample_batched['image'][0]
        input_image = image_torch.numpy().astype('uint8')
        instances = instances_torch.numpy().astype('uint8')
        classes = classes_torch.numpy().astype('uint8')
        depth = depth_torch.numpy()
        img_name = sample_batched['img_name'][0]  # because for GT construction, the batch size is always 1
        car_tracking_dict = get_car_detection_box(cf, input_image, car_tracking_dict, img_name, model, bbox_util)

    # we sort the dict by keys
    car_tracking_dict_sorted = collections.OrderedDict(sorted(car_tracking_dict.items()))
    json_file_path = os.path.join(cf.savepath, sequence_name+'.json')
    with open(json_file_path, 'w') as fp:
        json.dump(car_tracking_dict_sorted, fp, indent=4)
    print('Saving: '+json_file_path)
    return car_tracking_dict


def get_car_detection_box(cf, input_image, car_tracking_dict, img_name, model, bbox_util):
    from keras.applications.imagenet_utils import preprocess_input

    ssd_input = imresize(input_image, cf.ssd_input_shape[:2])
    inputs = preprocess_input(np.expand_dims(ssd_input, axis=0).astype('float64'))
    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)
    detected_rect = []
    if len(results[0]):
        # TODO: check which image has no detection
        det_conf = results[0][:, 1]
        det_xmin = results[0][:, 2]
        det_ymin = results[0][:, 3]
        det_xmax = results[0][:, 4]
        det_ymax = results[0][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= cf.ssd_conf]
        top_conf = det_conf[top_indices]
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for j in range(len(top_indices)):
            detected_rect.append([top_xmin[j], top_ymin[j], top_xmax[j], top_ymax[j], top_conf[j]])

    # Create figure and axes
    if cf.draw_flag:
        fig = plt.figure(1)
        fig.clear()
        tracking_figure_axes = fig.add_subplot(111, aspect='equal')
        tracking_figure_axes.imshow(input_image.transpose(1, 2, 0))
        _, W, H = input_image.shape
        for dr in detected_rect:
            x1, y1, x2, y2, _ = dr
            x1_i, y1_i, x2_i, y2_i = x1 * H, y1 * W, x2 * H, y2 * W
            tracking_rect = Rectangle(
                xy=(x1_i, y1_i),
                width=x2_i - x1_i,
                height=y2_i - y1_i,
                facecolor='none',
                edgecolor='g',
            )
            tracking_figure_axes.add_patch(tracking_rect)
        plt.waitforbuttonpress(0.01)

    car_tracking_dict[img_name] = detected_rect
    return car_tracking_dict


def car_tracking(cf):
    # Create the data generators
    cf.batch_size_train = 1  # because we want to read every single image sequentially
    dataset_path_list = cf.dataset_path
    processed_list = os.listdir(cf.savepath)
    for dataset_path in dataset_path_list:
        sequence_name = dataset_path.split('/')[-1]
        cf.dataset_path = dataset_path
        DG = Dataset_Generators_Synthia_Car_trajectory(cf)
        if cf.get_sequence_car_tracking:
            # now we count the number of pixels for each instances
            if sequence_name+'.json' not in processed_list:
                get_sequence_car_tracking(DG, cf, sequence_name)
    print(' ---> Finish experiment: ' + cf.exp_name + ' <---')


def get_sequence_car_tracking(DG, cf, sequence_name, show_set='train'):
    """
    We track all the frames with car larger than 0.2% Pixel Occupancy Rate
    :param DG:
    :param show_set:
    :return:
    """
    print("Processing sequence: "+cf.dataset_path)
    train_set = DG.dataloader[show_set]
    if cf.tracker == 'fDSST':
        import time
        from code_base.tools.bmvc_2014_pami_2014_fDSST import bmvc_2014_pami_2014_fDSST
        print('starting matlab engine...')
        start_time = time.time()
        import matlab.engine
        import matlab
        matlab_eng = matlab.engine.start_matlab("-nojvm -nodisplay -nosplash")
        matlab_eng.addpath('./tools/piotr_hog_matlab')
        total_time = time.time() - start_time
        print('matlab engine started, used %.2f second'%(total_time))

    car_detection_path_list = cf.savepath.split('/')
    car_detection_path_list[-2] = 'car_detection'  # this is a hard coded script, sorry
    with open(os.path.join('/'.join(car_detection_path_list), sequence_name+'.json'), 'r') as fp:
        car_detect_dict = json.load(fp)

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
        image_torch = sample_batched['image'][0]
        input_image = image_torch.numpy().astype('uint8')
        instances = instances_torch.numpy().astype('uint8')
        classes = classes_torch.numpy().astype('uint8')
        depth = depth_torch.numpy()
        img_name = sample_batched['img_name'][0]  # because for GT construction, the batch size is always 1
        if cf.tracker == 'fDSST':
            car_tracking_dict = get_car_tracking_box(cf, input_image, depth, car_tracking_dict, img_name,
                                                     car_detect_dict, matlab_eng, bmvc_2014_pami_2014_fDSST)
        else:
            car_tracking_dict = get_car_tracking_box(cf, input_image, depth, car_tracking_dict, img_name, car_detect_dict)

    # we sort the dict by keys
    for key in car_tracking_dict.keys():
        del car_tracking_dict[key]['tracker']

    npy_file_path = os.path.join(cf.savepath, sequence_name)
    np.save(npy_file_path, car_tracking_dict)

    return car_tracking_dict


def get_car_tracking_box(cf, input_image, depth, car_tracking_dict, img_name, car_detect_dict, matlab_eng=None,
                         bmvc_2014_pami_2014_fDSST=None):
    from code_base.tools.yolo_utils import box_iou, BoundBox
    if cf.tracker == 'dlib_dsst':
        import dlib
    elif cf.tracker == 'ECO_HC':
        import time
        import matlab.engine
        tic = time.time()
        # Start MATLAB Engine for Python
        print("start up matlab engine")
        eng = matlab.engine.start_matlab()
        eng.addpath('./ECO')
        print("Elapsed time is " + str(time.time() - tic) + " seconds.")
        tic = time.time()
    elif cf.tracker == 'KCF':
        # NOT implemented yet
        import cv2
        KCF_tracker = cv2.Tracker_create('KCF')

    h, w = depth.shape
    # this is a brand new sequence with brand new car to track
    if len(car_tracking_dict) == 0 and img_name in car_detect_dict.keys():
        CAR_ID = 0
        detected_boundingboxes = car_detect_dict[img_name]
        for bb in detected_boundingboxes:
            xmin, ymin, xmax, ymax, _ = bb
            h, w = depth.shape
            xdmin, ydmin, xdmax, ydmax = max(0, int(xmin*w)), max(0, int(ymin*h)), min(w, int(xmax*w)), min(h, int(ymax*h))

            if (ymax - ymin) * (xmax - xmin) > cf.threshold_car_POR_start:
                print('we start a new tracking list')
                car_tracking_dict[CAR_ID] = {}
                car_tracking_dict[CAR_ID]['start_frame'] = img_name
                car_tracking_dict[CAR_ID]['end_frame'] = None
                car_tracking_dict[CAR_ID]['tracking_rect'] = []
                car_tracking_dict[CAR_ID]['img_list'] = []
                car_tracking_dict[CAR_ID]['img_list'].append(img_name)
                car_tracking_dict[CAR_ID]['last_matched_detection_frame'] = img_name
                # we use: http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_thresholding.html
                # for depth thresholding
                if cf.depth_threshold_method == 'yen':
                    thresh = threshold_yen(depth[ydmin:ydmax, xdmin:xdmax])
                elif cf.depth_threshold_method == 'ostu':
                    thresh = threshold_otsu(depth[ydmin:ydmax, xdmin:xdmax])
                elif cf.depth_threshold_method == 'li':
                    thresh = threshold_li(depth[ydmin:ydmax, xdmin:xdmax])
                car_mask = depth[ydmin:ydmax, xdmin:xdmax] < thresh
                depth_min = depth[ydmin:ydmax, xdmin:xdmax][car_mask].min()
                depth_max = depth[ydmin:ydmax, xdmin:xdmax][car_mask].max()
                car_tracking_dict[CAR_ID]['tracking_rect'].append([xmin, ymin, xmax, ymax, depth_min, depth_max])
                if cf.tracker == 'dlib_dsst':
                    car_tracking_dict[CAR_ID]['tracker'] = dlib.correlation_tracker()
                    car_tracking_dict[CAR_ID]['tracker'].start_track(input_image.transpose(1, 2, 0), dlib.rectangle(xdmin, ydmin, xdmax, ydmax))
                elif cf.tracker == 'KCF':
                    bbox = (xdmin, ydmin, xdmax-xdmin, ydmax-ydmin)
                    car_tracking_dict[CAR_ID]['tracker'] = KCF_tracker.init(input_image.transpose(1, 2, 0), bbox)
                elif cf.tracker == 'fDSST':
                    w, h = xdmax-xdmin, ydmax-ydmin
                    bbox = [xdmin, ydmin, w, h]
                    car_tracking_dict[CAR_ID]['tracker'] = bmvc_2014_pami_2014_fDSST(number_of_scales=17,
                                                                                     padding=2.0,
                                                                                     interpolate_response=True,
                                                                                     kernel='linear',
                                                                                     compressed_features='gray_hog',
                                                                                     matlab_eng=matlab_eng)
                    car_tracking_dict[CAR_ID]['tracker'].train(input_image.transpose(1, 2, 0), bbox)

                CAR_ID += 1

    else:
        boxes_true = []
        boxes_pred = []
        for car_idx in car_tracking_dict.keys():
            bx = BoundBox(1)
            if not car_tracking_dict[car_idx]['end_frame']:
                if cf.tracker == 'dlib_dsst':
                    car_tracking_dict[car_idx]['tracker'].update(input_image.transpose(1, 2, 0))
                    pos = car_tracking_dict[car_idx]['tracker'].get_position()
                    bx.x, bx.y, bx.c = pos.left(), pos.top(), car_idx
                    bx.w, bx.h = pos.width(), pos.height()
                elif cf.tracker == 'fDSST':
                    bbox = car_tracking_dict[car_idx]['tracker'].detect(input_image.transpose(1, 2, 0), 1)
                    bx.x, bx.y, bx.c = bbox[0], bbox[1], car_idx
                    bx.w, bx.h = bbox[2], bbox[3]
                boxes_pred.append(bx)

        if img_name in car_detect_dict.keys():
            detected_boundingboxes = car_detect_dict[img_name]
        else:
            # there is no detected car in this frame
            detected_boundingboxes = []
        for bb in detected_boundingboxes:
            xmin, ymin, xmax, ymax, _ = bb
            xdmin, ydmin, xdmax, ydmax = max(0, int(xmin * w)), max(0, int(ymin * h)), \
                                         min(w, int(xmax * w)), min(h, int(ymax * h))
            gx = BoundBox(1)
            gx.x, gx.y, gx.c = xdmin, ydmin, 1
            gx.w, gx.h = (xdmax - xdmin), (ydmax - ydmin)
            # we count all GT boxes
            boxes_true.append(gx)

        # we check whether the boundingbox overlaps with detection result
        matched_iou = np.zeros(shape=len(boxes_true))
        matched_idx = np.zeros(shape=len(boxes_true))
        for ib, bx in enumerate(boxes_pred):
            for t, gx in enumerate(boxes_true):
                iou_new = box_iou(gx, bx)
                if iou_new > cf.iou_threshold:
                    matched_iou[t] = iou_new
                    matched_idx[t] = bx.c
            if matched_iou.sum() > 0:
                t_idx = np.argmax(matched_iou)
                gx = boxes_true[t_idx]
                print('IOU: %.3f, We update tracked cars: %d' % (matched_iou[t_idx], matched_idx[t_idx]))

                if cf.tracker == 'dlib_dsst':
                    car_tracking_dict[matched_idx[t_idx]]['tracker'].start_track(
                        input_image.transpose(1, 2, 0), dlib.rectangle(gx.x, gx.y, gx.w + gx.x, gx.h + gx.y))
                elif cf.tracker == 'KCF':
                    bbox = (int(gx.x), int(gx.y), int(gx.w), int(gx.h))
                    car_tracking_dict[matched_idx[t_idx]]['tracker'].init(
                        input_image.transpose(1, 2, 0), bbox)
                elif cf.tracker == 'fDSST':
                    bbox = (int(gx.x), int(gx.y), int(gx.w), int(gx.h))
                    car_tracking_dict[matched_idx[t_idx]]['tracker'].train(
                        input_image.transpose(1, 2, 0), bbox)

                car_tracking_dict[matched_idx[t_idx]]['last_matched_detection_frame'] = img_name
                if len(np.nonzero(matched_iou)) > 1:
                    print('More than one overlap of image: %s' % img_name)

        # We check whether there is a new car appear:
        for not_machted_idx, iou_value in enumerate(matched_iou):
            if iou_value == 0:
                gx = boxes_true[not_machted_idx]
                if gx.w * gx.h > cf.threshold_car_POR_start * np.prod(cf.im_size):
                    print('we start a new tracking list')
                    xdmin, ydmin, xdmax, ydmax = gx.x, gx.y, gx.x + gx.w, gx.y + gx.h
                    xmin, ymin, xmax, ymax = xdmin / w, ydmin / h, xdmax / w, ydmax / h
                    CAR_ID = max(list(car_tracking_dict.keys())) + 1
                    car_tracking_dict[CAR_ID] = {}
                    car_tracking_dict[CAR_ID]['start_frame'] = img_name
                    car_tracking_dict[CAR_ID]['end_frame'] = None
                    car_tracking_dict[CAR_ID]['tracking_rect'] = []
                    car_tracking_dict[CAR_ID]['img_list'] = []
                    car_tracking_dict[CAR_ID]['img_list'].append(img_name)
                    car_tracking_dict[CAR_ID]['last_matched_detection_frame'] = img_name
                    # we use: http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_thresholding.html
                    # for depth thresholding
                    if cf.depth_threshold_method == 'yen':
                        thresh = threshold_yen(depth[ydmin:ydmax, xdmin:xdmax])
                    elif cf.depth_threshold_method == 'ostu':
                        thresh = threshold_otsu(depth[ydmin:ydmax, xdmin:xdmax])
                    elif cf.depth_threshold_method == 'li':
                        thresh = threshold_li(depth[ydmin:ydmax, xdmin:xdmax])
                    car_mask = depth[ydmin:ydmax, xdmin:xdmax] < thresh
                    depth_min = depth[ydmin:ydmax, xdmin:xdmax][car_mask].min()
                    depth_max = depth[ydmin:ydmax, xdmin:xdmax][car_mask].max()
                    car_tracking_dict[CAR_ID]['tracking_rect'].append([xmin, ymin, xmax, ymax, depth_min, depth_max])
                    if cf.tracker == 'dlib_dsst':
                        car_tracking_dict[CAR_ID]['tracker'] = dlib.correlation_tracker()
                        car_tracking_dict[CAR_ID]['tracker'].start_track(input_image.transpose(1, 2, 0), dlib.rectangle(xdmin, ydmin, xdmax, ydmax))
                    elif cf.tracker == 'KCF':
                        bbox = (xdmin, ydmin, xdmax-xdmin, ydmax-ydmin)
                        car_tracking_dict[CAR_ID]['tracker'] = KCF_tracker.init(input_image.transpose(1, 2, 0), bbox)
                    elif cf.tracker == 'fDSST':
                        bbox = (xdmin, ydmin, xdmax - xdmin, ydmax - ydmin)
                        car_tracking_dict[CAR_ID]['tracker'] = car_tracking_dict[CAR_ID]['tracker'] = bmvc_2014_pami_2014_fDSST(number_of_scales=17,
                                                                                     padding=2.0,
                                                                                     interpolate_response=True,
                                                                                     kernel='linear',
                                                                                     compressed_features='gray_hog',
                                                                                     matlab_eng=matlab_eng)
                        car_tracking_dict[CAR_ID]['tracker'].train(input_image.transpose(1, 2, 0), bbox)


        # we update the tracked dictionary here:
        for car_idx in car_tracking_dict.keys():
            if not car_tracking_dict[car_idx]['end_frame']:
                if cf.tracker == 'dlib_dsst':
                    pos = car_tracking_dict[car_idx]['tracker'].get_position()
                    xdmin, ydmin, xdmax, ydmax = int(pos.left()), int(pos.top()), int(pos.left() + pos.width()), int(
                        pos.top() + pos.height())

                elif cf.tracker == 'fDSST':
                    bbox = car_tracking_dict[car_idx]['tracker'].detect(input_image.transpose(1, 2, 0), 1)
                    xdmin, ydmin, xdmax, ydmax = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

                xdmin, ydmin, xdmax, ydmax = max(0, xdmin), max(0, ydmin), min(w, xdmax), min(h, ydmax)
                if cf.depth_threshold_method == 'yen':
                    thresh = threshold_yen(depth[ydmin:ydmax, xdmin:xdmax])
                elif cf.depth_threshold_method == 'ostu':
                    thresh = threshold_otsu(depth[ydmin:ydmax, xdmin:xdmax])
                elif cf.depth_threshold_method == 'li':
                    thresh = threshold_li(depth[ydmin:ydmax, xdmin:xdmax])
                car_mask = depth[ydmin:ydmax, xdmin:xdmax] < thresh
                depth_min = depth[ydmin:ydmax, xdmin:xdmax][car_mask].min()
                depth_max = depth[ydmin:ydmax, xdmin:xdmax][car_mask].max()

                xmin, ymin, xmax, ymax  = xdmin/w, ydmin/h, xdmax/w, ydmax/h
                car_tracking_dict[car_idx]['tracking_rect'].append([xmin, ymin, xmax, ymax, depth_min, depth_max])
                car_tracking_dict[car_idx]['img_list'].append(img_name)

                # if there is more than 3 frames we didn't detection such object target, we stop tracking this id
                if int(img_name[:-4]) - int(car_tracking_dict[car_idx]['last_matched_detection_frame'][:-4]) > cf.minimum_detection_length:
                    car_tracking_dict[car_idx]['end_frame'] = img_name

    # We draw the visualisation here
    if cf.draw_flag:
        fig = plt.figure(1)
        fig.clear()
        tracking_figure_axes = fig.add_subplot(111, aspect='equal')
        tracking_figure_axes.set_title('Green: detecion; Red: tracking. Image: %s' % img_name)
        tracking_figure_axes.imshow(input_image.transpose(1, 2, 0))
        for car_idx in car_tracking_dict.keys():
            if not car_tracking_dict[car_idx]['end_frame']:
                if cf.tracker == 'dlib_dsst':
                    pos = car_tracking_dict[car_idx]['tracker'].get_position()
                    tracking_rect = Rectangle(
                        xy=(pos.left(), pos.top()),
                        width=pos.width(),
                        height=pos.height(),
                        facecolor='none',
                        edgecolor='r',
                    )
                    tracking_figure_axes.add_patch(tracking_rect)
                    tracking_figure_axes.annotate(str(car_idx), xy=(pos.left(), pos.top() + 20), color='red')
                elif cf.tracker == 'fDSST':
                    xdmin, ydmin, xdmax, ydmax = int(bbox[0] * w), int(bbox[1] * h), int(bbox[0] + bbox[2]) * w, int(bbox[1] + bbox[3])*h
                    tracking_rect = Rectangle(
                        xy=(int(xdmin), int(ydmin)),
                        width=int(xdmax-xdmin),
                        height=int(ydmax-ydmin),
                        facecolor='none',
                        edgecolor='r',
                    )
                    tracking_figure_axes.add_patch(tracking_rect)
                    tracking_figure_axes.annotate(str(car_idx), xy=(int(bbox[0]), int(bbox[1])+20), color='red')

        # we draw detected boundingboxes:
        detected_boundingboxes = car_detect_dict[img_name]
        for bb in detected_boundingboxes:
            xmin, ymin, xmax, ymax, _ = bb
            h, w = depth.shape
            xdmin, ydmin, xdmax, ydmax = max(0, int(xmin * w)), max(0, int(ymin * h)), \
                                         min(w, int(xmax * w)), min(h, int(ymax * h))
            if (ymax - ymin) * (xmax - xmin) > cf.threshold_car_POR_end:
                tracking_rect = Rectangle(
                    xy=(xdmin, ydmin),
                    width=(xdmax - xdmin),
                    height=(ydmax - ydmin),
                    facecolor='none',
                    edgecolor='g',
                )
                tracking_figure_axes.add_patch(tracking_rect)
                tracking_figure_axes.add_patch(tracking_rect)

        plt.waitforbuttonpress(0.01)

    return car_tracking_dict
