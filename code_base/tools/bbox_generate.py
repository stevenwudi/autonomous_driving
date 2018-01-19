import numpy as np
import cv2
import os

def get_ground_truth_box(instances, classes):
    # now we count the number of pixels for each instances
    resize_train = (760, 1280)
    threshold_car_POR_end = 1e-3
    car = 8
    unique_labels, counts_label = np.unique(instances, return_counts=True)
    instance_percentage = counts_label / np.prod(resize_train)
    # we only count if the sum of the pixels are larger than 1e-3
    # instance_large = unique_labels[instance_percentage > cf.threshold_car_POR_start]
    instance_median = unique_labels[instance_percentage > threshold_car_POR_end]
    # Create car bounding box x, y, w, h
    car_bbox = []

    for instance_number_in_seq in instance_median[1:]:
        instance_image = instances == instance_number_in_seq
        thresh = instance_image.astype('uint8')
        nonzeroX, nonzeroY = np.nonzero(thresh)
        bottom, right = nonzeroX.max(), nonzeroY.max()
        top, left = nonzeroX.min(), nonzeroY.min()
        width = right - left
        height = bottom - top
        centreX, centreY = int((top + bottom) / 2), int((left + right) / 2)
        if classes[centreX, centreY] == car:
            car_bbox.append([centreY - int(width / 2), centreX - int(height / 2), int(right), int(bottom)])

    return car_bbox





if __name__ == '__main__':

    folders = ['SYNTHIA-SEQS-01-DAWN', 'SYNTHIA-SEQS-01-FALL', 'SYNTHIA-SEQS-01-FOG',
               'SYNTHIA-SEQS-01-NIGHT', 'SYNTHIA-SEQS-01-SPRING', 'SYNTHIA-SEQS-01-SUMMER',
               'SYNTHIA-SEQS-01-SUNSET', 'SYNTHIA-SEQS-01-WINTER', 'SYNTHIA-SEQS-01-WINTERNIGHT'
               ]
    car_dict_list_train = []
    car_dict_list_validate = []
    car_dict_list_test = []
    for folder in folders:
        gt_dir = '/home/wzn/PycharmProjects/autonomous_driving/Datasets/' + folder +'/GT/LABELS/Stereo_Left/Omni_F'
        data_dir = '/home/wzn/PycharmProjects/autonomous_driving/Datasets/' + folder +'/RGB/Stereo_Left/Omni_F'
        images = os.listdir(gt_dir)
        images.sort()
        folder_car_dict_list = []
        import json

        for img in images:
            image_path = os.path.join(gt_dir, img)
            label = cv2.imread(image_path, -1)
            classes = np.uint8(label[:, :, 2])
            instances = np.uint8(label[:, :, 1])
            car_bbox = get_ground_truth_box(instances, classes)
            car_dict = {}
            # print(car_bbox)
            if len(car_bbox) > 0:
                car_dict['image_path'] = data_dir
                car_dict['image_name'] = img
                car_dict['boundingbox'] = car_bbox
                # print (type(car_bbox[0][0]))
                folder_car_dict_list.append(car_dict)

                # print (car_dict_list)

        num = len(folder_car_dict_list)
        train_size = int(0.8 * num)
        test_size = int(0.1 * num)
        car_dict_list_train += folder_car_dict_list[:train_size]
        car_dict_list_validate += folder_car_dict_list[train_size:(train_size + test_size)]
        car_dict_list_test += folder_car_dict_list[(train_size + test_size):]
        # print ('train_size:', train_size)
        # print('test_size:', test_size)

    print('train num', len(car_dict_list_train))
    print('validate num', len(car_dict_list_validate))
    print('test num', len(car_dict_list_test))
    json_file_train_path = os.path.join('/home/public/synthia', 'SYNTHIA-SEQS-01-TRAIN-shuffle_0.001.json')
    with open(json_file_train_path, 'w') as fp:
        json.dump(car_dict_list_train, fp, indent=4)

    json_file_validate_path = os.path.join('/home/public/synthia', 'SYNTHIA-SEQS-01-VALIDATE-shuffle_0.001.json')
    with open(json_file_validate_path, 'w') as fp:
        json.dump(car_dict_list_validate, fp, indent=4)

    json_file_test_path = os.path.join('/home/public/synthia', 'SYNTHIA-SEQS-01-TEST-shuffle_0.001.json')
    with open(json_file_test_path, 'w') as fp:
        json.dump(car_dict_list_test, fp, indent=4)
